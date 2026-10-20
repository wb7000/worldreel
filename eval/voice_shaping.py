"""A-11 measurement: shape TTS lines per emotion and compare with the unshaped lines.

    python -m eval.voice_shaping --src runs/kaggle/voice_acting2/out --pick BV,A --out DIR

For every line of `eval/emotion_lines.json`, take the first usable source among `--pick`
(directories of a voice-acting run: `A/<id>_s*.wav` chosen takes via report.json, or
`BV/<id>.wav`), apply the line's preset scaled by its intensity, and write `unshaped/` and
`shaped/` copies plus `shaping.json` (pitch spread before / after). Speaker identity and Whisper
intelligibility are measured separately in the speaker environment (data/envs/spk).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from stages.generate.voice_quality import PRESETS, shape_voice
from stages.generate.voices import pitch_spread_semitones

LINES = Path("eval/emotion_lines.json")


def source_for(run: Path, report: dict[str, object], system: str, line_id: str) -> Path | None:
    chosen = report.get("chosen", {})
    assert isinstance(chosen, dict)
    # converted folders (BV, BV2) take their usability from the take they were made from (B, B2)
    base = system.replace("V", "", 1) if "V" in system else system
    entry = chosen.get(f"{base}/{line_id}") or chosen.get(f"{system}/{line_id}")
    if isinstance(entry, dict) and not entry.get("usable", False):
        return None
    if "V" in system:
        p = run / system / f"{line_id}.wav"
        return p if p.exists() else None
    if isinstance(entry, dict):
        p = run / str(entry["file"])
        return p if p.exists() else None
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.voice_shaping")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--pick", default="BV,A", help="systems in order of preference")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    report = json.loads((a.src / "report.json").read_text())
    lines = json.loads(LINES.read_text(encoding="utf-8"))["lines"]
    (a.out / "unshaped").mkdir(parents=True, exist_ok=True)
    (a.out / "shaped").mkdir(parents=True, exist_ok=True)
    rows = []
    for ln in lines:
        src = next(
            (p for s in a.pick.split(",") if (p := source_for(a.src, report, s, ln["id"]))), None
        )
        if src is None:
            rows.append({"id": ln["id"], "source": None})
            continue
        fs, x = wavfile.read(src)
        x = x.astype(np.float64) / (32768.0 if x.dtype == np.int16 else 1.0)
        shape = PRESETS.get(ln["emotion"], PRESETS["neutral"]).scaled(ln["intensity"])
        y = shape_voice(x, fs, shape, seed=7)
        wavfile.write(a.out / "unshaped" / f"{ln['id']}.wav", fs, (x * 32767).astype(np.int16))
        wavfile.write(a.out / "shaped" / f"{ln['id']}.wav", fs, (y * 32767).astype(np.int16))
        rows.append(
            {
                "id": ln["id"],
                "emotion": ln["emotion"],
                "source": str(src.relative_to(a.src)),
                "spread_before": round(pitch_spread_semitones(x, fs), 3),
                "spread_after": round(pitch_spread_semitones(y, fs), 3),
                "seconds_before": round(len(x) / fs, 2),
                "seconds_after": round(len(y) / fs, 2),
            }
        )
    (a.out / "shaping.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
