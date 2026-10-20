"""Strengthen the vocal change of whole-conversation renders per turn, keeping their timing.

    python -m eval.conversation_shaping --turns runs/conv/turns.json --out runs/conv

For every exchange and strength k (0.8 and 1.5 x the A-11 preset of the turn's emotion, tempo
fixed at 1): each turn segment is shaped and put back at exactly the same place (length
matched, 10 ms crossfades at the borders). Writes `<out>/k<k>/<exchange>.wav`.
Default method: TD-PSOLA (A-11 v3, keeps the real waveform); `--method world` = vocoder (v2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from stages.generate.voice_quality import PRESETS, shape_voice, shape_voice_psola

STRENGTHS = (0.8, 1.5)


def reshape(
    x: np.ndarray, sr: int, turns: list[dict[str, object]], k: float, method: str = "psola"
) -> np.ndarray:
    y = x.copy()
    fade = int(0.01 * sr)
    for t in turns:
        s, e = int(float(t["start"]) * sr), int(float(t["end"]) * sr)  # type: ignore[arg-type]
        seg = x[s:e]
        if len(seg) < sr // 4:
            continue
        shape = PRESETS.get(str(t["emotion"]), PRESETS["neutral"]).amplified(k)
        fn = shape_voice_psola if method == "psola" else shape_voice
        out = fn(seg, sr, shape, seed=7)
        out = np.pad(out, (0, max(0, len(seg) - len(out))))[: len(seg)]
        ramp = np.ones(len(seg))
        ramp[:fade] = np.linspace(0, 1, fade)
        ramp[-fade:] = np.linspace(1, 0, fade)
        y[s:e] = seg * (1 - ramp) + out * ramp
    return y


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.conversation_shaping")
    ap.add_argument("--turns", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", choices=("psola", "world"), default="psola")
    a = ap.parse_args(argv)
    data = json.loads(a.turns.read_text(encoding="utf-8"))
    for k in STRENGTHS:
        d = a.out / f"k{k}"
        d.mkdir(parents=True, exist_ok=True)
        for ex_id, ex in data.items():
            sr, x = wavfile.read(ex["file"])
            x = x.astype(np.float64) / (32768.0 if x.dtype == np.int16 else 1.0)
            y = reshape(x, sr, ex["turns"], k, a.method)
            wavfile.write(d / f"{ex_id}.wav", sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
            print(f"k={k} {ex_id}: {len(x) / sr:.2f}s -> {len(y) / sr:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
