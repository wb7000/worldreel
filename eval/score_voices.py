"""Score folders of voiced lines: speaker identity (ECAPA vs designed references) and words
(Whisper). Runs in the speaker environment (torch, speechbrain, openai-whisper), e.g.

    data/envs/spk/Scripts/python -m eval.score_voices --refs data/voices_designed \
        --dirs runs/shaping/round3_D2/unshaped runs/shaping/round3_D2/shaped --out scores.json

Lines and their characters come from eval/emotion_lines.json (file name = line id).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stages.generate.voices import take_ok

ECAPA_SHA = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def main(argv: list[str] | None = None) -> int:
    import soundfile as sf
    import torch
    import torchaudio
    import whisper
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    ap = argparse.ArgumentParser(prog="python -m eval.score_voices")
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    lines = {ln["id"]: ln for ln in json.loads(Path("eval/emotion_lines.json").read_text())["lines"]}
    designed = json.loads((a.refs / "designed_voices.json").read_text())
    enc = EncoderClassifier.from_hparams(
        source=snapshot_download("speechbrain/spkrec-ecapa-voxceleb", revision=ECAPA_SHA),
        savedir="data/models/ecapa",
        local_strategy=LocalStrategy.COPY,
    )
    asr = whisper.load_model("small.en", device="cpu")

    def load(p: Path) -> tuple[torch.Tensor, float]:
        x, sr = sf.read(p, dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        return torchaudio.functional.resample(torch.from_numpy(x).unsqueeze(0), sr, 16000), len(x) / sr

    def emb(p: Path) -> np.ndarray:
        e = enc.encode_batch(load(p)[0]).squeeze().numpy()
        out: np.ndarray = e / np.linalg.norm(e)
        return out

    refs = {cid: emb(a.refs / v["file"]) for cid, v in designed.items()}
    result: dict[str, dict[str, object]] = {}
    for d in a.dirs:
        rows = {}
        for p in sorted(d.glob("*.wav")):
            ln = lines[p.stem]
            e = emb(p)
            cos = {cid: float(e @ r) for cid, r in refs.items()}
            text = asr.transcribe(str(p), language="en", fp16=False)["text"]
            ok, wer = take_ok(ln["text"], text, load(p)[1])
            rows[p.stem] = {"cos_own": round(cos[ln["cid"]], 3),
                            "nearest_ok": max(cos, key=lambda k: cos[k]) == ln["cid"],
                            "wer": wer, "usable": ok, "heard": text.strip()}  # fmt: skip
        n = len(rows)
        result[str(d)] = {
            "n": n,
            "nearest_ref_accuracy": round(sum(r["nearest_ok"] for r in rows.values()) / n, 3),
            "cos_own_mean": round(float(np.mean([r["cos_own"] for r in rows.values()])), 3),
            "wer_mean": round(float(np.mean([r["wer"] for r in rows.values()])), 3),
            "usable": sum(bool(r["usable"]) for r in rows.values()),
            "lines": rows,
        }
        print(d, {k: v for k, v in result[str(d)].items() if k != "lines"})
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
