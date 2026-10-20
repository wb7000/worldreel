"""Split whole-conversation renders into turns and check each turn's voice (D-8.6).

    data/envs/spk/Scripts/python -m eval.conversation_turns \
        --src runs/kaggle/voice_acting4/out/final --suffix R3 --refs data/voices_designed \
        --out runs/conv/turns.json

Whisper (word timestamps) transcribes each exchange; its words are aligned to the known spoken
text of every turn (difflib on normalised words); a turn boundary is put in the middle of the
pause between the last word of one turn and the first word of the next. Each turn segment gets
an ECAPA embedding compared with the designed voices: is it closest to its own character?
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import numpy as np

from stages.generate.voices import _words

ECAPA_SHA = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def boundaries(
    turn_words: list[list[str]], hyp: list[tuple[str, float, float]], dur: float
) -> list[tuple[float, float]]:
    """(start, end) seconds per turn: reference turns aligned word by word to Whisper words."""
    ref = [w for ws in turn_words for w in ws]
    owner = [k for k, ws in enumerate(turn_words) for _ in ws]
    hyp_words = [w for w, _, _ in hyp]
    sm = difflib.SequenceMatcher(a=ref, b=hyp_words, autojunk=False)
    ref2hyp: dict[int, int] = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            ref2hyp[blk.a + k] = blk.b + k
    first_last: list[tuple[int | None, int | None]] = []
    for t in range(len(turn_words)):
        idx = [ref2hyp[i] for i in range(len(ref)) if owner[i] == t and i in ref2hyp]
        first_last.append((min(idx), max(idx)) if idx else (None, None))
    cuts = [0.0]
    for t in range(1, len(turn_words)):
        prev_last, nxt_first = first_last[t - 1][1], first_last[t][0]
        if prev_last is not None and nxt_first is not None:
            cuts.append((hyp[prev_last][2] + hyp[nxt_first][1]) / 2)
        else:  # unaligned turn: split evenly in the remaining time
            cuts.append(cuts[-1] + (dur - cuts[-1]) / (len(turn_words) - t + 1))
    cuts.append(dur)
    return [(round(cuts[t], 3), round(cuts[t + 1], 3)) for t in range(len(turn_words))]


def main(argv: list[str] | None = None) -> int:
    import soundfile as sf
    import torch
    import torchaudio
    import whisper
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    ap = argparse.ArgumentParser(prog="python -m eval.conversation_turns")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--suffix", default="R3")
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    exchanges = json.loads(Path("eval/conversations.json").read_text(encoding="utf-8"))["exchanges"]
    designed = json.loads((a.refs / "designed_voices.json").read_text())
    asr = whisper.load_model("small.en", device="cpu")
    enc = EncoderClassifier.from_hparams(
        source=snapshot_download("speechbrain/spkrec-ecapa-voxceleb", revision=ECAPA_SHA),
        savedir="data/models/ecapa",
        local_strategy=LocalStrategy.COPY,
    )

    def emb(x: np.ndarray, sr: int) -> np.ndarray:
        w = torchaudio.functional.resample(
            torch.from_numpy(x.astype(np.float32)).unsqueeze(0), sr, 16000
        )
        e = enc.encode_batch(w).squeeze().numpy()
        out: np.ndarray = e / np.linalg.norm(e)
        return out

    refs = {}
    for cid, v in designed.items():
        x, sr = sf.read(a.refs / v["file"], dtype="float32")
        refs[cid] = emb(x if x.ndim == 1 else x.mean(1), sr)

    result = {}
    for ex in exchanges:
        path = a.src / f"{ex['id']}_{a.suffix}.wav"
        x, sr = sf.read(path, dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        res = asr.transcribe(str(path), language="en", fp16=False, word_timestamps=True)
        hyp = [
            (ww[0], float(w["start"]), float(w["end"]))
            for seg in res["segments"]
            for w in seg.get("words", [])
            if (ww := _words(w["word"]))
        ]
        turns = boundaries([_words(t["spoken"]) for t in ex["turns"]], hyp, len(x) / sr)
        rows = []
        for t, (s, e) in zip(ex["turns"], turns, strict=True):
            seg = x[int(s * sr) : int(e * sr)]
            cos = (
                {c: float(emb(seg, sr) @ r) for c, r in refs.items()} if len(seg) > sr // 2 else {}
            )
            rows.append({"cid": t["cid"], "emotion": t["emotion"], "manner": t["manner"],
                         "start": s, "end": e,
                         "nearest_ok": bool(cos) and max(cos, key=lambda k: cos[k]) == t["cid"],
                         "cos_own": round(cos.get(t["cid"], float("nan")), 3)})  # fmt: skip
        result[ex["id"]] = {
            "file": str(path),
            "sr": sr,
            "heard": res["text"].strip(),
            "turns": rows,
        }
        print(
            ex["id"],
            [(r["cid"], r["start"], r["end"], r["nearest_ok"], r["cos_own"]) for r in rows],
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
