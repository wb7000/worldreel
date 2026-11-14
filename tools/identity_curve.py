"""Per-frame identity curve: where does the face drift away from the reference?

`measure_identity.py` reports mean/min only, which cannot separate the two
distinct failure modes of a chained multi-segment generation:

- **gradual slope** — each segment conditions on the previous segment's tail, so
  identity walks away from the reference as the chain lengthens (chain drift);
- **step at a segment boundary** — the overlap handoff itself loses identity,
  i.e. the conditioning frames are not being reproduced faithfully.

The two need different fixes (re-anchor on the reference vs. fix the overlap),
so print the series with the segment boundaries marked and fit a slope to the
interior of each segment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"


def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Least-squares slope per frame; NaN when there is nothing to fit."""
    if xs.size < 2:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: identity_curve.py VIDEO PORTRAIT [BOUNDARY ...]")
        return 2
    video_path, portrait_path = sys.argv[1], sys.argv[2]
    boundaries = [int(b) for b in sys.argv[3:]]

    sw = FaceSwapper(YUNET, SFACE)
    img = cv2.imread(portrait_path)
    if img is None:
        print(f"FAIL: cannot read {portrait_path}")
        return 2
    ref_emb = sw.compute_embedding(np.asarray(img, dtype=np.uint8))
    if ref_emb is None:
        print("FAIL: no reference embedding")
        return 2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"FAIL: cannot open {video_path}")
        return 2

    frames: list[int] = []
    sims: list[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        emb = sw.compute_embedding(np.asarray(frame, dtype=np.uint8))
        if emb is not None:
            frames.append(idx)
            sims.append(float(emb @ ref_emb))
        idx += 1
    cap.release()

    if not sims:
        print("FAIL: no faces detected")
        return 2

    f = np.asarray(frames, dtype=np.int64)
    s = np.asarray(sims, dtype=np.float64)
    print(f"{video_path}: {len(s)} frames with a face out of {idx}")
    print(
        f"mean={s.mean():.4f} min={s.min():.4f} (frame {f[int(np.argmin(s))]}) "
        f"max={s.max():.4f} std={s.std():.4f}"
    )
    print(f"frames below 0.80: {int(np.sum(s < 0.80))}/{s.size}")
    print(f"frames below 0.75: {int(np.sum(s < 0.75))}/{s.size}")

    print("\nper-frame curve ('.' >=0.80, '-' >=0.70, 'x' <0.70):")
    row = "".join("." if v >= 0.80 else "-" if v >= 0.70 else "x" for v in s)
    for start in range(0, len(row), 60):
        print(f"  {start:4d} {row[start : start + 60]}")

    # Segment interior slopes: boundaries split the chain, and only the interior
    # is free of the handoff transient at each cut.
    edges = [0, *boundaries, int(f[-1]) + 1]
    print("\nper-segment fit:")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (f >= lo) & (f < hi)
        if m.sum() == 0:
            continue
        seg_s, seg_f = s[m], f[m]
        # Drop the first few frames: they are the overlap handoff, not the trend.
        k = min(5, seg_f.size - 1)
        interior = slice(k, None) if k else slice(None)
        slope = _slope(seg_f[interior].astype(np.float64), seg_s[interior])
        label = "chain" if i else "seed"
        print(
            f"  seg {i} [{lo:3d},{hi:3d}) {label:5s} "
            f"start={seg_s[0]:.4f} end={seg_s[-1]:.4f} "
            f"min={seg_s.min():.4f} interior_slope={slope:+.5f}/frame"
        )

    if boundaries:
        print("\nhandoff steps (frame before -> first interior frame after cut):")
        for b in boundaries:
            before = s[f == b - 1]
            after = s[f >= b + 5]
            if before.size and after.size:
                print(f"  cut@{b}: {before[0]:.4f} -> {after[0]:.4f} ({after[0] - before[0]:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
