"""Where inside a segment does the render get soft? Identity and sharpness by position.

`identity_curve.py` answers "does the face drift"; it says nothing about the
other half of the complaint, blur. Wan2GP's VACE notes claim the 1.3B model
"tends to blur final frames" of every window, which would mean the weak end of
each of our segments is not a seeding accident but the model's own behaviour --
and the fix would be to discard the tail rather than to re-render.

So measure both quantities against the position inside the segment: the face's
Laplacian variance (sharpness) and the SFace cosine to the reference portrait,
bucketed into head / middle / tail thirds, plus the per-frame curve.

Usage:
    python tools/tail_quality.py VIDEO PORTRAIT [BOUNDARY ...]
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


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _fit(xs: np.ndarray, ys: np.ndarray) -> float:
    """Least-squares slope per frame; NaN when there is nothing to fit."""
    if xs.size < 2:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def _thirds(values: np.ndarray) -> tuple[float, float, float]:
    """Mean of the first / middle / last third of a segment, by position."""
    n = int(values.size)
    if n == 0:
        return (float("nan"),) * 3
    a = max(1, n // 3)
    head = values[:a]
    tail = values[max(0, n - a) :]
    mid = values[a : max(a, n - a)]
    if mid.size == 0:
        mid = values
    return (_mean(head.tolist()), _mean(mid.tolist()), _mean(tail.tolist()))


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: tail_quality.py VIDEO PORTRAIT [BOUNDARY ...]")
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
    sharp: list[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = sw.detect_face(np.asarray(frame, dtype=np.uint8))
        if det is not None:
            bbox, _landmarks = det
            x, y, w, h = (int(v) for v in bbox[:4])
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            roi = frame[y0:y1, x0:x1]
            if roi.size:
                emb = sw.compute_embedding(np.asarray(frame, dtype=np.uint8))
                if emb is not None:
                    frames.append(idx)
                    sims.append(float(emb @ ref_emb))
                    sharp.append(float(cv2.Laplacian(roi, cv2.CV_64F).var()))
        idx += 1
    cap.release()

    if not sims:
        print("FAIL: no faces detected")
        return 2

    f = np.asarray(frames, dtype=np.int64)
    s = np.asarray(sims, dtype=np.float64)
    k = np.asarray(sharp, dtype=np.float64)
    print(f"{video_path}: {s.size} frames with a face out of {idx}")
    print(
        f"identity mean={s.mean():.4f} min={s.min():.4f}   "
        f"sharpness mean={k.mean():.1f} min={k.min():.1f}"
    )

    print("\nper-frame sharpness curve (each char = one frame, height 0-9 = lap/30):")
    row = "".join(str(min(9, int(v // 30))) for v in k)
    for start in range(0, len(row), 60):
        print(f"  {start:4d} {row[start : start + 60]}")

    edges = [0, *boundaries, int(f[-1]) + 1]
    print("\nper-segment head / middle / tail (position inside the segment):")
    print(f"  {'seg':>4} {'range':>12}  {'id.head':>7} {'id.mid':>7} {'id.tail':>7}  "
          f"{'sh.head':>7} {'sh.mid':>7} {'sh.tail':>7}  {'id.slope':>9} {'sh.slope':>9}")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (f >= lo) & (f < hi)
        if m.sum() == 0:
            continue
        seg_f = f[m].astype(np.float64)
        ih, im, it = _thirds(s[m])
        kh, km, kt = _thirds(k[m])
        # Skip the first frames: the conditioning frame is not the trend.
        skip = min(5, seg_f.size - 1)
        interior = slice(skip, None) if skip else slice(None)
        id_slope = _fit(seg_f[interior], s[m][interior])
        sh_slope = _fit(seg_f[interior], k[m][interior])
        print(
            f"  {i:4d} [{lo:3d},{hi:3d})  {ih:7.4f} {im:7.4f} {it:7.4f}  "
            f"{kh:7.1f} {km:7.1f} {kt:7.1f}  {id_slope:+9.5f} {sh_slope:+9.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
