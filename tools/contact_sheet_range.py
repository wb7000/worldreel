"""Contact sheet over a RANGE of frames of one clip (to see slow drift).

Usage:
    python tools/contact_sheet_range.py VIDEO NAME [INDEX ...]

With no indices, ten frames are spread evenly over the clip. The sheet is
written to runs/analysis/contact_sheets/<name>.png.

The metrics cannot tell "the head turned" from "the lighting flickered" --
only the frames can, and the user's verdict is the acceptance criterion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path("runs/analysis/contact_sheets")
OUT.mkdir(parents=True, exist_ok=True)


def build(video: str, indices: list[int], name: str) -> None:
    cap = cv2.VideoCapture(video)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print(f"{video}: no frames")
        return
    picked = [frames[min(i, len(frames) - 1)] for i in indices]
    h, w = picked[0].shape[:2]
    scale = 0.42
    h2, w2 = int(h * scale), int(w * scale)
    tiles = [cv2.resize(f, (w2, h2)) for f in picked]
    row = np.hstack(tiles)
    banner = np.full((30, row.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(
        banner,
        " | ".join(f"f{i}" for i in indices),
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        1,
    )
    grid = np.vstack([banner, row])
    path = OUT / f"{name}.png"
    cv2.imwrite(str(path), grid)
    print(f"{video} ({len(frames)} frames) -> {path}")


def default_indices(total: int, count: int = 10) -> list[int]:
    if total <= 1:
        return [0]
    step = (total - 1) / (count - 1)
    return [round(i * step) for i in range(count)]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: contact_sheet_range.py VIDEO NAME [INDEX ...]")
        raise SystemExit(2)
    video = sys.argv[1]
    name = sys.argv[2]
    if len(sys.argv) > 3:
        indices = [int(v) for v in sys.argv[3:]]
    else:
        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        indices = default_indices(total)
    build(video, indices, name)
