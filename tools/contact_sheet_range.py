"""Contact sheet over a RANGE of frames of one clip (to see slow drift)."""

from __future__ import annotations

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


if __name__ == "__main__":
    build(
        "runs/kaggle/character_consistency_v2/out/face_consistency.mp4",
        [0, 40, 80, 120, 150, 170, 187, 200, 220, 239],
        "v12_raw_drift",
    )
