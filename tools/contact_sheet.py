"""Contact sheets: frames 0/6/12/18 of a clip side by side.

The metrics cannot tell "the head turned" from "the lighting flickered" -- only
the frames can, and the user's verdict is the acceptance criterion. One row per
clip, so two clips can be compared directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path("runs/analysis/contact_sheets")
OUT.mkdir(parents=True, exist_ok=True)


def sheet(path: Path, indices: list[int], labels: list[str], out: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print(f"{path}: no frames")
        return
    picked = [frames[min(i, len(frames) - 1)] for i in indices]
    h, w = picked[0].shape[:2]
    scale = 0.55
    h2, w2 = int(h * scale), int(w * scale)
    tiles = [cv2.resize(f, (w2, h2)) for f in picked]
    row = np.hstack(tiles)
    banner = np.full((34, row.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(
        banner, " | ".join(labels), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1
    )
    grid = np.vstack([banner, row])
    cv2.imwrite(str(out), grid)
    print(f"{path.name}: {len(frames)} frames -> {out} ({row.shape[1]}x{row.shape[0]})")


if __name__ == "__main__":
    # (video, output name)
    targets = [
        ("runs/kaggle/character_render_lora_v4_sweep/out/sweep_s000.mp4", "sweep_base_s000"),
        ("runs/kaggle/character_render_lora_v4_sweep/out/sweep_s025.mp4", "sweep_static_s025"),
        ("runs/kaggle/character_render_lora_v4_sweep/out/sweep_s050.mp4", "sweep_static_s050"),
        ("runs/kaggle/character_render_lora_probe/out/probe_v4_late75.mp4", "probe_v4_late75"),
        ("runs/kaggle/character_render_lora_probe/out/probe_v4_late50.mp4", "probe_v4_late50"),
        ("runs/kaggle/character_render_lora_probe/out/probe_v5_step080.mp4", "probe_v5_step080"),
        ("runs/kaggle/character_render_lora_probe/out/probe_v5_step020.mp4", "probe_v5_step020"),
    ]
    indices = [0, 6, 12, 18]
    for video, name in targets:
        p = Path(video)
        if not p.exists():
            print(f"MISSING {p}")
            continue
        sheet(p, indices, [f"f{i}" for i in indices], OUT / f"{name}.png")
    if len(sys.argv) > 1:
        print("wrote", OUT)
