"""Mock `frames`: no ffmpeg. Synthesises a few low-res frames (sky, ground, a moving box),
seeded by the source hash so the same input always gives the same frames."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from pipeline.io import FRAMES_JSON, PROXY_NPY, save_npy, write_json
from pipeline.stage import SOURCE, Impl, StageContext

W, H, N, FPS = 64, 48, 8, 4.0


class MockFramesStage:
    name: ClassVar[str] = "frames"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "mock"
    depends_on: ClassVar[tuple[str, ...]] = (SOURCE,)

    def run(self, ctx: StageContext) -> None:
        rng = np.random.default_rng(int(ctx.item.sha256[:8], 16) ^ ctx.seed)
        box_color = rng.integers(60, 230, size=3)
        frames = np.zeros((N, H, W, 3), dtype=np.uint8)
        rows = np.linspace(0, 1, H // 2)[:, None]
        sky = (np.array([150, 185, 230]) * (1 - 0.3 * rows)).astype(np.uint8)
        for i in range(N):
            frames[i, : H // 2] = sky[:, None, :]
            frames[i, H // 2 :] = (110, 105, 95)
            x0 = 8 + i * 4
            frames[i, 18:34, x0 : x0 + 12] = box_color
        save_npy(ctx.out / PROXY_NPY, frames)
        write_json(
            ctx.out / FRAMES_JSON,
            {
                "width": W,
                "height": H,
                "source_fps": FPS,
                "duration": N / FPS,
                "codec": "mock",
                "has_audio": False,
                "fps": FPS,
                "count": 0,
                "frame_width": W,
                "frame_height": H,
                "proxy_count": N,
                "proxy_width": W,
                "proxy_height": H,
                "cuts": [],
                "audio": None,
                "mock": True,
            },
        )
