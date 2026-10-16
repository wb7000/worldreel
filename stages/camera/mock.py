"""Mock `camera`: a static camera 1.5 m above a flat ground plane, 60° horizontal field of view.
Depth is the ground plane below the horizon and invalid (0) above it. No model is run."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from pipeline.io import CAMERA_JSON, DEPTH_PROXY_NPY, FRAMES_JSON, read_json, save_npy, write_json
from pipeline.stage import Impl, StageContext

CAMERA_HEIGHT = 1.5
HFOV_DEG = 60.0


def intrinsics(width: int, height: int, hfov_deg: float = HFOV_DEG) -> dict[str, float]:
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {
        "fx": fx,
        "fy": fx,
        "cx": (width - 1) / 2,
        "cy": (height - 1) / 2,
        "width": width,
        "height": height,
    }


class MockCameraStage:
    name: ClassVar[str] = "camera"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "mock"
    depends_on: ClassVar[tuple[str, ...]] = ("frames",)

    def run(self, ctx: StageContext) -> None:
        meta = read_json(ctx.inputs["frames"] / FRAMES_JSON)
        n, w, h = int(meta["proxy_count"]), int(meta["proxy_width"]), int(meta["proxy_height"])
        k = intrinsics(w, h)
        v = np.arange(h, dtype=np.float32)[:, None] - k["cy"]
        with np.errstate(divide="ignore"):
            ground = np.where(v > 0.5, CAMERA_HEIGHT * k["fy"] / np.maximum(v, 1e-6), 0.0)
        depth = np.broadcast_to(ground, (h, w)).astype(np.float32)
        save_npy(ctx.out / DEPTH_PROXY_NPY, np.repeat(depth[None], n, axis=0))
        fps = float(meta["fps"])
        write_json(
            ctx.out / CAMERA_JSON,
            {
                "intrinsics": k,
                "fps": fps,
                "poses": [
                    {
                        "t": round(i / fps, 4),
                        "position": [0.0, CAMERA_HEIGHT, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                    }
                    for i in range(n)
                ],
                "mock": True,
            },
        )
