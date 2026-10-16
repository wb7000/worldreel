"""Mock `fill`: the area behind the camera was never seen, so it is filled with a flat grey floor
marked `guessed` with low confidence, plus a matching region record. This is the "gradient
first" baseline of brief §6; retrieval and generative fill come in Phase 4."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from formats.wreel import PROVENANCE_CLASSES, PointCloud
from pipeline.io import CAMERA_JSON, POINTS_PTS, REGIONS_JSON, read_json, write_json
from pipeline.stage import Impl, StageContext

GUESSED = PROVENANCE_CLASSES.index("guessed")
CONFIDENCE = 0.1


class MockFillStage:
    name: ClassVar[str] = "fill"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "mock"
    depends_on: ClassVar[tuple[str, ...]] = ("reconstruct", "camera")

    def run(self, ctx: StageContext) -> None:
        cam = read_json(ctx.inputs["camera"] / CAMERA_JSON)
        x0, _, z0 = cam["poses"][0]["position"]
        gx, gz = np.meshgrid(np.linspace(-6, 6, 25), np.linspace(0.5, 6, 12))
        pts = np.stack([gx.ravel() + x0, np.zeros(gx.size), gz.ravel() + z0], axis=1).astype(
            np.float32
        )
        n = len(pts)
        cloud = PointCloud(
            pts,
            np.full((n, 3), 128, np.uint8),
            np.full(n, GUESSED, np.uint8),
            np.full(n, CONFIDENCE, np.float32),
        )
        (ctx.out / POINTS_PTS).write_bytes(cloud.pack())
        write_json(
            ctx.out / REGIONS_JSON,
            {
                "regions": [
                    {
                        "id": "behind_camera",
                        "bounds": {
                            "min": [float(pts[:, 0].min()), 0.0, float(pts[:, 2].min())],
                            "max": [float(pts[:, 0].max()), 0.0, float(pts[:, 2].max())],
                        },
                        "class": "guessed",
                        "confidence": CONFIDENCE,
                        "sources": [
                            {"kind": "model", "model": "mock-flat-fill", "version": self.version}
                        ],
                    }
                ],
                "mock": True,
            },
        )
