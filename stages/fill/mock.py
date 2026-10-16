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
        pose = cam["poses"][0]
        x0, _, z0 = pose["position"]
        # Floor height: lowest reconstructed surfaces (2nd percentile), 0 if there are none.
        seen = PointCloud.unpack((ctx.inputs["reconstruct"] / POINTS_PTS).read_bytes())
        floor_y = float(np.percentile(seen.positions[:, 1], 2)) if len(seen) else 0.0
        # Behind the first camera, measured along its heading on the ground plane.
        qx, qy, qz, qw = pose["rotation"]
        fwd = np.array([-(2 * (qx * qz + qy * qw)), 0.0, -(1 - 2 * (qx * qx + qy * qy))])
        fwd = fwd / (np.linalg.norm(fwd) or 1.0)
        right = np.array([-fwd[2], 0.0, fwd[0]])
        a, b = np.meshgrid(np.linspace(-6, 6, 25), np.linspace(0.5, 6, 12))
        pts = (
            np.array([x0, floor_y, z0]) + a.ravel()[:, None] * right - b.ravel()[:, None] * fwd
        ).astype(np.float32)
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
                            "min": [float(v) for v in pts.min(axis=0)],
                            "max": [float(v) for v in pts.max(axis=0)],
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
