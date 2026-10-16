"""Mock `reconstruct`: back-projects the first proxy frame through the camera depth into a
point cloud (all points `observed`). A real implementation (Gaussian splatting) arrives in
Phase 3."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from formats.wreel import PROVENANCE_CLASSES, PointCloud
from pipeline.io import (
    CAMERA_JSON,
    DEPTH_PROXY_NPY,
    POINTS_JSON,
    POINTS_PTS,
    PROXY_NPY,
    load_npy,
    read_json,
    write_json,
)
from pipeline.stage import Impl, StageContext

OBSERVED = PROVENANCE_CLASSES.index("observed")


def quat_to_matrix(q: list[float]) -> npt.NDArray[np.float64]:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def backproject(
    rgb: npt.NDArray[np.uint8],
    depth: npt.NDArray[np.float32],
    k: dict[str, float],
    position: list[float],
    rotation: list[float],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Pixels with depth > 0 → world points. Camera looks along -Z with +Y up (spec §2)."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    valid = depth > 0
    z = depth[valid].astype(np.float64)
    cam = np.stack(
        [(u[valid] - k["cx"]) / k["fx"] * z, -(v[valid] - k["cy"]) / k["fy"] * z, -z], axis=1
    )
    world = cam @ quat_to_matrix(rotation).T + np.asarray(position)
    return world.astype(np.float32), rgb[valid].astype(np.uint8)


class MockReconstructStage:
    name: ClassVar[str] = "reconstruct"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "mock"
    depends_on: ClassVar[tuple[str, ...]] = ("frames", "camera")

    def run(self, ctx: StageContext) -> None:
        proxy = load_npy(ctx.inputs["frames"] / PROXY_NPY)
        depth = load_npy(ctx.inputs["camera"] / DEPTH_PROXY_NPY)
        cam = read_json(ctx.inputs["camera"] / CAMERA_JSON)
        pose = cam["poses"][0]
        pts, colors = backproject(
            proxy[0], depth[0], cam["intrinsics"], pose["position"], pose["rotation"]
        )
        n = len(pts)
        cloud = PointCloud(pts, colors, np.full(n, OBSERVED, np.uint8), np.full(n, 0.9, np.float32))
        (ctx.out / POINTS_PTS).write_bytes(cloud.pack())
        write_json(
            ctx.out / POINTS_JSON, {"count": n, "point_size": 0.12, "source_frame": 0, "mock": True}
        )
