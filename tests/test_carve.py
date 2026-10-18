from __future__ import annotations

import numpy as np

from formats.wreel import GaussianCloud
from stages.package.gaussians import carve_people


def _cloud(points: np.ndarray) -> GaussianCloud:
    n = len(points)
    return GaussianCloud(
        points.astype(np.float32),
        np.full((n, 3), 0.01, np.float32),
        np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1)),
        np.zeros((n, 3), np.uint8),
        np.ones(n, np.float32),
        np.zeros(n, np.uint8),
        np.ones(n, np.float32),
    )


def test_carve_removes_body_volume_along_the_path_but_keeps_floor_and_surroundings() -> None:
    points = np.array(
        [
            [0.0, 1.0, 0.0],  # chest at the first sample
            [0.5, 1.6, 0.0],  # head halfway between samples (interpolated path)
            [0.5, 0.0, 0.0],  # floor under the path: kept
            [0.5, 1.0, 1.0],  # 1 m to the side: kept
            [1.0, 2.5, 0.0],  # above the head: kept
        ]
    )
    track = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ups = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    kept, removed = carve_people(_cloud(points), [(track, ups, 1.75)])
    assert removed == 2
    assert np.allclose(kept.positions, points[2:])
