from __future__ import annotations

import numpy as np

from stages.fill.planes import MAX_CONFIDENCE, progressive_fill, ransac_planes


def _floor_with_hole() -> tuple[np.ndarray, np.ndarray]:
    """A 4 x 4 m floor (y = 0), red on the left half, blue on the right, with an unseen 0.5 m hole
    in the middle (smaller than the 15 % gap limit); plus clutter well above it."""
    rng = np.random.default_rng(0)
    xz = rng.uniform(-2, 2, (40000, 2))
    keep = ~((np.abs(xz[:, 0]) < 0.25) & (np.abs(xz[:, 1]) < 0.25))
    xz = xz[keep]
    pts = np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], axis=1)
    cols = np.where(xz[:, :1] < 0, [[200, 30, 30]], [[30, 30, 200]]).astype(np.uint8)
    clutter = rng.uniform([-2, 0.8, -2], [2, 1.5, 2], (500, 3))
    return np.vstack([pts, clutter]), np.vstack([cols, np.full((500, 3), 128, np.uint8)])


def test_ransac_finds_the_floor() -> None:
    pts, _ = _floor_with_hole()
    planes = ransac_planes(pts, extent=4.0)
    assert planes
    assert abs(abs(planes[0].normal[1]) - 1) < 1e-3
    assert abs(planes[0].offset) < 0.01


def test_fill_covers_the_hole_on_the_plane_and_stays_inside_the_footprint() -> None:
    pts, cols = _floor_with_hole()
    _, tier1, tier2 = progressive_fill(pts, cols, extent=4.0, budget_s=5.0, colour_mode="diffuse")
    for tier in (tier1, tier2):
        assert len(tier.positions) > 0
        assert np.allclose(tier.positions[:, 1], 0.0, atol=1e-6)  # on the floor plane
        assert np.all(np.abs(tier.positions[:, [0, 2]]) <= 2.05)  # never outside the footprint
        hole = (np.abs(tier.positions[:, 0]) < 0.2) & (np.abs(tier.positions[:, 2]) < 0.2)
        assert hole.sum() > 0.5 * (0.4 / tier.cell) ** 2  # most of the hole interior is filled
        assert tier.confidence.max() <= MAX_CONFIDENCE
        # Colour continues the border: red side stays red, blue side stays blue.
        left = tier.colors[hole & (tier.positions[:, 0] < -0.08)]
        right = tier.colors[hole & (tier.positions[:, 0] > 0.08)]
        assert left[:, 0].mean() > left[:, 2].mean() and right[:, 2].mean() > right[:, 0].mean()
    assert tier2.cell < tier1.cell and len(tier2.positions) > len(tier1.positions)
    assert tier1.seconds < 5.0


def test_tier_reports_incomplete_when_budget_is_zero() -> None:
    pts, cols = _floor_with_hole()
    _, tier1, _ = progressive_fill(pts, cols, extent=4.0, budget_s=0.0, refine_budget_s=1.0)
    assert not tier1.complete


def test_separate_surfaces_on_one_plane_are_not_joined() -> None:
    """Two box tops at the same height, 2 m apart: the air between them must stay empty."""
    rng = np.random.default_rng(1)
    a = rng.uniform([-2, 1.0, -0.5], [-1, 1.0, 0.5], (5000, 3))
    b = rng.uniform([1, 1.0, -0.5], [2, 1.0, 0.5], (5000, 3))
    pts = np.vstack([a, b])
    cols = np.full((len(pts), 3), 100, np.uint8)
    _, tier1, _ = progressive_fill(pts, cols, extent=4.0)
    assert not np.any(np.abs(tier1.positions[:, 0]) < 0.8)


def test_floor_under_an_observed_object_is_hidden_not_filled() -> None:
    rng = np.random.default_rng(2)
    xz = rng.uniform(-2, 2, (40000, 2))
    xz = xz[~((np.abs(xz[:, 0]) < 0.25) & (np.abs(xz[:, 1]) < 0.25))]
    floor = np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], axis=1)
    box_top = rng.uniform([-0.25, 0.3, -0.25], [0.25, 0.3, 0.25], (2000, 3))  # box over the hole
    pts = np.vstack([floor, box_top])
    cols = np.full((len(pts), 3), 100, np.uint8)
    _, tier1, _ = progressive_fill(pts, cols, extent=4.0)
    under = (np.abs(tier1.positions[:, 0]) < 0.2) & (np.abs(tier1.positions[:, 2]) < 0.2)
    assert not under.any()


def test_default_fill_colour_is_the_plane_mean() -> None:
    pts, cols = _floor_with_hole()
    planes, tier1, _ = progressive_fill(pts, cols, extent=4.0)
    floor_mean = cols[planes[0].inliers].astype(float).mean(0)
    assert np.allclose(tier1.colors.astype(float), floor_mean, atol=1.0)
