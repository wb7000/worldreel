from __future__ import annotations

import numpy as np
import pytest

from eval.benchmark import check_thresholds, load_thresholds
from eval.depth import depth_errors
from eval.geometry import (
    matrix_to_quat,
    quat_to_matrix,
    rotation_angle_deg,
    trajectory_errors,
    umeyama,
)
from eval.image import psnr, ssim
from eval.render import render_points


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return quat_to_matrix(q / np.linalg.norm(q))


def test_umeyama_recovers_similarity() -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3))
    r, s, t = random_rotation(rng), 2.7, np.array([1.0, -2.0, 0.5])
    sim = umeyama(src, s * src @ r.T + t)
    assert sim.scale == pytest.approx(s)
    assert np.allclose(sim.rotation, r)
    assert np.allclose(sim.translation, t)


def test_quaternion_round_trip() -> None:
    rng = np.random.default_rng(2)
    for _ in range(20):
        r = random_rotation(rng)
        assert np.allclose(quat_to_matrix(matrix_to_quat(r)), r, atol=1e-9)


def test_trajectory_errors_zero_for_a_similar_trajectory_on_a_short_arc() -> None:
    rng = np.random.default_rng(3)
    ang = np.linspace(0, 0.6, 30)  # a short arc: positions alone barely constrain rotation
    pos = np.stack([12 * np.cos(ang), 4 + 0 * ang, 12 * np.sin(ang)], 1)
    rots = np.stack([quat_to_matrix([0, np.sin(a / 2), 0, np.cos(a / 2)]) for a in ang])
    r, s, t = random_rotation(rng), 0.37, np.array([1.0, 2.0, 3.0])
    est_pos = (pos - t) @ r / s
    est_rot = np.einsum("ij,njk->nik", r.T, rots)
    errs, sim = trajectory_errors(est_pos, est_rot, pos, rots)
    assert errs.ate_rmse < 1e-9
    assert errs.rot_mean_deg < 1e-4
    assert errs.rpe_rot_deg < 1e-4
    assert sim.scale == pytest.approx(s)


def test_trajectory_errors_detect_a_rotation_bias() -> None:
    ang = np.linspace(0, 1.0, 20)
    pos = np.stack([10 * np.cos(ang), 0 * ang, 10 * np.sin(ang)], 1)
    rots = np.stack([quat_to_matrix([0, np.sin(a / 2), 0, np.cos(a / 2)]) for a in ang])
    bias = quat_to_matrix([np.sin(np.radians(1.5)), 0, 0, np.cos(np.radians(1.5))])  # 3° about x
    errs, _ = trajectory_errors(pos, np.einsum("nij,jk->nik", rots, bias), pos, rots)
    assert 1.0 < errs.rot_mean_deg < 3.5
    assert rotation_angle_deg(bias) == pytest.approx(3.0)


def test_depth_errors() -> None:
    gt = np.array([[2.0, 4.0], [0.0, 1.0]])
    est = np.array([[2.2, 4.0], [3.0, 0.0]])
    d = depth_errors(est, gt)
    assert d.count == 2
    assert d.abs_rel == pytest.approx(0.05)
    assert d.delta1 == 1.0
    assert d.completeness == pytest.approx(2 / 3)


def test_psnr_ssim() -> None:
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    assert psnr(img, img) == float("inf")
    assert ssim(img, img) == pytest.approx(1.0)

    def noisy(amount: int) -> np.ndarray:
        noise = rng.integers(-amount, amount + 1, img.shape)
        return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    light, heavy = noisy(20), noisy(80)
    assert 20 < psnr(img, light) < 35
    assert psnr(img, heavy) < psnr(img, light)
    assert ssim(img, heavy) < ssim(img, light) < 1.0


def test_render_projects_and_keeps_nearest() -> None:
    k = {"fx": 100.0, "fy": 100.0, "cx": 15.5, "cy": 15.5, "width": 32, "height": 32}
    # Camera at origin looking down -Z; two points on the optical axis, red one nearer.
    pts = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, -2.0], [1.0, 0.0, -10.0]])
    cols = np.array([[0, 255, 0], [255, 0, 0], [0, 0, 255]], np.uint8)
    img, depth = render_points(pts, cols, k, [0, 0, 0], [0, 0, 0, 1], radius_px=0)
    assert tuple(img[16, 16]) == (255, 0, 0)
    assert depth[16, 16] == pytest.approx(2.0)
    assert tuple(img[16, 26]) == (0, 0, 255)  # x=1 at depth 10 → u = 15.5 + 10 → 25.5 → 26
    assert depth[0, 0] == 0


def test_thresholds_file_and_checker() -> None:
    limits = load_thresholds()
    assert {"ate_rel_max", "depth_abs_rel_max", "novel_psnr_gap_max", "novel_coverage_min"} <= set(
        limits
    )
    checks = check_thresholds(
        {"ate_rel": 0.01, "novel_coverage": 0.1}, {"ate_rel_max": 0.02, "novel_coverage_min": 0.6}
    )
    assert checks["ate_rel"]["pass"] is True
    assert checks["novel_coverage"]["pass"] is False
    missing = check_thresholds({}, {"ate_rel_max": 0.02})
    assert missing["ate_rel"]["pass"] is False
