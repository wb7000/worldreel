from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eval.geometry import quat_to_matrix
from formats.common import FormatError
from formats.wreel import GAUSSIAN_STRIDE, GaussianCloud, read_wreel, validate_wreel, write_wreel
from stages.package.gaussians import (
    coarse_lod,
    matrix_to_quat_xyzw,
    quat_wxyz_to_matrix,
    splats_to_gaussians,
    wreel_with_gaussians,
)

FLIP = np.diag([1.0, -1.0, -1.0])


def random_cloud(n: int = 64, seed: int = 0) -> GaussianCloud:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return GaussianCloud(
        positions=rng.normal(size=(n, 3)).astype(np.float32),
        scales=rng.uniform(0.01, 0.2, size=(n, 3)).astype(np.float32),
        rotations=q.astype(np.float32),
        colors=rng.integers(0, 256, size=(n, 3)).astype(np.uint8),
        opacity=(rng.integers(0, 256, size=n) / 255).astype(np.float32),
        cls=np.zeros(n, np.uint8),
        confidence=rng.uniform(size=n).astype(np.float32),
    )


def test_gaussian_record_round_trip_and_layout() -> None:
    g = random_cloud()
    data = g.pack()
    assert len(data) == len(g) * GAUSSIAN_STRIDE == len(g) * 52
    back = GaussianCloud.unpack(data)
    assert back.pack() == data
    assert np.allclose(back.opacity, g.opacity, atol=1 / 510)
    assert data[40:43] == bytes(g.colors[0])
    assert data[44] == g.cls[0]


def test_quaternion_conversions() -> None:
    rng = np.random.default_rng(1)
    q_wxyz = rng.normal(size=(200, 4))
    r = quat_wxyz_to_matrix(q_wxyz)
    q = matrix_to_quat_xyzw(r)
    assert np.allclose(np.linalg.norm(q, axis=1), 1)
    for i in range(len(r)):
        assert np.allclose(quat_to_matrix(q[i]), r[i], atol=1e-9)


def test_splat_conversion_preserves_covariance_and_position() -> None:
    rng = np.random.default_rng(2)
    n = 50
    means_cv = rng.normal(size=(n, 3))
    params: dict[str, Any] = {
        "means": means_cv.astype(np.float32),
        "scales": np.log(rng.uniform(0.05, 0.3, size=(n, 3))).astype(np.float32),
        "quats": rng.normal(size=(n, 4)).astype(np.float32),
        "opacities": np.full(n, 2.0, np.float32),
        "colors": np.zeros((n, 3), np.float32),
    }
    g = splats_to_gaussians(params, centre_cv=np.zeros(3), background_beyond=100.0)
    assert len(g) == n
    assert np.allclose(g.positions, means_cv @ FLIP.T, atol=1e-6)
    for i in range(n):
        r_cv = quat_wxyz_to_matrix(params["quats"][i : i + 1].astype(np.float64))[0]
        s2 = np.diag(np.exp(2 * params["scales"][i].astype(np.float64)))
        cov_cv = r_cv @ s2 @ r_cv.T
        r_wr = quat_to_matrix(g.rotations[i])
        cov_wr = r_wr @ np.diag(g.scales[i].astype(np.float64) ** 2) @ r_wr.T
        assert np.allclose(cov_wr, FLIP @ cov_cv @ FLIP, atol=1e-6)
    assert np.all(g.colors == 128)  # sigmoid(0) = 0.5


def test_conversion_drops_invisible_and_marks_background() -> None:
    params: dict[str, Any] = {
        "means": np.array([[0, 0, 1], [0, 0, 2], [0, 0, 500]], np.float32),
        "scales": np.zeros((3, 3), np.float32),
        "quats": np.tile([1, 0, 0, 0], (3, 1)).astype(np.float32),
        "opacities": np.array([3.0, -20.0, 3.0], np.float32),  # the second is invisible
        "colors": np.zeros((3, 3), np.float32),
    }
    g = splats_to_gaussians(params, centre_cv=np.zeros(3), background_beyond=50.0)
    assert len(g) == 2
    assert g.confidence[0] == pytest.approx(0.9)
    assert g.confidence[1] == pytest.approx(0.6)


def test_coarse_lod_keeps_biggest_contributors() -> None:
    g = random_cloud(100)
    coarse = coarse_lod(g, 0.2)
    assert len(coarse) == 20
    weight = g.opacity * np.sort(g.scales, 1)[:, 2] * np.sort(g.scales, 1)[:, 1]
    assert (
        np.sort(weight)[-1]
        in coarse.opacity * np.sort(coarse.scales, 1)[:, 2] * np.sort(coarse.scales, 1)[:, 1]
    )


def test_gaussian_wreel_validates_and_needs_version_1_1(
    tmp_path: Path, sample: tuple[dict[str, Any], dict[str, bytes]], key: Ed25519PrivateKey
) -> None:
    base = tmp_path / "points.wreel"
    write_wreel(base, sample[0], sample[1], key)
    template = read_wreel(base)
    manifest, entries = wreel_with_gaussians(
        template, random_cloud(500), stage_record={"name": "splat", "version": "test"}
    )
    path = tmp_path / "splats.wreel"
    write_wreel(path, manifest, entries, key)
    doc = read_wreel(path)
    assert doc.report.ok, doc.report.issues
    assert doc.manifest["format_version"] == "1.1"
    layer = doc.manifest["scene"]["layers"][0]
    assert layer["kind"] == "gaussians" and [lod["count"] for lod in layer["lods"]] == [100, 500]
    manifest["format_version"] = "1.0"
    with pytest.raises(FormatError, match=r"1\.1"):
        write_wreel(tmp_path / "bad.wreel", manifest, entries, key)


def test_bad_gaussian_rotation_rejected(
    tmp_path: Path, sample: tuple[dict[str, Any], dict[str, bytes]], key: Ed25519PrivateKey
) -> None:
    base = tmp_path / "points.wreel"
    write_wreel(base, sample[0], sample[1], key)
    g = random_cloud(10)
    g.rotations[0] = [0, 0, 0, 2]
    manifest, entries = wreel_with_gaussians(
        read_wreel(base), g, stage_record={"name": "s", "version": "t"}
    )
    write_wreel(tmp_path / "x.wreel", manifest, entries, key, validate=False)
    assert "bad_quaternion" in validate_wreel(tmp_path / "x.wreel").codes()
