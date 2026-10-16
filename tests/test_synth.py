from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stages.reconstruct.mock import quat_to_matrix
from tests.media import HAS_BLENDER
from tools import synth
from tools.blender.convert import (
    actor_quat_zup_to_yup,
    blender_actor_yaw,
    intrinsics,
    quat_zup_to_yup,
    radial_to_planar,
    rotate,
    vec_zup_to_yup,
)


def close(a: object, b: object, tol: float = 1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol))


# --------------------------------------------------------------------------- conversions


def test_axes_conversion() -> None:
    assert vec_zup_to_yup((1.0, 2.0, 3.0)) == (1.0, 3.0, -2.0)
    # An unrotated Blender camera looks straight down (-Z); in WorldReel that is -Y.
    q = quat_zup_to_yup((1.0, 0.0, 0.0, 0.0))
    assert close(rotate(q, (0.0, 0.0, -1.0)), (0.0, -1.0, 0.0))


def test_camera_rotation_matches_vector_conversion() -> None:
    """Rotating then converting equals converting then rotating, for any Blender rotation."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        w, x, y, z = (float(c) for c in rng.normal(size=4))
        n = math.sqrt(w * w + x * x + y * y + z * z)
        qb = (w / n, x / n, y / n, z / n)
        v = (float(rng.normal()), float(rng.normal()), float(rng.normal()))
        rotated_in_blender = rotate((qb[1], qb[2], qb[3], qb[0]), v)
        assert close(vec_zup_to_yup(rotated_in_blender), rotate(quat_zup_to_yup(qb), v), 1e-9)


def test_actor_conversion() -> None:
    assert close(actor_quat_zup_to_yup((1.0, 0.0, 0.0, 0.0)), (0.0, 0.0, 0.0, 1.0))
    # Moving along Blender +X ( = WorldReel +X): the actor must face +X in WorldReel.
    yaw = blender_actor_yaw(1.0, 0.0)
    qb = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    q = actor_quat_zup_to_yup(qb)
    assert close(rotate(q, (0.0, 0.0, -1.0)), (1.0, 0.0, 0.0), 1e-9)
    assert close(rotate(q, (0.0, 1.0, 0.0)), (0.0, 1.0, 0.0), 1e-9)


def test_intrinsics_and_radial() -> None:
    k = intrinsics(30.0, 36.0, 24.0, "AUTO", 320, 240)
    assert k["fx"] == pytest.approx(30 / 36 * 320)
    assert (k["cx"], k["cy"]) == (159.5, 119.5)
    assert radial_to_planar(5.0, k["cx"], k["cy"], k) == pytest.approx(5.0)
    assert radial_to_planar(5.0, 0.0, 0.0, k) < 5.0


# ------------------------------------------------------------------ self-check (no Blender)


def analytic_scene(tmp: Path, *, broken: bool = False) -> Path:
    """Camera 3 m up looking down 30°, flat ground; depth computed exactly."""
    w, h = 64, 48
    k = intrinsics(30.0, 36.0, 24.0, "AUTO", w, h)
    frames = []
    (tmp / "rgb").mkdir(parents=True)
    (tmp / "depth").mkdir()
    for i, x in enumerate((0.0, 0.5)):
        pitch = math.radians(-30)
        rot = [math.sin(pitch / 2), 0.0, 0.0, math.cos(pitch / 2)]
        pos = [x, 3.0, 0.0]
        r = quat_to_matrix(rot)
        v, u = np.mgrid[0:h, 0:w]
        dirs = np.stack(
            [(u - k["cx"]) / k["fx"], -(v - k["cy"]) / k["fy"], -np.ones_like(u, float)], axis=-1
        )
        world_dirs = dirs @ r.T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(world_dirs[..., 1] < 0, -pos[1] / world_dirs[..., 1], 0.0)
        depth = s.astype(np.float32)  # planar depth = s because dirs have z = -1
        np.save(tmp / "depth" / f"{i + 1:04d}.npy", depth)
        (tmp / "rgb" / f"{i + 1:04d}.png").write_bytes(b"png")
        cam_pos = [x, 3.0 if not (broken and i == 1) else 3.6, 0.0]
        frames.append(
            {
                "frame": i + 1,
                "t": i * 0.1,
                "rgb": f"rgb/{i + 1:04d}.png",
                "depth": f"depth/{i + 1:04d}.npy",
                "camera": {"position": cam_pos, "rotation": rot},
            }
        )
    gt = {
        "intrinsics": k,
        "frames": frames,
        "actors": [],
        "seed": 0,
        "fps": 10,
        "generator_version": "test",
    }
    (tmp / "gt.json").write_text(json.dumps(gt))
    return tmp


def test_check_accepts_consistent_ground_truth(tmp_path: Path) -> None:
    report = synth.check(analytic_scene(tmp_path))
    assert report.ok, report.errors
    assert report.stats["reprojection_median_rel_error"] < 1e-3


def test_check_catches_wrong_camera_pose(tmp_path: Path) -> None:
    report = synth.check(analytic_scene(tmp_path, broken=True))
    assert not report.ok


# --------------------------------------------------------------------------- real Blender render


@pytest.mark.skipif(not HAS_BLENDER, reason="Blender not installed")
def test_blender_render_has_consistent_ground_truth(tmp_path: Path) -> None:
    out = synth.render(
        tmp_path / "scene",
        seed=3,
        frames=6,
        fps=6,
        width=96,
        height=72,
        engine="EEVEE",
        samples=4,
        actors=2,
        boxes=3,
        timeout=600,
    )
    gt = json.loads((out / "gt.json").read_text())
    assert len(gt["frames"]) == 6
    assert len(list((out / "rgb").glob("*.png"))) == 6
    assert np.load(out / "depth" / "0001.npy").shape == (72, 96)
    assert len(gt["actors"]) == 2 and len(gt["actors"][0]["track"]["times"]) == 6
    report = synth.check(out)
    assert report.ok, (report.errors, report.stats)
    assert report.stats["actor_a0_facing_cos"] > 0.99
