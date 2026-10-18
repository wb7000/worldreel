"""3D lifting of person masks: synthetic geometry with an exact answer, plus the Blender scene."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from eval.geometry import quat_to_matrix
from stages.actors.lift import (
    fit_ground_plane,
    intersect_plane,
    lift_track,
    mask_foot_and_head,
    pixel_ray,
)
from tests.media import HAS_BLENDER

K = {"fx": 200.0, "fy": 200.0, "cx": 159.5, "cy": 119.5, "width": 320, "height": 240}


def look_down_pose(eye: list[float], pitch_deg: float = -20.0) -> dict[str, list[float]]:
    a = math.radians(pitch_deg) / 2
    return {"position": eye, "rotation": [math.sin(a), 0.0, 0.0, math.cos(a)]}


def project(p: np.ndarray, pose: dict[str, list[float]]) -> tuple[float, float]:
    cam = quat_to_matrix(pose["rotation"]).T @ (p - np.asarray(pose["position"]))
    z = -cam[2]
    return cam[0] / z * K["fx"] + K["cx"], -cam[1] / z * K["fy"] + K["cy"]


def test_ground_plane_from_noisy_points_with_clutter() -> None:
    rng = np.random.default_rng(0)
    floor = np.c_[
        rng.uniform(-5, 5, 2000), rng.normal(-1.5, 0.003, 2000), rng.uniform(-10, -1, 2000)
    ]
    wall = np.c_[rng.uniform(-5, 5, 1500), rng.uniform(-1.5, 2, 1500), np.full(1500, -10.0)]
    boxes = rng.uniform([-3, -1.5, -8], [3, 0.5, -3], (800, 3))
    plane = fit_ground_plane(np.concatenate([floor, wall, boxes]))
    assert plane.normal[1] > 0.999
    assert plane.offset == pytest.approx(-1.5, abs=0.01)


def test_ray_hits_floor_point() -> None:
    pose = look_down_pose([0.0, 0.0, 0.0])
    target = np.array([0.7, -1.5, -6.0])
    u, v = project(target, pose)
    o, d = pixel_ray(u, v, K, pose["position"], pose["rotation"])
    plane = fit_ground_plane(
        np.c_[
            np.random.default_rng(1).uniform(-5, 5, 300),
            np.full(300, -1.5),
            np.random.default_rng(2).uniform(-9, -2, 300),
        ]
    )
    hit = intersect_plane(o, d, plane)
    assert hit is not None
    assert np.allclose(hit, target, atol=1e-6)


def test_mask_foot_and_head() -> None:
    m = np.zeros((20, 20), bool)
    m[5:15, 8:11] = True
    result = mask_foot_and_head(m)
    assert result is not None
    (fu, fv), (hu, hv) = result
    assert (fu, fv) == (9.0, 14.5)
    assert (hu, hv) == (9.0, 4.5)
    assert mask_foot_and_head(np.zeros((4, 4), bool)) is None


def test_lift_walking_person_recovers_path_height_and_facing() -> None:
    rng = np.random.default_rng(3)
    floor_pts = np.c_[rng.uniform(-6, 6, 500), np.full(500, -1.5), rng.uniform(-12, -2, 500)]
    plane = fit_ground_plane(floor_pts)
    poses = [look_down_pose([0.0, 0.0, 0.0]) for _ in range(10)]
    times = [i * 0.25 for i in range(10)]
    masks = {}
    truth = []
    for i, t in enumerate(times):
        feet = np.array([-1.0 + 0.8 * t, -1.5, -6.0])  # walking along +x at 0.8 m/s
        truth.append(feet)
        mask = np.zeros((240, 320), bool)
        fu, fv = project(feet, poses[i])
        _, hv = project(feet + np.array([0, 1.75, 0]), poses[i])
        mask[round(hv) : round(fv), int(fu) - 3 : int(fu) + 4] = True
        masks[i] = mask
    track = lift_track(masks, poses, times, K, plane, body_half_depth_ratio=0.0)
    assert track is not None and len(track.positions) == 10
    shifted = lift_track(masks, poses, times, K, plane)
    assert shifted is not None
    # Default: feet move away from the camera (here -z) by 0.075 x height along the floor.
    delta = np.asarray(shifted.positions) - np.asarray(track.positions)
    assert np.allclose(np.linalg.norm(delta, axis=1), 0.075 * shifted.height, atol=1e-3)
    assert np.all(delta[:, 2] < 0)
    err = np.linalg.norm(np.asarray(track.positions) - np.asarray(truth), axis=1)
    assert err.max() < 0.08  # metres, from 1-pixel mask quantisation at 6 m
    assert track.height == pytest.approx(1.75, abs=0.1)
    forward = quat_to_matrix(track.rotations[5]) @ np.array([0.0, 0.0, -1.0])
    assert forward[0] > 0.99  # facing +x, the walking direction


@pytest.mark.skipif(not HAS_BLENDER, reason="Blender not installed")
def test_lift_on_blender_ground_truth(tmp_path: Path) -> None:
    """With exact masks, cameras and floor points, lifting recovers the actors' true paths."""
    from tools import synth

    scene = synth.render(
        tmp_path / "s",
        seed=21,
        frames=12,
        fps=6,
        width=320,
        height=240,
        engine="EEVEE",
        samples=4,
        actors=2,
        boxes=3,
        timeout=900,
    )
    gt = json.loads((scene / "gt.json").read_text())
    poses = [f["camera"] for f in gt["frames"]]
    times = [f["t"] for f in gt["frames"]]
    floor = []
    for f in gt["frames"][:3]:
        pts = synth.backproject(
            np.load(scene / f["depth"]),
            gt["intrinsics"],
            f["camera"]["position"],
            f["camera"]["rotation"],
        )
        floor.append(pts[::50])
    plane = fit_ground_plane(np.concatenate(floor))
    labels = [synth.actor_masks(scene, i) for i in range(len(gt["frames"]))]
    for k, actor in enumerate(gt["actors"], start=1):
        masks = {i: lab == k for i, lab in enumerate(labels) if (lab == k).sum() > 50}
        track = lift_track(masks, poses, times, gt["intrinsics"], plane)
        assert track is not None
        truth = np.asarray(actor["track"]["positions"])[track.frames]
        err = np.linalg.norm(np.asarray(track.positions) - truth, axis=1)
        # The mask bottom is the nearest visible edge of the body, not its centre: up to ~radius.
        assert np.median(err) < 0.35, (actor["id"], err)


def test_is_moving_separates_walkers_from_static_tracks() -> None:
    from stages.package.actors import is_moving

    still = np.array([[0.0, 0, 0], [0.05, 0, 0.02], [0.1, 0, -0.03], [0.02, 0, 0.01]])
    walk = np.array([[0.0, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [1.5, 0, 0]])
    assert not is_moving(still, 1.75)
    assert is_moving(walk, 1.75)
    assert not is_moving(np.zeros((0, 3)), 1.75)


def test_height_outliers_need_three_people() -> None:
    from stages.package.actors import height_inliers

    assert height_inliers([0.66, 0.80, 1.93]) == [True, True, False]  # mirror reflection case
    assert height_inliers([1.1, 1.75, 1.8]) == [True, True, True]  # child with adults
    assert height_inliers([0.7, 1.9]) == [True, True]  # two people: no reference
