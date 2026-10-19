from __future__ import annotations

import math

import numpy as np

from formats.wreel import PointCloud
from stages.export.render import CameraPose, Scene, camera_at, fog_amount, render

BG = np.array([0.1, 0.2, 0.3])


def _scene(
    g_pos: list[list[float]], g_rgb: list[list[float]], g_fog: list[float], sigma: float = 0.05
) -> Scene:
    n = len(g_pos)
    cov = np.tile(np.eye(3) * sigma**2, (n, 1, 1))
    return Scene(
        np.array(g_pos, float),
        cov,
        np.array(g_rgb, float),
        np.full(n, 0.99),
        np.array(g_fog, float),
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros(0),
        np.zeros(0),
        [],
        {},
        BG,
    )


CAM = CameraPose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 60.0)  # at origin looking down -Z


def test_splat_lands_at_the_projected_pixel_with_its_depth() -> None:
    w, h = 64, 48
    f = (h / 2) / math.tan(math.radians(30))
    # 1 m to the right of the axis, 4 m away: u = cx + f * 1 / 4
    fr = render(_scene([[1.0, 0.0, -4.0]], [[1.0, 0.0, 0.0]], [0.0]), CAM, 0.0, w, h)
    ys, xs = np.nonzero(fr.rgb[..., 0] > 200)
    assert abs(xs.mean() + 0.5 - (w / 2 + f / 4)) < 1.0 and abs(ys.mean() + 0.5 - h / 2) < 1.0
    assert abs(fr.depth[int(ys.mean()), int(xs.mean())] - 4.0) < 1e-6
    # Background elsewhere, depth 0 for "nothing".
    assert np.allclose(fr.rgb[0, 0] / 255.0, BG, atol=1 / 255) and fr.depth[0, 0] == 0.0


def test_observed_splats_are_clear_and_guessed_ones_fogged() -> None:
    assert fog_amount(np.array([0]), np.array([0.9]))[0] == 0.0
    assert abs(fog_amount(np.array([2]), np.array([0.9]))[0] - 0.6) < 1e-9
    assert fog_amount(np.array([1]), np.array([0.0]))[0] == 0.92
    clear = render(
        _scene([[0.0, 0.0, -3.0]], [[0.0, 1.0, 0.0]], [0.0], sigma=0.5), CAM, 0.0, 32, 32
    )
    foggy = render(
        _scene([[0.0, 0.0, -3.0]], [[0.0, 1.0, 0.0]], [0.9], sigma=0.5), CAM, 0.0, 32, 32
    )
    assert clear.rgb[16, 16, 1] > 240 and clear.rgb[16, 16, 0] < 15
    assert foggy.rgb[16, 16, 0] > 120  # pulled towards the lavender fog


def test_actor_points_make_a_mask_and_hide_what_is_behind() -> None:
    scene = _scene([[0.0, 0.0, -6.0]], [[1.0, 1.0, 1.0]], [0.0], sigma=0.5)  # wall behind
    cloud = PointCloud(
        np.array([[0.0, 0.0, 0.0]], np.float32),
        np.array([[255, 0, 0]], np.uint8),
        np.zeros(1, np.uint8),
        np.ones(1, np.float32),
    )
    scene.actors = [("a1", cloud, 0.3)]
    scene.tracks = {
        "a1": {
            "times": [0.0, 1.0],
            "positions": [[0, 0, -3], [0, 0, -3]],
            "rotations": [[0, 0, 0, 1], [0, 0, 0, 1]],
        }
    }
    fr = render(scene, CAM, 0.5, 32, 32)
    assert fr.actor_ids[16, 16] == 1 and fr.rgb[16, 16, 0] > 240 and fr.rgb[16, 16, 1] < 15
    assert abs(fr.depth[16, 16] - 3.0) < 1e-6
    # Outside the track's time range the actor is not drawn.
    assert render(scene, CAM, 2.0, 32, 32).actor_ids.max() == 0


def test_camera_path_interpolates() -> None:
    manifest = {
        "cameras": [
            {
                "id": "c",
                "keyframes": [
                    {"t": 0.0, "position": [0, 0, 0], "rotation": [0, 0, 0, 1], "fov_deg": 40},
                    {"t": 2.0, "position": [2, 0, 0], "rotation": [0, 0, 0, 1], "fov_deg": 60},
                ],
            }
        ]
    }
    pose = camera_at(manifest, "c", 1.0)
    assert np.allclose(pose.position, [1, 0, 0]) and abs(pose.fov_deg - 50) < 1e-9
    assert np.allclose(camera_at(manifest, "c", 5.0).position, [2, 0, 0])  # clamps at the end
