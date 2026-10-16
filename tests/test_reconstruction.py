"""Classical reconstruction: plane sweep on an analytic scene (fast, no Blender) and the full
SfM + MVS benchmark on a small Blender scene (skipped without Blender)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from stages.reconstruct.planesweep import best_depth, sweep
from tests.media import HAS_BLENDER


def plane_views(
    depth: float,
) -> tuple[list[np.ndarray], np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """A random-textured fronto-parallel plane at `depth` seen by 3 cameras translated along x
    (OpenCV convention). Images are produced exactly with the plane homography."""
    w, h = 160, 120
    k = np.array([[120.0, 0, (w - 1) / 2], [0, 120.0, (h - 1) / 2], [0, 0, 1]])
    rng = np.random.default_rng(5)
    texture = cv2.resize(
        rng.random((60, 80)).astype(np.float32), (w * 2, h * 2), interpolation=cv2.INTER_NEAREST
    )
    texture = cv2.GaussianBlur(texture, (3, 3), 0.8)
    # Image pixel (x, y) of the reference camera → texture pixel (x + w/2, y + h/2): the 2w x 2h
    # texture covers every camera's view with margin.
    ref_cam = np.array([[1.0, 0, w / 2], [0, 1.0, h / 2], [0, 0, 1]])
    poses = [(np.eye(3), np.array([tx, 0.0, 0.0])) for tx in (0.0, -0.3, 0.3)]
    images = []
    n = np.array([[0.0, 0.0, 1.0]])
    for r, t in poses:
        # pixel in this camera → point on the plane in reference camera → texture coordinates
        hom = k @ (r + t[:, None] @ n / depth) @ np.linalg.inv(k)
        to_texture = ref_cam @ np.linalg.inv(hom)
        img = cv2.warpPerspective(
            texture, np.linalg.inv(to_texture), (w, h), flags=cv2.INTER_LINEAR
        )
        images.append(img)
    return images, k, poses


def test_plane_sweep_recovers_plane_depth() -> None:
    true_depth = 4.0
    images, k, poses = plane_views(true_depth)
    depths = 1.0 / np.linspace(1 / 2.0, 1 / 10.0, 64)
    (r0, t0) = poses[0]
    volume, _ = sweep(images[0], [(images[i], *poses[i]) for i in (1, 2)], k, r0, t0, depths, 7)
    depth, cost = best_depth(volume, depths)
    centre = depth[30:90, 40:120]
    good = centre[centre > 0]
    assert good.size > 0.9 * centre.size
    assert np.median(np.abs(good - true_depth) / true_depth) < 0.02
    assert np.median(1 - cost[30:90, 40:120]) > 0.9


@pytest.mark.skipif(not HAS_BLENDER, reason="Blender not installed")
def test_classical_reconstruction_benchmark(tmp_path: Path) -> None:
    """The Phase 3a benchmark at its real size (about 2 minutes on the dev laptop). Smaller
    renders (320x240, 16 frames) measurably fail the orientation threshold; see DECISIONS."""
    from eval.benchmark import run
    from tools import synth

    scene = synth.render(
        tmp_path / "scene",
        seed=12,
        frames=30,
        fps=12,
        width=480,
        height=360,
        engine="EEVEE",
        samples=16,
        actors=0,
        boxes=7,
        eval_views=4,
        timeout=1800,
    )
    report = run(scene, tmp_path / "work", calibrated=True)
    failed = {n: c for n, c in report["checks"].items() if not c["pass"]}
    assert not failed, failed
    assert report["wreel_valid"]
