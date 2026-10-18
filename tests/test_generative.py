from __future__ import annotations

import numpy as np
import pytest

from stages.fill.generative import LAMA_MODEL, generative_colours, lama_inpainter
from stages.fill.planes import progressive_fill


def _striped_floor_with_hole() -> tuple[np.ndarray, np.ndarray]:
    """4 x 4 m floor with 0.5 m stripes (red / blue along x) and a 0.5 m unseen hole."""
    rng = np.random.default_rng(0)
    xz = rng.uniform(-2, 2, (60000, 2))
    xz = xz[~((np.abs(xz[:, 0]) < 0.25) & (np.abs(xz[:, 1]) < 0.25))]
    pts = np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], axis=1)
    stripe = (np.floor(xz[:, 0] / 0.5) % 2).astype(bool)
    cols = np.where(stripe[:, None], [[200, 40, 40]], [[40, 40, 200]]).astype(np.uint8)
    return pts, cols


def test_only_hole_elements_are_recoloured_and_mapping_is_per_plane() -> None:
    pts, cols = _striped_floor_with_hole()
    planes, _, tier = progressive_fill(pts, cols, extent=4.0)
    calls: list[int] = []

    def fake(image: np.ndarray, hole: np.ndarray) -> np.ndarray:
        calls.append(int(hole.sum()))
        out = image.copy()
        out[hole] = (0, 255, 0)  # paint holes green
        return out

    colours, _ = generative_colours(planes, pts, cols, np.ones(len(pts)), tier, fake)
    assert calls and all(c > 0 for c in calls)
    green = np.all(colours == (0, 255, 0), axis=1)
    centre = (np.abs(tier.positions[:, 0]) < 0.12) & (np.abs(tier.positions[:, 2]) < 0.12)
    assert green[centre].all()  # the hole interior goes to the model
    assert not green.all()  # pinhole gaps between samples are interpolated, not generated


@pytest.mark.skipif(not LAMA_MODEL.is_file(), reason="LaMa model not downloaded (docs/MODELS.md)")
def test_lama_continues_the_stripes() -> None:
    pts, cols = _striped_floor_with_hole()
    planes, _, tier = progressive_fill(pts, cols, extent=4.0)
    colours, _ = generative_colours(planes, pts, cols, np.ones(len(pts)), tier, lama_inpainter())
    hole = (np.abs(tier.positions[:, 0]) < 0.2) & (np.abs(tier.positions[:, 2]) < 0.2)
    truth_red = (np.floor(tier.positions[hole, 0] / 0.5) % 2).astype(bool)
    red = colours[hole, 0].astype(int) > colours[hole, 2].astype(int)
    assert (red == truth_red).mean() > 0.7  # stripes continue through the hole
