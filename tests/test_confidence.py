from __future__ import annotations

import numpy as np

from stages.package.confidence import (
    GUESSED,
    INFERRED,
    OBSERVED,
    SupportConfig,
    View,
    classify,
    view_support,
)
from stages.reconstruct.planesweep import cv_pose

K = np.array([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]])


def _view(x: float, mask: np.ndarray | None = None) -> View:
    # WorldReel camera at (x, 0, 5) looking down -Z (identity rotation).
    r, t = cv_pose([x, 0.0, 5.0], [0.0, 0.0, 0.0, 1.0])
    return View(r, t, K, 320, 240, mask)


def test_wall_point_seen_from_many_angles_is_observed_and_hidden_point_is_guessed() -> None:
    wall = np.array(
        [[gx, gy, 0.0] for gx in np.linspace(-1, 1, 41) for gy in np.linspace(-1, 1, 41)]
    )
    hidden = np.array([[0.0, 0.0, -1.0]])  # directly behind the wall for every camera
    pos = np.vstack([wall, hidden])
    opacity = np.ones(len(pos))
    views = [_view(x) for x in np.linspace(-1.5, 1.5, 10)]
    n, spread = view_support(pos, opacity, views, SupportConfig())
    cls, conf = classify(n, spread, SupportConfig())
    centre = 20 * 41 + 20  # wall point (0, 0, 0)
    assert n[centre] == 10 and spread[centre] > 10
    assert cls[centre] == OBSERVED and conf[centre] > 0.9
    assert n[-1] == 0 and cls[-1] == GUESSED and conf[-1] == np.float32(0.1)


def test_single_view_or_no_parallax_is_inferred() -> None:
    pos = np.array([[0.0, 0.0, 0.0]])
    one = view_support(pos, np.ones(1), [_view(0.0)], SupportConfig())
    assert classify(*one, SupportConfig())[0][0] == INFERRED
    # Ten views from the same spot: count is full but there is no parallax.
    same = view_support(pos, np.ones(1), [_view(0.0)] * 10, SupportConfig())
    assert classify(*same, SupportConfig())[0][0] == INFERRED
    # Ablation without the spread term would call it observed.
    no_spread = SupportConfig(use_spread=False)
    assert classify(*same, no_spread)[0][0] == OBSERVED


def test_masked_person_pixels_do_not_count_as_evidence() -> None:
    pos = np.array([[0.0, 0.0, 0.0]])
    mask = np.zeros((240, 320), bool)
    mask[100:140, 140:180] = True  # covers the image centre where the point projects
    n, _ = view_support(pos, np.ones(1), [_view(0.0, mask), _view(0.0)], SupportConfig())
    assert n[0] == 1


def test_relabel_flags_floater_in_front_of_stereo_surface_but_not_in_stereo_gaps() -> None:
    from stages.package.confidence import relabel_gaussians

    wall = np.array(
        [[gx, gy, 0.0] for gx in np.linspace(-1, 1, 41) for gy in np.linspace(-1, 1, 41)]
    )
    stereo = wall[wall[:, 0] < 0.5]  # stereo found the left part of the wall only
    floater = [0.0, 0.0, 2.5]  # hangs in front of the stereo-covered wall
    gap_point = [0.9, 0.0, 0.0]  # on the wall, where stereo has a hole
    pos = np.vstack([wall, [floater, gap_point]])
    views = [_view(x) for x in np.linspace(-1.5, 1.5, 10)]
    fg = np.ones(len(pos), bool)
    cls, _, _ = relabel_gaussians(pos, np.ones(len(pos)), fg, views, stereo, extent=2.0)
    assert cls[-2] != OBSERVED  # floater
    assert cls[-1] == OBSERVED  # gap in stereo is not evidence against it
