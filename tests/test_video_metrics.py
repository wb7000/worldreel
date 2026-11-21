"""The shared clip metrics (D-9.26).

The motion metric is the one that matters here: the D-9.25 adapter render
passed every other gate and was still "a still image that shakes". These tests
pin the thresholds and the properties so a future change cannot quietly turn
the motion gate into a no-op.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.video_metrics import STILL_DIFF, ClipMetrics


def _metrics(motion: list[float], identity: list[float]) -> ClipMetrics:
    m = ClipMetrics(path="x")
    m.frames = len(motion) + 1
    m.motion = motion
    m.identity = identity
    return m


def test_still_frames_count_the_frames_below_the_threshold() -> None:
    m = _metrics([0.1, 0.4, 0.6, 2.0], [])
    assert m.still_frames == 2
    assert m.still_fraction == 0.5
    assert m.motion_mean == 0.775


def test_a_mostly_still_clip_is_flagged_as_static() -> None:
    # The D-9.25 adapter render: 211 of 239 still -> static.
    assert _metrics([0.1] * 211 + [2.0] * 28, []).is_static
    # The base model: none still -> a real video.
    assert not _metrics([2.0] * 239, []).is_static


def test_identity_properties() -> None:
    m = _metrics([], [0.9, 0.8, 0.75, 0.95])
    assert m.identity_mean == pytest.approx(0.85)
    assert m.identity_min == pytest.approx(0.75)
    assert m.identity_below == 1  # only the 0.75 frame


def test_geometry_cv_is_the_relative_spread() -> None:
    m = ClipMetrics(path="x")
    m.iod = [100.0, 110.0]
    m.face_heights = [200.0, 200.0]
    assert abs(m.iod_cv - 0.0476) < 0.001
    assert m.height_cv == 0.0
    assert m.iod_mean == 105.0


def test_empty_metrics_do_not_raise() -> None:
    m = ClipMetrics(path="x")
    assert m.identity_mean != m.identity_mean  # NaN
    assert m.motion_mean == 0.0
    assert m.still_frames == 0
    assert not m.is_static


def test_still_threshold_is_the_documented_one() -> None:
    assert STILL_DIFF == 0.5
    assert np.isfinite(STILL_DIFF)
