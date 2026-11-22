"""Joining rendered clips must never add generation loss.

The delivery render is split over several Kaggle jobs (D-9.28), so the clips
have to be joined before the local chain runs. A stream copy concatenates the
encoded bits -- nothing is decoded, nothing is re-encoded -- and it is only
valid when the clips agree on every stream parameter. When they disagree the
join falls back to a lossless FFV1 re-encode; a lossy re-encode is never
allowed (D-9.22/23).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.join_clips import main, same_parameters

BASE = {
    "codec_name": "h264",
    "width": "832",
    "height": "480",
    "pix_fmt": "yuv420p",
    "r_frame_rate": "16/1",
    "time_base": "1/16384",
}


def test_identical_parameters_allow_a_stream_copy() -> None:
    assert same_parameters([dict(BASE), dict(BASE)])


def test_a_different_resolution_forces_the_lossless_path() -> None:
    other = dict(BASE, width="1280")
    assert not same_parameters([dict(BASE), other])


def test_a_different_frame_rate_forces_the_lossless_path() -> None:
    other = dict(BASE, r_frame_rate="24/1")
    assert not same_parameters([dict(BASE), other])


def test_a_different_codec_forces_the_lossless_path() -> None:
    other = dict(BASE, codec_name="vp9")
    assert not same_parameters([dict(BASE), other])


def test_no_clips_is_not_a_stream_copy() -> None:
    assert not same_parameters([])


def test_a_missing_clip_is_reported_before_ffmpeg_runs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path / "out.mp4"), str(missing)])
    assert "missing clip" in str(excinfo.value)
