"""Segment assembly: reorder + trim without generation loss (D-9.32).

The assembly order is free (seeds carry no narrative) and a permutation
search over the pair jump matrix picks the order whose boundaries jump the
least.  These tests pin the mechanics: order applied, trims applied at the
right boundary, output cuts reported for `cut_smooth --bounds`, and every
config error rejected loudly (an empty or misplaced take would silently
corrupt the delivery timeline).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from tools.assemble_segments import _parse_trims, assemble

U8 = npt.NDArray[np.uint8]


def _frames(n: int) -> list[U8]:
    """n single-value frames: value == source index, so provenance is visible."""
    return [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]


def _values(frames: list[U8]) -> list[int]:
    return [int(f[0, 0, 0]) for f in frames]


def test_reorder_only_changes_order_not_content() -> None:
    frames = _frames(12)  # 3 segments of 4
    out, cuts = assemble(frames, [4, 8], order=[2, 0, 1], trims=[(0, 0), (0, 0)])
    assert _values(out) == [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7]
    assert cuts == [4, 8]
    assert len(out) == 12


def test_trims_drop_frames_at_each_boundary() -> None:
    frames = _frames(12)  # segments [0..4), [4..8), [8..12)
    out, cuts = assemble(frames, [4, 8], order=[0, 1, 2], trims=[(2, 1), (1, 3)])
    # seg0 loses tail 2 -> 0,1 ; seg1 loses head 1 and tail 1 -> 5,6 ;
    # seg2 loses head 3 -> 11
    assert _values(out) == [0, 1, 5, 6, 11]
    assert cuts == [2, 4]  # boundaries in the OUTPUT timeline


def test_reported_cuts_match_output_lengths() -> None:
    # The D-9.32 spec: 5x48 frames, order [3,4,2,1,0], trims 2:5 5:2 5:5 0:2
    frames = _frames(240)
    bounds = [48, 96, 144, 192]
    order = [3, 4, 2, 1, 0]
    trims = [(2, 5), (5, 2), (5, 5), (0, 2)]
    out, cuts = assemble(frames, bounds, order, trims)
    assert cuts == [46, 84, 125, 168]
    assert len(out) == 214
    # segment lengths: 46, 38, 41, 43, 46
    assert np.diff([0, *cuts, len(out)]).tolist() == [46, 38, 41, 43, 46]
    # first output frame is segment 3's first kept frame (index 144)
    assert _values(out)[0] == 144
    # and the last is segment 0's last frame (index 47, no tail trim on last)
    assert _values(out)[-1] == 47


def test_unsorted_or_out_of_range_bounds_raise() -> None:
    frames = _frames(12)
    with pytest.raises(ValueError, match="bounds"):
        assemble(frames, [8, 4], order=[0, 1, 2], trims=[(0, 0), (0, 0)])
    with pytest.raises(ValueError, match="bounds"):
        assemble(frames, [4, 12], order=[0, 1, 2], trims=[(0, 0), (0, 0)])
    with pytest.raises(ValueError, match="bounds"):
        assemble(frames, [4, 4], order=[0, 1, 2], trims=[(0, 0), (0, 0)])


def test_non_permutation_order_raises() -> None:
    frames = _frames(12)
    with pytest.raises(ValueError, match="permutation"):
        assemble(frames, [4, 8], order=[0, 1, 1], trims=[(0, 0), (0, 0)])
    with pytest.raises(ValueError, match="permutation"):
        assemble(frames, [4, 8], order=[0, 1], trims=[(0, 0)])


def test_wrong_trim_count_or_negative_trim_raises() -> None:
    frames = _frames(12)
    with pytest.raises(ValueError, match="trim pairs"):
        assemble(frames, [4, 8], order=[0, 1, 2], trims=[(0, 0)])
    with pytest.raises(ValueError, match="non-negative"):
        assemble(frames, [4, 8], order=[0, 1, 2], trims=[(-1, 0), (0, 0)])


def test_trims_that_empty_a_segment_raise() -> None:
    frames = _frames(12)
    with pytest.raises(ValueError, match="empty segment"):
        assemble(frames, [4, 8], order=[0, 1, 2], trims=[(4, 0), (0, 0)])


def test_parse_trims_pairs() -> None:
    assert _parse_trims(["2:5", "0:2"]) == [(2, 5), (0, 2)]
    assert _parse_trims([]) == []
    with pytest.raises(SystemExit):
        _parse_trims(["2"])
    with pytest.raises(SystemExit):
        _parse_trims(["a:b"])
