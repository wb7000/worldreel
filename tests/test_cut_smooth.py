"""Tests for cut smoothing: exposure match + face-aware cross-dissolve (D-9.21)."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from stages.generate.cut_smooth import cross_dissolve, match_levels, smooth_cuts

U8 = npt.NDArray[np.uint8]
F64 = npt.NDArray[np.float64]

_OFFSETS = np.array([[-8.0, -8.0], [8.0, -8.0], [0.0, 0.0], [-8.0, 8.0], [8.0, 8.0]])


def _two_takes(
    n_each: int = 30,
    pos_a: tuple[float, float] = (60.0, 60.0),
    pos_b: tuple[float, float] = (110.0, 60.0),
    bg_b: int = 0,
    blob_a: int = 200,
    blob_b: int = 200,
) -> tuple[list[U8], F64]:
    """Two flat takes of a bright square: take B differs in position and level."""
    frames: list[U8] = []
    lms = []
    for t in range(2 * n_each):
        take_b = t >= n_each
        center = np.array(pos_b if take_b else pos_a)
        bg = 40 + (bg_b if take_b else 0)
        blob = blob_b if take_b else blob_a
        frame = np.full((140, 180, 3), bg, dtype=np.uint8)
        c = center
        frame[int(c[1]) - 12 : int(c[1]) + 12, int(c[0]) - 12 : int(c[0]) + 12] = blob
        frames.append(frame)
        lms.append(_OFFSETS + center)
    return frames, np.stack(lms)


def _wall_mean(frame: U8) -> float:
    h, w = frame.shape[:2]
    cols = np.zeros((h, w), dtype=bool)
    cols[:, : w // 4] = True
    cols[:, -w // 4 :] = True
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[cols].mean())


def test_match_levels_equalises_wall_between_takes() -> None:
    frames, _ = _two_takes(bg_b=24)
    takes = [(0, 30), (30, 60)]
    out, gains = match_levels(frames, takes)
    assert len(out) == len(frames)
    assert len(gains) == 2
    assert abs(gains[0] - 1.30) < 0.01  # brightened to the shared target
    assert abs(gains[1] - 0.81) < 0.01
    wa = float(np.mean([_wall_mean(out[t]) for t in range(30)]))
    wb = float(np.mean([_wall_mean(out[t]) for t in range(30, 60)]))
    assert abs(wa - wb) < 0.5, f"wall means {wa:.1f} vs {wb:.1f}"


def test_match_levels_is_a_uniform_gain() -> None:
    frames, _ = _two_takes(bg_b=24)
    out, gains = match_levels(frames, [(0, 30), (30, 60)])
    expected = np.round(40 * gains[0])
    assert float(out[5][3, 3, 0]) == expected  # background pixel
    assert float(out[5][60, 60, 0]) == float(
        np.round(np.clip(200 * gains[0], 0, 255))
    )  # blob scaled by the same factor


def test_match_levels_no_takes_returns_input() -> None:
    frames, _ = _two_takes()
    out, gains = match_levels(frames, [])
    assert len(out) == len(frames)
    assert all(a is b for a, b in zip(out, frames, strict=True))
    assert gains == []


def test_cross_dissolve_length_and_starts() -> None:
    frames, lm = _two_takes(n_each=30)
    out, starts = cross_dissolve(frames, lm, [30], k=6)
    assert len(out) == 60 - 6
    assert starts == [24]
    out2, starts2 = cross_dissolve(frames, lm, [20, 44], k=6)
    assert len(out2) == 60 - 12
    assert starts2 == [14, 32]


def test_cross_dissolve_validates_windows() -> None:
    frames, lm = _two_takes(n_each=30)
    for bad_cuts in ([3], [30, 36], [57], [-1]):
        try:
            cross_dissolve(frames, lm, bad_cuts, k=6)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for cuts {bad_cuts}")
    try:
        cross_dissolve(frames, lm, [30], k=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for dissolve 0")


def test_cross_dissolve_length_mismatch() -> None:
    frames, lm = _two_takes(n_each=10)
    try:
        cross_dissolve(frames[:5], lm, [10], k=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for frame/landmark mismatch")


def test_dissolve_replaces_the_hard_cut_with_a_ramp() -> None:
    # Take B is much brighter and its square sits elsewhere: the raw cut jumps.
    frames, lm = _two_takes(bg_b=40, blob_b=230)
    raw_jump = float(np.abs(frames[30].astype(np.float32) - frames[29].astype(np.float32)).mean())
    out, _ = cross_dissolve(frames, lm, [30], k=6)
    steps = [
        float(np.abs(out[i + 1].astype(np.float32) - out[i].astype(np.float32)).mean())
        for i in range(23, 33)
    ]
    assert raw_jump > 15.0, f"synthetic cut should be big, got {raw_jump:.1f}"
    assert max(steps) < 0.5 * raw_jump, f"ramp steps {max(steps):.1f} vs cut {raw_jump:.1f}"


def test_dissolve_does_not_double_image_the_face() -> None:
    # Disjoint squares: a naive crossfade shows BOTH at mid-window; the
    # face-aware dissolve must show one, on the travelling path.
    frames, lm = _two_takes()  # positions 50 px apart
    out, starts = cross_dissolve(frames, lm, [30], k=6)
    s = starts[0]
    single = 24 * 24

    def blob_px(frame: U8) -> int:
        return int(np.count_nonzero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 130))

    mid = [blob_px(out[s + i]) for i in range(1, 5)]  # alpha 0.33..0.83
    assert max(mid) < 1.5 * single, f"double image: bright px {mid} (single={single})"


def test_dissolve_face_travels_between_the_takes() -> None:
    frames, lm = _two_takes()
    out, starts = cross_dissolve(frames, lm, [30], k=6)
    s = starts[0]

    def centroid(frame: U8) -> tuple[float, float]:
        ys, xs = np.nonzero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 130)
        return float(xs.mean()), float(ys.mean())

    xs = [centroid(out[s + i])[0] for i in range(6)]
    assert abs(xs[0] - 60.0) < 5.0, f"first dissolve frame at x={xs[0]:.1f}, want 60"
    assert abs(xs[-1] - 110.0) < 5.0, f"last dissolve frame at x={xs[-1]:.1f}, want 110"
    diffs = [xs[i + 1] - xs[i] for i in range(5)]
    assert all(d >= -1.0 for d in diffs), f"face moved backwards: {diffs}"
    # no teleport: nothing jumps the whole gap in one frame
    assert max(diffs) < 35.0, f"teleport step {max(diffs):.1f} px"
    # continuity with the pure frames on either side
    assert abs(centroid(out[s - 1])[0] - 60.0) < 5.0
    assert abs(centroid(out[s + 6])[0] - 110.0) < 5.0


def test_smooth_cuts_end_to_end() -> None:
    frames, lm = _two_takes(bg_b=24, pos_b=(110.0, 70.0))
    out, stats = smooth_cuts(frames, lm, dissolve=6, levels=True)
    assert stats.cuts == [30]
    assert stats.takes == [(0, 30), (30, 60)]
    assert len(stats.gains) == 2
    assert stats.dissolve_out == [24]
    assert stats.frames_out == len(out) == 54
    # wall matched before the dissolve, face present in every frame
    assert abs(_wall_mean(out[0]) - _wall_mean(out[53])) < 12.0
    for frame in out:
        assert np.count_nonzero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 130) > 100
