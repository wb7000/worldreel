"""Tests for temporal face stabilization (D-9.20)."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from stages.generate.face_stabilize import (
    _ecc_affine,
    _full_from_local,
    _local_from_full,
    _oval_mask,
    _similarity,
    lowpass,
    split_at_cuts,
    stabilize,
)

U8 = npt.NDArray[np.uint8]
F64 = npt.NDArray[np.float64]


def _trajectory(n: int, jitter: float, seed: int = 0) -> F64:
    """Five landmark points drifting smoothly plus per-frame jitter."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    base = np.array([[30.0, 30.0], [70.0, 30.0], [50.0, 55.0], [35.0, 75.0], [65.0, 75.0]])
    drift = np.stack([10.0 * np.sin(t), 6.0 * np.cos(t)], axis=1)  # (n, 2)
    noise = rng.normal(0.0, jitter, size=(n, 5, 2))
    return base[None, :, :] + drift[:, None, :] + noise


def test_lowpass_preserves_constant_trajectory() -> None:
    lm = np.broadcast_to(np.array([40.0, 50.0]), (20, 5, 2)).copy()
    out = lowpass(lm, 2.0)
    np.testing.assert_allclose(out, lm, atol=1e-9)


def test_lowpass_removes_frame_to_frame_wobble() -> None:
    # A face that jumps +-4 px every frame: no real motion can do that at 16 fps.
    lm = np.broadcast_to(np.array([40.0, 50.0]), (40, 5, 2)).copy()
    lm[1::2] += 4.0
    out = lowpass(lm, 2.0)
    step = np.linalg.norm(np.diff(out, axis=0), axis=-1)
    raw_step = np.linalg.norm(np.diff(lm, axis=0), axis=-1)
    assert step.mean() < 0.2 * raw_step.mean()


def test_lowpass_keeps_slow_drift() -> None:
    t = np.arange(60, dtype=np.float64)
    lm = np.zeros((60, 5, 2))
    lm[..., 0] = 0.5 * t[:, None]  # steady 0.5 px/frame drift
    lm[..., 1] = 20.0
    out = lowpass(lm, 2.0)
    mid = slice(10, 50)
    np.testing.assert_allclose(out[mid, 0, 0], lm[mid, 0, 0], atol=0.5)


def test_split_at_cuts_finds_the_take_boundary() -> None:
    lm = _trajectory(40, jitter=1.0, seed=2)
    lm[20:] += np.array([40.0, -25.0])  # a jump far beyond within-take motion
    segments = split_at_cuts(lm, threshold=12.0)
    assert segments == [(0, 20), (20, 40)]


def test_split_at_cuts_single_take_when_quiet() -> None:
    lm = _trajectory(40, jitter=2.0, seed=3)
    assert split_at_cuts(lm, threshold=12.0) == [(0, 40)]


def test_similarity_fits_a_known_transform() -> None:
    rng = np.random.default_rng(4)
    src = rng.normal(50.0, 15.0, size=(5, 2))
    angle, scale = 0.7, 1.4
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = scale * src @ rot.T + np.array([12.0, -7.0])
    m = _similarity(src, dst)
    np.testing.assert_allclose(src @ m[:, :2].T + m[:, 2], dst, atol=1e-9)


def test_similarity_identity_for_equal_points() -> None:
    pts = np.array([[10.0, 20.0], [80.0, 25.0], [45.0, 90.0]])
    m = _similarity(pts, pts)
    np.testing.assert_allclose(m, np.hstack([np.eye(2), np.zeros((2, 1))]), atol=1e-9)


def test_local_full_roundtrip() -> None:
    m = np.array([[1.0, 0.1, 30.0], [-0.1, 1.0, -12.0]])
    origin = (37, 91)
    back = _full_from_local(_local_from_full(m, origin), origin)
    np.testing.assert_allclose(back, m, atol=1e-9)


def test_local_conversion_keeps_pure_translation() -> None:
    m = np.hstack([np.eye(2), np.array([[5.0], [-3.0]])])
    np.testing.assert_allclose(_local_from_full(m, (12, 34)), m, atol=1e-9)
    np.testing.assert_allclose(_full_from_local(m, (12, 34)), m, atol=1e-9)


def test_ecc_recovers_a_known_translation() -> None:
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 45.0, size=(140, 180)).clip(0, 255).astype(np.uint8)
    img[40:90, 60:120] = 30  # structure so ECC has gradients to lock onto
    m = np.array([[1, 0, 5], [0, 1, -4]], dtype=np.float32)
    shifted = cv2.warpAffine(img, m, (180, 140), borderMode=cv2.BORDER_REPLICATE).astype(
        np.uint8, copy=False
    )
    warp, cc, used = _ecc_affine(img, shifted, np.eye(2, 3, dtype=np.float64))
    assert used, f"ECC rejected (cc={cc})"
    assert cc > 0.9
    # template -> input: content moved by (5, -4)
    np.testing.assert_allclose(warp[:, 2], [5.0, -4.0], atol=0.5)


def test_ecc_falls_back_when_it_loses_the_face() -> None:
    flat = np.full((60, 80), 128, dtype=np.uint8)
    init = np.hstack([np.eye(2), np.array([[2.0], [1.0]])])
    _, cc, used = _ecc_affine(flat, flat, init)
    assert not used  # no texture: correlation collapses, init kept
    assert cc < 1.0


def test_ecc_det_bounds_decide_whether_a_scale_jump_is_accepted() -> None:
    # A 1.6x zoom is a legitimate cross-take face size change (det = 2.56),
    # but outside the stabilizer's within-take window (0.5..2.0).
    rng = np.random.default_rng(1)
    img = rng.normal(128.0, 45.0, size=(160, 200)).clip(0, 255).astype(np.uint8)
    img[50:110, 70:140] = 30
    scale, cx, cy = 1.6, 100.0, 80.0
    m = np.array(
        [[scale, 0.0, cx - scale * cx], [0.0, scale, cy - scale * cy]],
        dtype=np.float64,
    )
    zoomed = cv2.warpAffine(img, m.astype(np.float32), (200, 160), flags=cv2.INTER_CUBIC).astype(
        np.uint8, copy=False
    )
    warp, cc, used_default = _ecc_affine(img, zoomed, m)
    assert cc > 0.9
    assert not used_default, "default bounds must reject det ~2.56"
    np.testing.assert_allclose(warp, m)  # init kept on rejection
    _, _, used_wide = _ecc_affine(img, zoomed, m, (0.3, 4.0))
    assert used_wide, "wider bounds must accept the same registration"


def test_oval_mask_is_feathered_and_centred() -> None:
    cfg = np.array([[40.0, 40.0], [80.0, 40.0], [60.0, 60.0], [45.0, 80.0], [75.0, 80.0]])
    mask = _oval_mask((120, 160), cfg)
    assert mask.shape == (120, 160)
    assert mask.dtype == np.float32
    assert float(mask[60, 60]) > 0.9  # inside the face
    assert float(mask[5, 5]) < 0.05  # background corner
    # feather: the rim is between 0 and 1, not a hard edge
    rim = float(mask[60, int(60 + 1.4 * 40)])  # just outside the semi-axis
    assert 0.0 < rim < 1.0


def _square_clip(n: int = 60) -> tuple[list[U8], F64, F64]:
    """A bright square whose position jitters +-3 px around a smooth path."""
    frames: list[U8] = []
    centers = np.zeros((n, 2))
    for t in range(n):
        drift = np.array([0.15 * t, 4.0 * np.sin(0.2 * t)])
        jitter = np.array([3.0 if t % 2 else -3.0, -3.0 if t % 2 else 3.0])
        centers[t] = np.array([70.0, 60.0]) + drift + jitter
        frame = np.full((140, 180, 3), 40, dtype=np.uint8)
        c = centers[t]
        frame[int(c[1]) - 12 : int(c[1]) + 12, int(c[0]) - 12 : int(c[0]) + 12] = 220
        frames.append(frame)
    offsets = np.array([[-8.0, -8.0], [8.0, -8.0], [0.0, 0.0], [-8.0, 8.0], [8.0, 8.0]])
    lm = offsets[None, :, :] + centers[:, None, :]
    return frames, lm, centers


def test_stabilize_follows_the_lowpassed_trajectory() -> None:
    frames, lm, _centers = _square_clip()
    out = stabilize(frames, lm, sigma=2.0, window_sigma=1.5, gate_h=12.0)

    def centroid(frame: U8) -> np.ndarray:
        ys, xs = np.nonzero(frame[:, :, 0] > 130)
        return np.array([xs.mean(), ys.mean()])

    raw_cent = np.stack([centroid(f) for f in frames])
    out_cent = np.stack([centroid(f) for f in out])
    target = lowpass(lm, 2.0).mean(axis=1)  # the 5-point centroid path

    raw_err = np.linalg.norm(raw_cent - target, axis=1).mean()
    out_err = np.linalg.norm(out_cent - target, axis=1).mean()
    assert out_err < 0.5 * raw_err, f"raw err {raw_err:.2f} -> out err {out_err:.2f}"

    raw_step = np.linalg.norm(np.diff(raw_cent, axis=0), axis=1).mean()
    out_step = np.linalg.norm(np.diff(out_cent, axis=0), axis=1).mean()
    assert out_step < 0.5 * raw_step, f"raw step {raw_step:.2f} -> out step {out_step:.2f}"


def test_stabilize_leaves_the_background_alone() -> None:
    frames, lm, _centers = _square_clip()
    out = stabilize(frames, lm, sigma=2.0, window_sigma=1.5, gate_h=12.0)
    for frame in (out[10], out[30], out[50]):
        assert int(frame[3, 3, 0]) == 40  # background corner untouched
        assert int(frame[135, 175, 0]) == 40


def test_stabilize_leaves_flat_frames_flat() -> None:
    frames: list[U8] = [np.full((100, 140, 3), 128, dtype=np.uint8) for _ in range(30)]
    lm = _trajectory(30, jitter=0.0, seed=1)
    out = stabilize(frames, lm, sigma=2.0)
    assert len(out) == 30
    assert int(out[0].max()) == 128 and int(out[0].min()) == 128


def test_stabilize_rejects_length_mismatch() -> None:
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    lm = _trajectory(5, jitter=0.0)
    try:
        stabilize(frames, lm)
    except ValueError:
        return
    raise AssertionError("expected ValueError for frame/landmark mismatch")
