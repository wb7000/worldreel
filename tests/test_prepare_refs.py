"""Tests for the VACE reference normaliser (tools/prepare_refs.py).

Only the pure geometry and the resample are exercised here: the ONNX face
models are Kaggle-mounted and must not be needed to prove the window maths
(D-9.29).
"""

import cv2
import numpy as np

from tools.prepare_refs import (
    OUT_SIZE,
    crop_resize,
    normalize_reference,
    window_for_face,
)


def test_window_has_output_aspect_and_face_fractions() -> None:
    # A centred, small enough face that the window fits the 512x512 source
    # without being clamped, so the fractions below are the pure geometry.
    win = window_for_face((216.0, 150.0, 80.0, 100.0), (512, 512))
    x0, y0, win_w, win_h = win
    out_w, out_h = OUT_SIZE
    # aspect matches the output exactly, so the reference reaches the model
    # full-bleed and `preprocess_conditions` adds no white bars.
    assert abs(win_w / win_h - out_w / out_h) < 1e-9
    # the face fills the configured share of the window height
    assert abs(100.0 / win_h - 0.42) < 1e-9
    # horizontally centred on the face
    assert abs((x0 + win_w / 2.0) - (216.0 + 80.0 / 2.0)) < 1e-9
    # face centre lands at FACE_CY_FRAC of the window height
    assert abs(((150.0 + 100.0 / 2.0) - y0) / win_h - 0.38) < 1e-9


def test_window_is_clamped_inside_a_source_it_fits() -> None:
    # A face low in the frame: the unclamped window would hang off the bottom
    # and border replication would smear the sweater into streaks.
    img_shape = (512, 512)
    x0, y0, win_w, win_h = window_for_face((200.0, 430.0, 80.0, 90.0), img_shape)
    assert y0 >= 0.0
    assert y0 + win_h <= img_shape[0]
    assert x0 >= 0.0
    assert x0 + win_w <= img_shape[1]


def test_window_is_padded_only_when_it_cannot_fit() -> None:
    # A tiny source: the 832x480-aspect window is capped at 1.6x the source
    # width, which still leaves it wider than the source, so the caller has to
    # replicate the border instead of clamping.
    x0, _y0, win_w, _win_h = window_for_face((10.0, 10.0, 40.0, 50.0), (64, 64))
    assert win_w > 64
    assert x0 < 0.0


def test_crop_resize_returns_the_requested_size_and_no_white_bars() -> None:
    # 512x512 source, window wider than the source: the pad must be a
    # replication of the edge, never the white canvas VACE would add itself.
    img = np.full((512, 512, 3), 90, dtype=np.uint8)
    img[:, :256] = 40
    out = crop_resize(img, (-100.0, 0.0, 712.0, 480.0), sharpen=0.0)
    assert out.shape == (480, 832, 3)
    assert out.dtype == np.uint8
    # no pixel is white (a letterbox bar would be 255) and both edges keep the
    # source's own values rather than a synthetic fill.
    assert int(out.max()) < 255
    assert abs(int(out[0, 0, 0]) - 40) <= 2
    assert abs(int(out[0, -1, 0]) - 90) <= 2


def test_sharpening_increases_local_contrast() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(480, 832, 3), dtype=np.uint8)
    flat = crop_resize(img, (0.0, 0.0, 832.0, 480.0), sharpen=0.0)
    crisp = crop_resize(img, (0.0, 0.0, 832.0, 480.0), sharpen=0.6)

    def lap(a: np.ndarray) -> float:
        g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    assert lap(crisp) > lap(flat)


def test_normalize_reference_keeps_the_output_geometry() -> None:
    # 512x512 portrait, face bbox like YuNet's, one call: the result must be
    # exactly the output size so VACE's letterbox padding is a no-op.
    img = np.full((512, 512, 3), 120, dtype=np.uint8)
    img[180:340, 170:350] = 200
    out = normalize_reference(img, (170.0, 180.0, 180.0, 160.0))
    assert out.shape == (480, 832, 3)
