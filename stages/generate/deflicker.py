"""Remove colour and light changes from a generated clip with a fixed camera (D-8.13d).

Video diffusion shifts colour between frames and between chained pieces. Fine detail is kept. Two
steps: slow drift is measured on the wall beside the subject and corrected by a smooth surface, as
a whole-frame average would leave a glow where the head moved; then fast flicker is pulled towards
the average of nearby frames.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
SMALL = 8  # the light field is smooth, so work at 1/8 size
WALL_COLS = 0.25  # share of the width on each side, and of the height from the top, seen as wall
WALL_ROWS = 0.6


def _light(frame: U8, sigma: float) -> F32:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // SMALL, h // SMALL), interpolation=cv2.INTER_AREA)
    out: F32 = np.asarray(cv2.GaussianBlur(small.astype(np.float32) + 1.0, (0, 0), sigma / SMALL))
    return out


def _basis(x: F64, y: F64) -> F64:
    out: F64 = np.stack([np.ones_like(x), x, y, x * x, x * y], axis=-1)
    return out


def steady_wall(frames: Sequence[U8]) -> list[U8]:
    """Pull each frame's light on the wall to the wall's light over the whole clip."""
    h, w = frames[0].shape[:2]
    small = np.stack(
        [
            cv2.resize(f, (w // SMALL, h // SMALL), interpolation=cv2.INTER_AREA).astype(np.float64)
            for f in frames
        ]
    )
    sh, sw = small.shape[1:3]
    yy, xx = np.mgrid[:sh, :sw]
    side = np.minimum(xx, sw - 1 - xx) < WALL_COLS * sw
    wall = side & (yy < WALL_ROWS * sh)
    a = _basis(xx[wall] / sw - 0.5, yy[wall] / sh - 0.5)
    target = np.log(small.mean(axis=0)[wall] + 1.0)
    fy, fx = np.mgrid[:h, :w]
    full = _basis(fx / w - 0.5, fy / h - 0.5)
    out: list[U8] = []
    for frame, s in zip(frames, small, strict=True):
        coef, *_ = np.linalg.lstsq(a, target - np.log(s[wall] + 1.0), rcond=None)
        gain = np.exp(full @ coef)
        out.append(np.clip(np.round(frame * gain), 0, 255).astype(np.uint8))
    return out


def smooth_flicker(frames: Sequence[U8], window: int = 16, sigma: float = 48.0) -> list[U8]:
    """Pull each frame's blurred image towards its average over `window` frames either side.
    `sigma`: blur in pixels, so that a face's features are not treated as light."""
    lights = np.stack([_light(f, sigma) for f in frames])
    # running sum, so a long clip stays cheap
    total = np.concatenate([np.zeros_like(lights[:1]), np.cumsum(lights, axis=0)])
    h, w = frames[0].shape[:2]
    out: list[U8] = []
    for t, frame in enumerate(frames):
        a, b = max(0, t - window), min(len(frames), t + window + 1)
        target = (total[b] - total[a]) / (b - a)
        gain = cv2.resize(target / lights[t], (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(np.clip(np.round(frame * gain), 0, 255).astype(np.uint8))
    return out


def deflicker(frames: Sequence[U8]) -> list[U8]:
    return smooth_flicker(steady_wall(frames))
