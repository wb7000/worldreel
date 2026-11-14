"""Canonical 5-point face alignment shared by every SFace consumer.

Why this module exists
----------------------
`cv2.FaceRecognizerSF.alignCrop` is unreliable in OpenCV 5.0.0. On the same
input it returns an all-black 112x112 image on most runs (20/20 black in one
measurement, a misaligned crop in another), and its embedding norm collapses
from 14.2 to 3.5 with cosine 0.03 against the template-aligned embedding. A
black crop makes `FaceRecognizerSF.feature()` return the same degenerate vector
for every face, so all pairwise cosine similarities collapse to 1.0000 and
every identity metric becomes a constant no-op.

Because the identity metric silently stops discriminating, we never call
`alignCrop`. We implement the alignment ourselves from the canonical ArcFace /
SFace 5-point template, which is the same geometry SFace was trained with and
works identically on OpenCV 4.x and 5.x.

Landmark order
--------------
YuNet emits five landmarks in the order:
    right eye, left eye, nose tip, right mouth corner, left mouth corner
For a frontal face the subject's right eye is the *smaller-x* one in image
coordinates, and the canonical template lists the eye slots in the same
left-to-right order. So the mapping is the identity permutation -- verified
empirically in `tools/face_align_probe.py`:

    YuNet order  : same character 0.6890, different character 0.2472 (+0.4418)
    swapped order: same character 0.5365, different character 0.4815 (+0.0549)
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

# Canonical ArcFace/SFace 5-point template in a 112x112 crop.
# eye-left, eye-right, nose, mouth-left, mouth-right (image x/y order).
SFACE_TEMPLATE: NDArray[np.float32] = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

ALIGN_SIZE = 112


def to_uint8(arr: object) -> NDArray[np.uint8]:
    """Convert any array-like to uint8 with a precise static type.

    OpenCV's bundled stubs type every array return as
    `ndarray[Any, dtype[integer | floating]]`. numpy's `asarray`/`astype` cannot
    narrow that union, so the conversion is done here once and the static type
    is asserted with an explicit cast.
    """
    return cast(NDArray[np.uint8], np.asarray(arr, dtype=np.uint8))


def to_float32(arr: object) -> NDArray[np.float32]:
    """Convert any array-like to float32 with a precise static type."""
    return cast(NDArray[np.float32], np.asarray(arr, dtype=np.float32))


def similarity_transform(
    src: NDArray[np.float32], dst: NDArray[np.float32]
) -> NDArray[np.float32] | None:
    """Least-squares similarity transform (rotation + uniform scale + translation).

    Maps `src` onto `dst`. Returns a 2x3 matrix, or None if the fit fails.
    """
    if src.shape != dst.shape or src.shape[0] < 2:
        return None
    m, _ = cv2.estimateAffinePartial2D(
        to_float32(src).reshape(-1, 1, 2),
        to_float32(dst).reshape(-1, 1, 2),
        method=cv2.LMEDS,
    )
    if m is None:
        return None
    return to_float32(m)


def align_face(
    img_bgr: NDArray[np.uint8], landmarks: NDArray[np.float32]
) -> NDArray[np.uint8] | None:
    """Warp the face described by `landmarks` into the canonical 112x112 template.

    `landmarks` must be the five YuNet/SFace landmarks in their native order.
    Returns None when the transform cannot be estimated.
    """
    lm = to_float32(landmarks).reshape(-1, 2)
    if lm.shape[0] != 5:
        return None
    m = similarity_transform(lm, SFACE_TEMPLATE)
    if m is None:
        return None
    aligned = cv2.warpAffine(
        to_uint8(img_bgr),
        m,
        (ALIGN_SIZE, ALIGN_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return to_uint8(aligned)


def face_oval_mask(
    center: tuple[float, float] = (56.0, 54.0),
    axes: tuple[float, float] = (42.0, 51.0),
    size: int = ALIGN_SIZE,
) -> NDArray[np.uint8]:
    """Elliptical face-oval mask in template space (255 = inside the face).

    Geometry chosen by grid search over identity similarity on a test frame
    (`tools/face_align_probe.py`): the tall ellipse scores 0.8401, a clipped
    variant that stops above the hairline scores 0.8054, and a shorter ellipse
    0.7579. SFace reads the forehead as part of the identity, so keeping the
    full forehead in the swapped region matters more than avoiding the hairline.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (round(center[0]), round(center[1])),
        (round(axes[0]), round(axes[1])),
        0,
        0,
        360,
        255,
        -1,
    )
    return mask


def tone_match(
    source: NDArray[np.uint8],
    source_mask: NDArray[np.uint8],
    target: NDArray[np.uint8],
    target_mask: NDArray[np.uint8],
    strength: float = 1.0,
) -> NDArray[np.uint8]:
    """Per-channel Reinhard tone transfer from `target` onto `source`.

    Both masks select the same anatomical region in their own images, so the
    statistics are comparable. `strength` in [0, 1] scales how far the
    correction is applied; 0 leaves the source untouched.
    """
    if strength <= 0.0:
        return source
    src_sel = source[source_mask > 200]
    dst_sel = target[target_mask > 200]
    if src_sel.size == 0 or dst_sel.size == 0:
        return source
    out = source.astype(np.float32)
    for ch in range(3):
        s = src_sel[:, ch].astype(np.float32)
        d = dst_sel[:, ch].astype(np.float32)
        s_mean, s_std = float(s.mean()), float(s.std())
        d_mean, d_std = float(d.mean()), float(d.std())
        scale = 1.0 if s_std < 1e-3 else d_std / s_std
        corrected = (out[:, :, ch] - s_mean) * scale + d_mean
        out[:, :, ch] = out[:, :, ch] * (1.0 - strength) + corrected * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def feather(mask: NDArray[np.uint8], blur: int = 9) -> NDArray[np.float32]:
    """Gaussian-feather a binary mask into a float alpha map in [0, 1]."""
    if blur <= 1:
        return to_float32(mask.astype(np.float32) / 255.0)
    blurred = to_float32(cv2.GaussianBlur(mask.astype(np.float32), (blur | 1, blur | 1), 0))
    return to_float32(np.clip(blurred / 255.0, 0.0, 1.0))
