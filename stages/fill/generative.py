"""Generative texture for filled planes (algorithm A-5, docs/ALGORITHMS.md).

Each plane that A-3 filled gets an orthographic texture rasterised from the observed Gaussians;
the cells holding fill elements are inpainted by LaMa (Apache-2.0, ONNX port `Carve/LaMa-ONNX`
lama_fp32.onnx, see docs/MODELS.md) and the fill elements take their colours from it. The
result is labelled `generated`: the model invents plausible texture, it does not know what was
there. Cells outside the plane's footprint are pre-filled with OpenCV's classical inpainting so
the model sees surface-like context, not black.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from stages.fill.planes import FillTier, Plane, _basis

F64 = npt.NDArray[np.float64]
U8 = npt.NDArray[np.uint8]
LAMA_MODEL = Path("data/models/lama_fp32.onnx")
LAMA_SIZE = 512
GENERATED_CONFIDENCE = 0.2

Inpainter = Callable[[U8, npt.NDArray[np.bool_]], U8]


def lama_inpainter(model: Path = LAMA_MODEL) -> Inpainter:
    """LaMa via onnxruntime. Input image [0,1], mask 1 = hole, output [0,255] (measured, D-4.5).
    Any image size: resampled to 512x512 and back; only hole pixels are replaced."""
    if not model.is_file():
        raise FileNotFoundError(f"LaMa model missing: {model} (see docs/MODELS.md)")
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    def run(image: U8, hole: npt.NDArray[np.bool_]) -> U8:
        h, w = image.shape[:2]
        img = cv2.resize(image, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_CUBIC)
        m = cv2.resize(
            hole.astype(np.uint8), (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST
        )
        x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        mk = m[None, None].astype(np.float32)
        out = session.run(None, {"image": x * (1 - mk), "mask": mk})[0][0].transpose(1, 2, 0)
        out8 = np.clip(out, 0, 255).astype(np.uint8)
        back = cv2.resize(out8, (w, h), interpolation=cv2.INTER_CUBIC)
        result = image.copy()
        result[hole] = back[hole]
        return result

    return run


def plane_texture(
    plane: Plane,
    points: F64,
    colors: U8,
    weights: F64,
    targets: F64,
    cell: float,
) -> tuple[U8, npt.NDArray[np.bool_], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """(texture H x W x 3 from observed inliers, mask of target cells, target rows, target cols).
    Unknown cells that are not targets are pre-filled classically."""
    a, b = _basis(plane.normal)
    obs = points[plane.inliers]
    uv_obs = np.stack([obs @ a, obs @ b], axis=1)
    uv_t = np.stack([targets @ a, targets @ b], axis=1)
    lo = np.minimum(uv_obs.min(0), uv_t.min(0)) if len(uv_t) else uv_obs.min(0)
    ij_o = np.floor((uv_obs - lo) / cell).astype(np.int64)
    ij_t = np.floor((uv_t - lo) / cell).astype(np.int64)
    w = int(max(ij_o[:, 0].max(), ij_t[:, 0].max() if len(ij_t) else 0)) + 1
    h = int(max(ij_o[:, 1].max(), ij_t[:, 1].max() if len(ij_t) else 0)) + 1
    flat = ij_o[:, 1] * w + ij_o[:, 0]
    wt = weights[plane.inliers]
    wsum = np.bincount(flat, wt, w * h)
    tex = np.stack([np.bincount(flat, wt * colors[plane.inliers, k], w * h) for k in range(3)], 1)
    known = wsum > 0
    tex[known] /= wsum[known, None]
    tex8 = np.clip(tex, 0, 255).astype(np.uint8).reshape(h, w, 3)
    target = np.zeros((h, w), bool)
    target[ij_t[:, 1], ij_t[:, 0]] = True
    target &= ~known.reshape(h, w)
    other_unknown = ~known.reshape(h, w) & ~target
    if other_unknown.any():
        tex8 = cv2.inpaint(tex8, other_unknown.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return tex8, target, ij_t[:, 1], ij_t[:, 0]


def generative_colours(
    planes: list[Plane],
    points: F64,
    colors: U8,
    weights: F64,
    tier: FillTier,
    inpainter: Inpainter,
) -> tuple[U8, float]:
    """Colours for every element of `tier` from its plane's inpainted texture; seconds taken."""
    t0 = time.perf_counter()
    out = tier.colors.copy()
    for k, plane in enumerate(planes):
        sel = np.nonzero(tier.plane_index == k)[0]
        if not len(sel):
            continue
        tex, target, rows, cols = plane_texture(
            plane, points, colors, weights, tier.positions[sel], tier.cell
        )
        if target.any():
            tex = inpainter(tex, target)
        out[sel] = tex[rows, cols]
    return out, time.perf_counter() - t0
