"""Image-quality metrics for novel views: PSNR and SSIM (Wang et al. 2004, 11x11 Gaussian
window, sigma = 1.5, K1 = 0.01, K2 = 0.03). LPIPS needs a pretrained network and is not implemented
yet (see DECISIONS D-3a.x)."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]


def _to_float(img: npt.NDArray[np.generic]) -> npt.NDArray[np.float64]:
    a = img.astype(np.float64)
    return a / 255.0 if img.dtype == np.uint8 else a


def psnr(
    a: npt.NDArray[np.generic],
    b: npt.NDArray[np.generic],
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    x, y = _to_float(a), _to_float(b)
    diff = (x - y) ** 2
    if mask is not None:
        diff = diff[mask]
    mse = float(diff.mean()) if diff.size else float("nan")
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim(
    a: npt.NDArray[np.generic],
    b: npt.NDArray[np.generic],
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """Mean SSIM over all channels (and over `mask` pixels if given)."""
    x, y = _to_float(a), _to_float(b)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    c1, c2 = 0.01**2, 0.03**2

    def blur(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = np.asarray(
            cv2.GaussianBlur(z, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT), np.float64
        )
        return out

    maps = []
    for ch in range(x.shape[2]):
        xc, yc = x[..., ch], y[..., ch]
        mx, my = blur(xc), blur(yc)
        sxx = blur(xc * xc) - mx * mx
        syy = blur(yc * yc) - my * my
        sxy = blur(xc * yc) - mx * my
        maps.append(
            ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        )
    m = np.mean(maps, axis=0)
    return float(m[mask].mean() if mask is not None else m.mean())
