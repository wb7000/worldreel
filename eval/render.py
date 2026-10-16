"""Reference point-cloud renderer for evaluation: projects coloured points into a pinhole camera
with a z-buffer, drawing each point as a small square. Deliberately simple and deterministic; it
measures the reconstruction, not the WorldReel player's renderer."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eval.geometry import quat_to_matrix


def render_points(
    points: npt.NDArray[np.floating],
    colors: npt.NDArray[np.uint8],
    k: dict[str, float],
    position: list[float],
    rotation: list[float],
    *,
    radius_px: int = 1,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """Returns (image (H, W, 3) uint8, depth (H, W) float32 with 0 = empty)."""
    w, h = int(k["width"]), int(k["height"])
    r = quat_to_matrix(rotation)
    cam = (np.asarray(points, np.float64) - np.asarray(position)) @ r  # world → camera
    z = -cam[:, 2]
    front = z > 1e-3
    cam, z, col = cam[front], z[front], colors[front]
    u = np.round(cam[:, 0] / z * k["fx"] + k["cx"]).astype(np.int64)
    v = np.round(-cam[:, 1] / z * k["fy"] + k["cy"]).astype(np.int64)
    offsets = [
        (du, dv)
        for du in range(-radius_px, radius_px + 1)
        for dv in range(-radius_px, radius_px + 1)
    ]
    uu = np.concatenate([u + du for du, _ in offsets])
    vv = np.concatenate([v + dv for _, dv in offsets])
    zz = np.tile(z, len(offsets))
    cc = np.tile(col, (len(offsets), 1))
    ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    pix, zz, cc = (vv * w + uu)[ok], zz[ok], cc[ok]
    # Nearest point per pixel: sort by (pixel, depth) and keep the first of each pixel.
    order = np.lexsort((zz, pix))
    pix, zz, cc = pix[order], zz[order], cc[order]
    first = np.ones(len(pix), bool)
    first[1:] = pix[1:] != pix[:-1]
    image = np.zeros((h * w, 3), np.uint8)
    depth = np.zeros(h * w, np.float32)
    image[pix[first]] = cc[first]
    depth[pix[first]] = zz[first]
    return image.reshape(h, w, 3), depth.reshape(h, w)
