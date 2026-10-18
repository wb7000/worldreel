"""Progressive plane-completion fill (algorithm A-3, docs/ALGORITHMS.md).

Holes inside the observed footprint of a large plane (floor, wall, ground) are filled on that
plane with colours diffused in from the observed border. Nothing is extended beyond the
footprint. Output is `inferred` with confidence < 0.5 that falls with distance from evidence,
in two tiers: a quick one within `budget_s` and a refined one within `refine_budget_s`.
numpy/scipy only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_closing, distance_transform_edt, uniform_filter

F64 = npt.NDArray[np.float64]
MAX_CONFIDENCE = 0.45  # below 0.5: the player keeps marking fill (spec §10.3)
DECAY_CELLS = 4.0


@dataclass(frozen=True)
class Plane:
    normal: F64
    offset: float  # n . x = offset
    inliers: npt.NDArray[np.int64]


@dataclass
class FillTier:
    positions: F64 = field(default_factory=lambda: np.zeros((0, 3)))
    normals: F64 = field(default_factory=lambda: np.zeros((0, 3)))
    colors: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    confidence: npt.NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, np.float32))
    cell: float = 0.0
    seconds: float = 0.0
    complete: bool = True  # False when the budget ran out before every plane was done


STRUCTURAL_TOLERANCE_DEG = 20.0


def structural(plane: Plane, up: F64, tol_deg: float = STRUCTURAL_TOLERANCE_DEG) -> bool:
    """Level (floor, ceiling, ground, table top) or upright (wall). Slanted fits in rooms and
    courtyards are almost always planes through clutter (walking clip: a 37 deg plane, D-4.3)."""
    c = abs(float(plane.normal @ up)) / float(np.linalg.norm(up))
    return bool(c >= np.cos(np.radians(tol_deg)) or c <= np.sin(np.radians(tol_deg)))


def ransac_planes(
    points: F64,
    extent: float,
    *,
    max_planes: int = 6,
    band_frac: float = 0.01,
    min_frac: float = 0.03,
    iterations: int = 400,
    seed: int = 0,
    sample: int = 60000,
) -> list[Plane]:
    """Sequential RANSAC on a subsample; inliers are reported for the full point set."""
    rng = np.random.default_rng(seed)
    band = band_frac * extent
    remaining = np.arange(len(points))
    planes: list[Plane] = []
    min_count = min_frac * len(points)
    for _ in range(max_planes):
        if len(remaining) < max(3, min_count):
            break
        sub = (
            remaining if len(remaining) <= sample else rng.choice(remaining, sample, replace=False)
        )
        p = points[sub]
        best_n, best_d, best_score = None, 0.0, 0
        for _ in range(iterations):
            a, b, c = p[rng.choice(len(p), 3, replace=False)]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n = n / norm
            d = float(n @ a)
            score = int((np.abs(p @ n - d) < band).sum())
            if score > best_score:
                best_n, best_d, best_score = n, d, score
        if best_n is None:
            break
        inl = remaining[np.abs(points[remaining] @ best_n - best_d) < band]
        if len(inl) < min_count:
            break
        # Least-squares refinement on the inliers.
        mu = points[inl].mean(0)
        _, _, vt = np.linalg.svd(points[inl] - mu, full_matrices=False)
        n = vt[-1] / np.linalg.norm(vt[-1])
        d = float(n @ mu)
        inl = remaining[np.abs(points[remaining] @ n - d) < band]
        planes.append(Plane(n, d, inl))
        remaining = np.setdiff1d(remaining, inl, assume_unique=True)
    return planes


def _basis(n: F64) -> tuple[F64, F64]:
    a = np.cross(n, [0.0, 1.0, 0.0] if abs(n[1]) < 0.9 else [1.0, 0.0, 0.0])
    a = a / np.linalg.norm(a)
    return a, np.cross(n, a)


def _fill_plane(
    plane: Plane,
    points: F64,
    colors: npt.NDArray[np.uint8],
    cell: float,
    deadline: float,
    *,
    max_gap: float,
    hide_height: float,
    band: float,
    weights: F64 | None = None,
    smooth_cells: int = 5,
    colour_mode: str = "plane_mean",
) -> tuple[F64, F64, F64, F64, bool]:
    """(hole centres 3D, colours 0-255 float, distance in cells, normals, finished).

    Footprint = observed cells plus gaps they enclose, up to `max_gap` wide (morphological
    closing). Separate surfaces that happen to share a plane (several box tops) are not joined.
    Cells under observed geometry within `hide_height` of the plane are hidden, not missing,
    and are not filled."""
    a, b = _basis(plane.normal)
    pts = points[plane.inliers]
    uv = np.stack([pts @ a, pts @ b], axis=1)
    lo = uv.min(0)
    ij = np.floor((uv - lo) / cell).astype(np.int64)
    w, h = ij.max(0) + 1
    if w * h > 4_000_000 or len(uv) < 4:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), True
    flat = ij[:, 1] * w + ij[:, 0]
    count = np.bincount(flat, minlength=w * h).astype(np.float64)
    occupied = count > 0
    # Border colours: opacity-weighted and averaged over a small window of cells. Single splat
    # colours are noisy (they blend into pixels), and diffusion would spread that noise (D-4.3).
    wt = np.ones(len(flat)) if weights is None else weights[plane.inliers]
    wsum = uniform_filter(np.bincount(flat, wt, w * h).reshape(h, w), smooth_cells, mode="constant")
    col = np.stack(
        [
            uniform_filter(
                np.bincount(flat, wt * colors[plane.inliers, k], w * h).reshape(h, w),
                smooth_cells,
                mode="constant",
            ).ravel()
            for k in range(3)
        ],
        axis=1,
    )
    col[occupied] /= np.maximum(wsum.ravel()[occupied, None], 1e-12)
    gi, gj = np.meshgrid(np.arange(w), np.arange(h))
    centres = np.stack([gi.ravel(), gj.ravel()], axis=1) * cell + lo + cell / 2
    known2 = occupied.reshape(h, w)
    radius = max(1, round(max_gap / (2 * cell)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = (xx * xx + yy * yy) <= radius * radius
    padded = np.pad(known2, radius + 1)
    closed = binary_closing(padded, structure=disk)[
        radius + 1 : -radius - 1, radius + 1 : -radius - 1
    ]
    inside = (closed | known2).ravel()
    # Hidden cells: something observed stands on / in front of the plane there.
    side = points @ plane.normal - plane.offset
    near = (np.abs(side) > band) & (np.abs(side) <= hide_height)
    if near.any():
        q = points[near]
        qij = np.floor((np.stack([q @ a, q @ b], axis=1) - lo) / cell).astype(np.int64)
        ok = (qij[:, 0] >= 0) & (qij[:, 0] < w) & (qij[:, 1] >= 0) & (qij[:, 1] < h)
        hidden = np.zeros(w * h, bool)
        hidden[qij[ok, 1] * w + qij[ok, 0]] = True
        inside &= ~(hidden & ~occupied)
    holes = inside & ~occupied
    if not holes.any():
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), True
    grid = col.reshape(h, w, 3).copy()
    known = occupied.reshape(h, w)
    region = inside.reshape(h, w) | known
    grid[~known] = col[occupied].mean(0)
    finished = True
    # Jacobi Laplace diffusion inside the footprint; known cells stay fixed.
    for it in range(4 * max(w, h)):
        if time.perf_counter() > deadline:
            finished = False
            break
        acc = np.zeros_like(grid)
        num = np.zeros((h, w, 1))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.roll(grid, (dy, dx), axis=(0, 1))
            ok = np.roll(region, (dy, dx), axis=(0, 1))[..., None]
            acc += shifted * ok
            num += ok
        new = np.where(num > 0, acc / np.maximum(num, 1), grid)
        update = region & ~known
        delta = float(np.abs(new[update] - grid[update]).max()) if update.any() else 0.0
        grid[update] = new[update]
        if it > 10 and delta < 0.5:
            break
    if colour_mode == "plane_mean":
        # The plane's own opacity-weighted mean colour (robust to noisy border splats).
        mean = (wt[:, None] * colors[plane.inliers]).sum(0) / max(float(wt.sum()), 1e-12)
        grid.reshape(-1, 3)[holes] = mean
    dist = distance_transform_edt(~known).ravel()[holes]
    c = centres[holes]
    pos3 = c[:, :1] * a + c[:, 1:] * b + plane.offset * plane.normal
    normals = np.tile(plane.normal, (len(pos3), 1))
    return pos3, grid.reshape(-1, 3)[holes], dist, normals, finished


def fill_tier(
    planes: list[Plane],
    points: F64,
    colors: npt.NDArray[np.uint8],
    cell: float,
    budget_s: float,
    *,
    extent: float,
    max_gap_frac: float = 0.15,
    hide_frac: float = 0.1,
    band_frac: float = 0.01,
    weights: F64 | None = None,
    colour_mode: str = "plane_mean",
) -> FillTier:
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    parts: list[tuple[F64, F64, F64, F64]] = []
    complete = True
    for plane in planes:
        if time.perf_counter() > deadline:
            complete = False
            break
        pos, col, dist, nrm, done = _fill_plane(
            plane,
            points,
            colors,
            cell,
            deadline,
            max_gap=max_gap_frac * extent,
            hide_height=hide_frac * extent,
            band=band_frac * extent,
            weights=weights,
            colour_mode=colour_mode,
        )
        complete &= done
        if len(pos):
            parts.append((pos, col, dist, nrm))
    tier = FillTier(cell=cell, complete=complete)
    if parts:
        tier.positions = np.concatenate([p[0] for p in parts])
        tier.colors = (
            np.clip(np.concatenate([p[1] for p in parts]), 0, 255).round().astype(np.uint8)
        )
        dist = np.concatenate([p[2] for p in parts])
        tier.confidence = (MAX_CONFIDENCE * np.exp(-dist / DECAY_CELLS)).astype(np.float32)
        tier.normals = np.concatenate([p[3] for p in parts])
    tier.seconds = time.perf_counter() - t0
    return tier


def progressive_fill(
    points: F64,
    colors: npt.NDArray[np.uint8],
    extent: float,
    *,
    budget_s: float = 5.0,
    refine_budget_s: float = 60.0,
    seed: int = 0,
    weights: F64 | None = None,
    colour_mode: str = "plane_mean",
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[list[Plane], FillTier, FillTier]:
    """Tier 1 (coarse cells, within `budget_s` including plane detection) and tier 2 (fine)."""
    t0 = time.perf_counter()
    planes = [p for p in ransac_planes(points, extent, seed=seed) if structural(p, np.asarray(up))]
    spent = time.perf_counter() - t0
    tier1 = fill_tier(
        planes,
        points,
        colors,
        extent / 60,
        max(0.0, budget_s - spent),
        extent=extent,
        weights=weights,
        colour_mode=colour_mode,
    )
    tier1.seconds += spent
    tier2 = fill_tier(
        planes,
        points,
        colors,
        extent / 120,
        refine_budget_s,
        extent=extent,
        weights=weights,
        colour_mode=colour_mode,
    )
    return planes, tier1, tier2


def drop_seen_through(tier: FillTier, views: list[Any], observed: F64, min_views: int = 2) -> int:
    """Remove fill cells that cameras saw through: in >= `min_views` training views the nearest
    observed surface in that image cell lies clearly behind the fill cell, so the cell is open
    space (a doorway, the gap beside a TV), not a hole. Returns how many were removed."""
    from stages.package.confidence import stereo_evidence

    if not len(tier.positions) or not views:
        return 0
    _, in_front = stereo_evidence(tier.positions, views, observed)
    keep = in_front < min_views
    removed = int((~keep).sum())
    tier.positions = tier.positions[keep]
    tier.normals = tier.normals[keep]
    tier.colors = tier.colors[keep]
    tier.confidence = tier.confidence[keep]
    return removed
