"""View-support confidence for Gaussian splats (algorithm A-1, docs/ALGORITHMS.md).

A Gaussian is trustworthy when several training views saw its centre un-occluded and from
different directions. Per Gaussian centre: `n` = views where it projects inside the image, is
not hidden behind nearer opaque Gaussians (coarse z-buffer) and is not on a masked person;
`spread` = angular spread of those viewing directions. Both map to a provenance class and a
confidence (spec §10): observed / inferred / guessed. numpy only: runs on the laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from formats.wreel import PROVENANCE_CLASSES
from stages.reconstruct.planesweep import cv_pose

F64 = npt.NDArray[np.float64]
_FLIP = np.diag([1.0, -1.0, -1.0])
OBSERVED = PROVENANCE_CLASSES.index("observed")
INFERRED = PROVENANCE_CLASSES.index("inferred")
GUESSED = PROVENANCE_CLASSES.index("guessed")


@dataclass(frozen=True)
class SupportConfig:
    n_full: int = 8  # views for full count score
    spread_full_deg: float = 10.0  # angular spread for full parallax score
    observed_score: float = 0.6
    cell_px: int = 4  # z-buffer cell size
    depth_tol: float = 0.05  # relative depth tolerance of the occlusion test
    occluder_opacity: float = 0.5
    near: float = 1e-3
    use_spread: bool = True  # ablation switch


@dataclass(frozen=True)
class View:
    """One training view: OpenCV world-to-camera (r, t) in OpenCV world axes, intrinsics, size,
    optional boolean mask (H x W) of pixels that must not count (moving people)."""

    r: F64
    t: F64
    k: F64
    width: int
    height: int
    mask: npt.NDArray[np.bool_] | None = None


def views_from_camera_json(
    cam: dict[str, Any], masks: npt.NDArray[np.uint8] | None = None
) -> list[View]:
    """Registered poses only (unregistered ones are interpolated guesses, not evidence)."""
    k = cam["intrinsics"]
    kmat = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], np.float64)
    w, h = int(k["width"]), int(k["height"])
    views = []
    for i, pose in enumerate(cam["poses"]):
        if not pose.get("registered", True):
            continue
        r, t = cv_pose(pose["position"], pose["rotation"])
        mask = None
        if masks is not None and i < len(masks) and masks[i].shape == (h, w) and masks[i].any():
            mask = masks[i] > 0
        views.append(View(r, t, kmat, w, h, mask))
    return views


def view_support(
    positions: F64, opacity: npt.NDArray[np.floating[Any]], views: list[View], cfg: SupportConfig
) -> tuple[npt.NDArray[np.int32], F64]:
    """Returns (n views supporting each centre, angular spread in degrees)."""
    x_cv = positions.astype(np.float64) @ _FLIP.T  # WorldReel world -> OpenCV world axes
    g = len(x_cv)
    n = np.zeros(g, np.int32)
    dir_sum = np.zeros((g, 3))
    seen: list[tuple[npt.NDArray[np.int64], F64]] = []
    occluder = opacity >= cfg.occluder_opacity
    for v in views:
        pc = x_cv @ v.r.T + v.t
        z = pc[:, 2]
        front = z > cfg.near
        u = np.full(g, -1.0)
        w_ = np.full(g, -1.0)
        u[front] = v.k[0, 0] * pc[front, 0] / z[front] + v.k[0, 2]
        w_[front] = v.k[1, 1] * pc[front, 1] / z[front] + v.k[1, 2]
        inside = front & (u >= 0) & (u < v.width) & (w_ >= 0) & (w_ < v.height)
        idx = np.nonzero(inside)[0]
        if not len(idx):
            continue
        cu = (u[idx] // cfg.cell_px).astype(np.int64)
        cv_ = (w_[idx] // cfg.cell_px).astype(np.int64)
        cols = -(-v.width // cfg.cell_px)
        cell = cv_ * cols + cu
        zbuf = np.full((-(-v.height // cfg.cell_px)) * cols, np.inf)
        occ = occluder[idx]
        np.minimum.at(zbuf, cell[occ], z[idx][occ])
        visible = z[idx] <= zbuf[cell] * (1 + cfg.depth_tol)
        if v.mask is not None:
            px = v.mask[w_[idx].astype(np.int64), u[idx].astype(np.int64)]
            visible &= ~px
        vis = idx[visible]
        centre = -v.r.T @ v.t
        d = x_cv[vis] - centre
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        n[vis] += 1
        dir_sum[vis] += d
        seen.append((vis, d))
    mean = dir_sum / np.maximum(np.linalg.norm(dir_sum, axis=1, keepdims=True), 1e-12)
    half = np.zeros(g)
    for vis, d in seen:
        ang = np.degrees(np.arccos(np.clip(np.sum(d * mean[vis], axis=1), -1.0, 1.0)))
        half[vis] = np.maximum(half[vis], ang)
    return n, 2.0 * half


def _project_cells(
    p: F64, v: View, cell_px: int
) -> tuple[npt.NDArray[np.int64], F64, npt.NDArray[np.int64]]:
    """(indices inside the image, their depths, their cell ids) for points in OpenCV axes."""
    cols = -(-v.width // cell_px)
    pc = p @ v.r.T + v.t
    z = pc[:, 2]
    ok = z > 1e-3
    safe = np.where(ok, z, 1.0)
    u = np.where(ok, v.k[0, 0] * pc[:, 0] / safe + v.k[0, 2], -1.0)
    w = np.where(ok, v.k[1, 1] * pc[:, 1] / safe + v.k[1, 2], -1.0)
    ok &= (u >= 0) & (u < v.width) & (w >= 0) & (w < v.height)
    idx = np.nonzero(ok)[0]
    cell = (w[idx] // cell_px).astype(np.int64) * cols + (u[idx] // cell_px).astype(np.int64)
    return idx, z[idx], cell


def stereo_evidence(
    positions: F64, views: list[View], stereo: F64, *, cell_px: int = 8, margin: float = 0.1
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Per Gaussian: (views where stereo covers its image cell, views where it lies clearly in
    front of the nearest stereo surface in that cell, i.e. in space stereo saw as empty)."""
    x_cv = positions.astype(np.float64) @ _FLIP.T
    s_cv = stereo.astype(np.float64) @ _FLIP.T
    covered = np.zeros(len(x_cv), np.int32)
    in_front = np.zeros(len(x_cv), np.int32)
    for v in views:
        s_idx, s_z, s_cell = _project_cells(s_cv, v, cell_px)
        if not len(s_idx):
            continue
        surface = np.full((-(-v.height // cell_px)) * (-(-v.width // cell_px)), np.inf)
        np.minimum.at(surface, s_cell, s_z)
        g_idx, g_z, g_cell = _project_cells(x_cv, v, cell_px)
        depth = surface[g_cell]
        has = np.isfinite(depth)
        covered[g_idx[has]] += 1
        in_front[g_idx[has & (g_z < (1 - margin) * depth)]] += 1
    return covered, in_front


def free_space_floaters(
    covered: npt.NDArray[np.int32], in_front: npt.NDArray[np.int32], min_views: int = 3
) -> npt.NDArray[np.bool_]:
    """In front of a stereo surface in at least half of the covered views (>= `min_views`)."""
    return np.asarray((covered >= min_views) & (2 * in_front >= covered))


def far_from_points(positions: F64, support: F64, radius: float) -> npt.NDArray[np.bool_]:
    from scipy.spatial import cKDTree

    if not len(support):
        return np.zeros(len(positions), bool)
    dist, _ = cKDTree(support).query(positions, workers=-1)
    return np.asarray(dist > radius)


def classify(
    n: npt.NDArray[np.int32],
    spread: F64,
    cfg: SupportConfig,
    *,
    far_from_mvs: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """`far_from_mvs`: Gaussians with no independent multi-view-stereo point nearby; their
    score is halved (a floater in front of the cameras is seen by every view, but the
    classical reconstruction found no surface there)."""
    score = np.minimum(1.0, n / cfg.n_full)
    if cfg.use_spread:
        score = score * np.minimum(1.0, spread / cfg.spread_full_deg)
    if far_from_mvs is not None:
        score = np.where(far_from_mvs, 0.5 * score, score)
    cls = np.where(score >= cfg.observed_score, OBSERVED, np.where(n >= 1, INFERRED, GUESSED))
    conf = np.where(
        cls == OBSERVED,
        0.5 + 0.5 * score,
        np.where(cls == INFERRED, np.minimum(0.2 + 0.5 * score, 0.49), 0.1),
    )
    return cls.astype(np.uint8), conf.astype(np.float32)


MVS_RADIUS_FRACTION = 0.02  # x scene extent (D-4.1)
MIN_COVERED_VIEWS = 3


def relabel_gaussians(
    positions: F64,
    opacity: npt.NDArray[np.floating[Any]],
    foreground: npt.NDArray[np.bool_],
    views: list[View],
    mvs_points: F64,
    extent: float,
    cfg: SupportConfig | None = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], dict[str, Any]]:
    """A-1 labels for the `foreground` Gaussians (background-sphere ones are left to the caller).

    Variant chosen by the D-4.1 protocol on seeds 11-13: view count x angular spread, score
    halved for Gaussians farther than 2 % of the extent from any stereo point, counted only
    where the stereo reconstruction covers the Gaussian's image cell in >= 3 views (a gap in
    sparse stereo is no evidence against a Gaussian). Returns (class, confidence, report).
    """
    cfg = cfg or SupportConfig()
    pos = positions[foreground].astype(np.float64)
    n, spread = view_support(pos, opacity[foreground], views, cfg)
    covered, _ = stereo_evidence(pos, views, mvs_points)
    far = far_from_points(pos, mvs_points, MVS_RADIUS_FRACTION * extent)
    flagged = far & (covered >= MIN_COVERED_VIEWS)
    cls, conf = classify(n, spread, cfg, far_from_mvs=flagged)
    report = {
        "algorithm": "A-1 view support + covered stereo distance (adopted, D-4.1)",
        "views": len(views),
        "observed": round(float((cls == OBSERVED).mean()), 3),
        "inferred": round(float((cls == INFERRED).mean()), 3),
        "guessed": round(float((cls == GUESSED).mean()), 3),
        "flagged_off_surface": round(float(flagged.mean()), 3),
    }
    return cls, conf, report
