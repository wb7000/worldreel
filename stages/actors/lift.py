"""Lift 2D person tracks (per-frame masks) into 3D actor tracks on the ground plane.

Classical, CPU only:
1. Ground plane: RANSAC plane fit on the sparse SfM points, keeping only planes whose normal is
   within `max_tilt_deg` of world +Y (the world is the first camera's frame, roughly upright),
   choosing the plane with most inliers and orienting its normal upwards.
2. Feet: the ray through the bottom-centre of the person's mask (median x of the lowest few
   mask rows) intersected with the ground plane.
3. Height: the ray through the top of the mask meets the vertical line above the feet.
4. Smoothing: median filter over time, then facing = direction of motion on the plane.
Coordinates are WorldReel (spec §2): cameras look along -Z, +Y up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from eval.geometry import quat_to_matrix

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Plane:
    normal: F64  # unit, pointing up
    offset: float  # plane: normal · x = offset
    inliers: int

    def height(self, points: F64) -> F64:
        result: F64 = points @ self.normal - self.offset
        return result


def fit_ground_plane(
    points: F64,
    *,
    threshold: float | None = None,
    max_tilt_deg: float = 35.0,
    iterations: int = 2000,
    seed: int = 0,
) -> Plane:
    """RANSAC ground plane among near-horizontal planes. `threshold` defaults to 1 % of the
    median distance of the points from their median."""
    if len(points) < 3:
        raise ValueError("need at least 3 points")
    rng = np.random.default_rng(seed)
    centre = np.median(points, axis=0)
    thr = threshold or 0.01 * float(np.median(np.linalg.norm(points - centre, axis=1)))
    cos_tilt = math.cos(math.radians(max_tilt_deg))
    best: Plane | None = None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        if n[1] < 0:
            n = -n
        if n[1] < cos_tilt:
            continue
        d = float(n @ a)
        count = int((np.abs(points @ n - d) < thr).sum())
        if best is None or count > best.inliers:
            best = Plane(n, d, count)
    if best is None:
        raise ValueError("no near-horizontal plane found")
    # Least-squares refinement on the inliers.
    inl = points[np.abs(points @ best.normal - best.offset) < thr]
    mu = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - mu)
    n = vt[2] if vt[2][1] > 0 else -vt[2]
    return Plane(n, float(n @ mu), len(inl))


def pixel_ray(
    u: float, v: float, k: dict[str, float], position: list[float], rotation: list[float]
) -> tuple[F64, F64]:
    """World-space origin and unit direction of the ray through pixel (u, v)."""
    d_cam = np.array([(u - k["cx"]) / k["fx"], -(v - k["cy"]) / k["fy"], -1.0])
    d = quat_to_matrix(rotation) @ d_cam
    return np.asarray(position, np.float64), d / np.linalg.norm(d)


def intersect_plane(origin: F64, direction: F64, plane: Plane) -> F64 | None:
    denom = float(plane.normal @ direction)
    if abs(denom) < 1e-9:
        return None
    t = (plane.offset - float(plane.normal @ origin)) / denom
    return None if t <= 0 else origin + t * direction


def mask_foot_and_head(
    mask: npt.NDArray[np.bool_], rows: int = 3
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """(u, v) of the bottom-centre and top-centre of a binary mask, or None if empty."""
    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        return None
    bottom, top = vs.max(), vs.min()
    foot_u = float(np.median(us[vs >= bottom - rows + 1]))
    head_u = float(np.median(us[vs <= top + rows - 1]))
    return (foot_u, float(bottom) + 0.5), (head_u, float(top) - 0.5)


@dataclass
class Track3D:
    times: list[float] = field(default_factory=list)
    positions: list[list[float]] = field(default_factory=list)
    rotations: list[list[float]] = field(default_factory=list)
    height: float = 0.0
    frames: list[int] = field(default_factory=list)


def yaw_quat(direction: F64) -> list[float]:
    """Rotation about +Y that points the actor's -Z axis along `direction` (projected)."""
    yaw = math.atan2(-direction[0], -direction[2])
    return [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]


def lift_track(
    masks: dict[int, npt.NDArray[np.bool_]],
    poses: list[dict[str, list[float]]],
    times: list[float],
    k: dict[str, float],
    plane: Plane,
    *,
    smooth: int = 5,
    min_speed: float = 0.05,
) -> Track3D | None:
    """3D track of one person from its per-frame masks (frame index → mask)."""
    feet: dict[int, F64] = {}
    heights: list[float] = []
    for f in sorted(masks):
        fh = mask_foot_and_head(masks[f])
        if fh is None:
            continue
        (fu, fv), (hu, hv) = fh
        o, d = pixel_ray(fu, fv, k, poses[f]["position"], poses[f]["rotation"])
        foot = intersect_plane(o, d, plane)
        if foot is None:
            continue
        feet[f] = foot
        o2, d2 = pixel_ray(hu, hv, k, poses[f]["position"], poses[f]["rotation"])
        # Closest point on the head ray to the vertical line through the foot.
        w = o2 - foot
        a, b, c = float(d2 @ d2), float(d2 @ plane.normal), 1.0
        dd, e = float(d2 @ w), float(plane.normal @ w)
        den = a * c - b * b
        if abs(den) > 1e-9:
            s_line = (a * e - b * dd) / den
            if s_line > 0:
                heights.append(s_line)
    if len(feet) < 2:
        return None
    frames = sorted(feet)
    pos = np.stack([feet[f] for f in frames])
    # Median filter with a window symmetric around each sample (shrinking at the ends): a
    # one-sided window at the ends would drag a steadily walking person towards the middle.
    half = smooth // 2
    n = len(pos)
    smoothed = np.stack(
        [
            np.median(pos[i - r : i + r + 1], axis=0)
            for i in range(n)
            for r in [min(half, i, n - 1 - i)]
        ]
    )
    track = Track3D(height=float(np.median(heights)) if heights else 1.7, frames=frames)
    last_dir = np.array([0.0, 0.0, -1.0])
    for i, f in enumerate(frames):
        j0, j1 = max(0, i - 1), min(len(frames) - 1, i + 1)
        dt = times[frames[j1]] - times[frames[j0]]
        vel = (smoothed[j1] - smoothed[j0]) / dt if dt > 0 else np.zeros(3)
        vel = vel - plane.normal * float(vel @ plane.normal)
        if np.linalg.norm(vel) > min_speed:
            last_dir = vel / np.linalg.norm(vel)
        track.times.append(round(times[f], 6))
        track.positions.append([float(x) for x in smoothed[i]])
        track.rotations.append(yaw_quat(last_dir))
    return track
