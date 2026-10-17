"""Classical incremental structure-from-motion for an ordered frame sequence (CPU only).

Pipeline (all standard techniques, see Hartley & Zisserman, "Multiple View Geometry", and
Schönberger & Frahm 2016, "Structure-from-Motion Revisited"):
1. SIFT features per frame; ratio-test + mutual matching between nearby frames.
2. Initialise from the first frame and the first later frame with enough parallax:
   essential matrix (RANSAC) → relative pose → triangulation.
3. Register the remaining frames in order with PnP (RANSAC + Levenberg-Marquardt refinement),
   triangulating new points against already-registered neighbours.
4. Global bundle adjustment (SciPy least squares, Huber loss), then outlier removal.

Everything here uses the OpenCV camera convention (x right, y down, z forward); conversion to
WorldReel's convention happens in `stages/camera/classical.py`. Monocular SfM recovers the scene
only up to scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

F64 = npt.NDArray[np.float64]


@dataclass
class SfmConfig:
    max_features: int = 4000
    ratio: float = 0.75
    match_window: int = 3  # match each frame against this many previous registered frames
    init_min_parallax_frac: float = 0.03  # of image width, median feature displacement
    init_max_frame: int = 25
    init_anywhere: bool = True  # if frame 0 cannot start, search pairs across the video
    init_search_stride: int = 2
    init_search_span: int = 12
    init_min_inliers: int = 80
    ransac_px: float = 1.0
    pnp_ransac_px: float = 2.0
    min_pnp_inliers: int = 30
    min_triangulation_deg: float = 1.5
    max_reproj_px: float = 2.0
    bundle_adjust: bool = True
    ba_max_nfev: int = 60
    ba_fix_scale: bool = True
    ba_every: int = 0  # also bundle-adjust after every N newly registered frames (0 = off)
    ba_periodic_nfev: int = 20


@dataclass
class SfmResult:
    rotations: list[F64 | None]  # world-to-camera R per frame (OpenCV), None if unregistered
    translations: list[F64 | None]  # world-to-camera t per frame
    points: F64  # (M, 3)
    colors: npt.NDArray[np.uint8]  # (M, 3) RGB
    observations: npt.NDArray[np.float64]  # (K, 4): frame, point, u, v
    stats: dict[str, float] = field(default_factory=dict)


class SfmError(Exception):
    pass


def _pose(rot: list[F64 | None], trans: list[F64 | None], f: int) -> tuple[F64, F64]:
    r, t = rot[f], trans[f]
    if r is None or t is None:
        raise SfmError(f"frame {f} is not registered")
    return r, t


def _rodrigues(v: npt.ArrayLike) -> F64:
    """Rotation vector → rotation matrix (or matrix → vector), as float64."""
    return np.asarray(cv2.Rodrigues(np.asarray(v, np.float64))[0], np.float64)


@dataclass
class _Frame:
    kps: F64  # (n, 2)
    desc: npt.NDArray[np.float32]
    colors: npt.NDArray[np.uint8]  # (n, 3) RGB at keypoints
    point_of: dict[int, int] = field(default_factory=dict)  # keypoint index -> map point id


def _features(
    images: list[npt.NDArray[np.uint8]],
    cfg: SfmConfig,
    ignore: list[npt.NDArray[np.bool_]] | None = None,
) -> list[_Frame]:
    sift = cv2.SIFT.create(nfeatures=cfg.max_features)
    frames: list[_Frame] = []
    for rgb in images:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kps], np.float64).reshape(-1, 2)
        ui = np.clip(np.round(pts[:, 0]).astype(int), 0, rgb.shape[1] - 1)
        vi = np.clip(np.round(pts[:, 1]).astype(int), 0, rgb.shape[0] - 1)
        if ignore is not None and desc is not None:
            # Moving people violate the static-scene assumption: drop their keypoints.
            keep = ~ignore[len(frames)][vi, ui]
            pts, ui, vi, desc = pts[keep], ui[keep], vi[keep], desc[keep]
        frames.append(
            _Frame(
                pts,
                np.asarray(desc, np.float32)
                if desc is not None
                else np.zeros((0, 128), np.float32),
                rgb[vi, ui],
            )
        )
    return frames


def _match(a: _Frame, b: _Frame, ratio: float) -> npt.NDArray[np.int64]:
    """Mutual ratio-test matches as (k, 2) index pairs into a and b."""
    if len(a.desc) < 2 or len(b.desc) < 2:
        return np.zeros((0, 2), np.int64)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def one_way(x: npt.NDArray[np.float32], y: npt.NDArray[np.float32]) -> dict[int, int]:
        out = {}
        for pair in bf.knnMatch(x, y, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                out[pair[0].queryIdx] = pair[0].trainIdx
        return out

    ab, ba = one_way(a.desc, b.desc), one_way(b.desc, a.desc)
    pairs = [(i, j) for i, j in ab.items() if ba.get(j) == i]
    return np.array(pairs, np.int64).reshape(-1, 2)


def _project(r: F64, t: F64, k: F64, x: F64) -> tuple[F64, F64]:
    cam = x @ r.T + t
    z = cam[:, 2]
    uv = cam[:, :2] / z[:, None] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    return uv, z


def _triangulate(
    k: F64, r1: F64, t1: F64, r2: F64, t2: F64, p1: F64, p2: F64, cfg: SfmConfig
) -> tuple[F64, npt.NDArray[np.bool_]]:
    """Triangulate and keep points in front of both cameras, with small reprojection error and
    enough triangulation angle."""
    if len(p1) == 0:
        return np.zeros((0, 3)), np.zeros(0, bool)
    proj1 = k @ np.hstack([r1, t1[:, None]])
    proj2 = k @ np.hstack([r2, t2[:, None]])
    xh = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)
    x = (xh[:3] / xh[3]).T
    uv1, z1 = _project(r1, t1, k, x)
    uv2, z2 = _project(r2, t2, k, x)
    err = np.maximum(np.linalg.norm(uv1 - p1, axis=1), np.linalg.norm(uv2 - p2, axis=1))
    c1, c2 = -r1.T @ t1, -r2.T @ t2
    ray1, ray2 = x - c1, x - c2
    cos = np.sum(ray1 * ray2, axis=1) / (
        np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1) + 1e-12
    )
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ok = (z1 > 0) & (z2 > 0) & (err < cfg.max_reproj_px) & (angle > cfg.min_triangulation_deg)
    ok &= np.all(np.isfinite(x), axis=1)
    return x, ok


class _Map:
    def __init__(self) -> None:
        self.points: list[F64] = []
        self.colors: list[npt.NDArray[np.uint8]] = []
        self.obs: list[tuple[int, int, float, float]] = []  # frame, point, u, v

    def add(self, x: F64, color: npt.NDArray[np.uint8]) -> int:
        self.points.append(x)
        self.colors.append(color)
        return len(self.points) - 1


def run_sfm(
    images: list[npt.NDArray[np.uint8]],
    k: F64,
    cfg: SfmConfig | None = None,
    ignore: list[npt.NDArray[np.bool_]] | None = None,
) -> SfmResult:
    """`ignore`: optional per-frame masks of pixels whose features must not be used."""
    cfg = cfg or SfmConfig()
    n = len(images)
    if n < 3:
        raise SfmError("need at least 3 frames")
    width = images[0].shape[1]
    frames = _features(images, cfg, ignore)
    rot: list[F64 | None] = [None] * n
    trans: list[F64 | None] = [None] * n
    world = _Map()
    match_cache: dict[tuple[int, int], npt.NDArray[np.int64]] = {}

    def matches(i: int, j: int) -> npt.NDArray[np.int64]:
        if (i, j) not in match_cache:
            match_cache[(i, j)] = _match(frames[i], frames[j], cfg.ratio)
        return match_cache[(i, j)]

    # ---- initialisation. Prefer frame 0 with the frame 2..init_max_frame that gives the widest
    # baseline (median feature displacement) while still having enough essential-matrix inliers:
    # a short initial baseline leaves the first structure poorly conditioned and bundle
    # adjustment then settles in a sheared local minimum (DECISIONS D-3a.4). Real handheld video
    # often starts with an untrackable shot, so if frame 0 cannot start, the same search runs
    # from other start frames across the video (DECISIONS D-3a.10).
    def search(
        starts: list[int], span: int
    ) -> tuple[int, int, npt.NDArray[np.int64], F64, F64] | None:
        best = None
        best_parallax = 0.0
        for i0 in starts:
            for j in range(i0 + 2, min(n, i0 + span + 1)):
                m = matches(i0, j)
                if len(m) < cfg.init_min_inliers:
                    continue
                pa, pb = frames[i0].kps[m[:, 0]], frames[j].kps[m[:, 1]]
                parallax = float(np.median(np.linalg.norm(pa - pb, axis=1)))
                if parallax < cfg.init_min_parallax_frac * width or parallax <= best_parallax:
                    continue
                e, mask = cv2.findEssentialMat(
                    pa, pb, k, method=cv2.RANSAC, prob=0.999, threshold=cfg.ransac_px
                )
                if e is None or e.shape != (3, 3):
                    continue
                inl, r, t, mask2 = cv2.recoverPose(e, pa, pb, k, mask=mask)
                if inl < cfg.init_min_inliers:
                    continue
                best = (
                    i0,
                    j,
                    m[mask2.ravel() > 0],
                    np.asarray(r, np.float64),
                    np.asarray(t, np.float64).ravel(),
                )
                best_parallax = parallax
        return best

    init = search([0], cfg.init_max_frame)
    if init is None and cfg.init_anywhere:
        init = search(list(range(0, n - 2, cfg.init_search_stride)), cfg.init_search_span)
    if init is None:
        raise SfmError("could not initialise: no frame pair with enough matches and parallax")
    i0, j0, m0, r_init, t_init = init
    rot[i0], trans[i0] = np.eye(3), np.zeros(3)
    rot[j0], trans[j0] = r_init, t_init
    x, ok = _triangulate(
        k,
        *_pose(rot, trans, i0),
        *_pose(rot, trans, j0),
        frames[i0].kps[m0[:, 0]],
        frames[j0].kps[m0[:, 1]],
        cfg,
    )
    for (a, b), xi, good in zip(m0, x, ok, strict=True):
        if good:
            pid = world.add(xi, frames[i0].colors[a])
            frames[i0].point_of[int(a)] = pid
            frames[j0].point_of[int(b)] = pid
    if len(world.points) < 50:
        raise SfmError(f"initial triangulation produced only {len(world.points)} points")

    # ---- incremental registration: frames between the init pair, then outward in both
    # directions, alternating, so each new frame is next to already-registered ones.
    forward, backward = list(range(j0 + 1, n)), list(range(i0 - 1, -1, -1))
    order = list(range(i0 + 1, j0))
    for a in range(max(len(forward), len(backward))):
        order += forward[a : a + 1] + backward[a : a + 1]
    newly_registered = 0
    for i in order:
        registered = [f for f in range(n) if rot[f] is not None]
        neighbours = sorted(registered, key=lambda f: abs(f - i))[: cfg.match_window]
        obj, img, kp_ids = [], [], []
        seen: set[int] = set()
        for f in neighbours:
            m = matches(min(f, i), max(f, i))
            if f > i:
                m = m[:, ::-1]
            for kf, ki in m:
                known = frames[f].point_of.get(int(kf))
                if known is not None and int(ki) not in seen:
                    seen.add(int(ki))
                    obj.append(world.points[known])
                    img.append(frames[i].kps[ki])
                    kp_ids.append((int(ki), known))
        if len(obj) < cfg.min_pnp_inliers:
            continue
        ok_pnp, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(obj),
            np.array(img),
            k,
            None,
            iterationsCount=300,
            reprojectionError=cfg.pnp_ransac_px,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok_pnp or inliers is None or len(inliers) < cfg.min_pnp_inliers:
            continue
        idx = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(
            np.array(obj)[idx], np.array(img)[idx], k, None, rvec, tvec
        )
        rot[i], trans[i] = _rodrigues(rvec), np.asarray(tvec, np.float64).ravel()
        for n_idx in idx:
            ki, pid = kp_ids[int(n_idx)]
            frames[i].point_of[ki] = pid
        newly_registered += 1
        # Triangulate new points with each neighbour.
        for f in neighbours:
            m = matches(min(f, i), max(f, i))
            if f > i:
                m = m[:, ::-1]
            new = np.array(
                [
                    (kf, ki)
                    for kf, ki in m
                    if int(kf) not in frames[f].point_of and int(ki) not in frames[i].point_of
                ],
                np.int64,
            ).reshape(-1, 2)
            if len(new) == 0:
                continue
            x, ok = _triangulate(
                k,
                *_pose(rot, trans, f),
                *_pose(rot, trans, i),
                frames[f].kps[new[:, 0]],
                frames[i].kps[new[:, 1]],
                cfg,
            )
            for (kf, ki), xi, good in zip(new, x, ok, strict=True):
                if good:
                    pid = world.add(xi, frames[i].colors[ki])
                    frames[f].point_of[int(kf)] = pid
                    frames[i].point_of[int(ki)] = pid
        if cfg.bundle_adjust and cfg.ba_every and newly_registered % cfg.ba_every == 0:
            pts_now = np.array(world.points)
            obs_now = np.array(_collect_obs(frames, rot), np.float64)
            _bundle_adjust(
                rot, trans, k, pts_now, obs_now, cfg, max_nfev=cfg.ba_periodic_nfev, ref=i0
            )
            world.points = list(pts_now)

    world.obs = _collect_obs(frames, rot)

    points = np.array(world.points)
    obs = np.array(world.obs, np.float64)
    stats: dict[str, float] = {
        "registered": float(sum(r is not None for r in rot)),
        "frames": float(n),
        "init_first_frame": float(i0),
        "init_second_frame": float(j0),
    }
    stats["reproj_rmse_before_ba"] = _reproj_rmse(rot, trans, k, points, obs)
    if cfg.bundle_adjust:
        _bundle_adjust(rot, trans, k, points, obs, cfg, ref=i0)
    stats["reproj_rmse_after_ba"] = _reproj_rmse(rot, trans, k, points, obs)
    keep_pts, keep_obs = _filter(rot, trans, k, points, obs, cfg.max_reproj_px * 2)
    remap = -np.ones(len(points), np.int64)
    remap[keep_pts] = np.arange(keep_pts.sum())
    obs = obs[keep_obs]
    obs[:, 1] = remap[obs[:, 1].astype(int)]
    stats["points"] = float(keep_pts.sum())
    return SfmResult(
        rot, trans, points[keep_pts], np.array(world.colors, np.uint8)[keep_pts], obs, stats
    )


def _residuals_for(
    rot: list[F64 | None], trans: list[F64 | None], k: F64, points: F64, obs: F64
) -> F64:
    res = np.zeros((len(obs), 2))
    for f in np.unique(obs[:, 0]).astype(int):
        sel = obs[:, 0] == f
        uv, _ = _project(*_pose(rot, trans, f), k, points[obs[sel, 1].astype(int)])
        res[sel] = uv - obs[sel, 2:4]
    return res


def _reproj_rmse(
    rot: list[F64 | None], trans: list[F64 | None], k: F64, points: F64, obs: F64
) -> float:
    if len(obs) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(_residuals_for(rot, trans, k, points, obs) ** 2, axis=1))))


def _filter(
    rot: list[F64 | None], trans: list[F64 | None], k: F64, points: F64, obs: F64, max_px: float
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    err = np.linalg.norm(_residuals_for(rot, trans, k, points, obs), axis=1)
    good_obs = err < max_px
    counts = np.bincount(obs[good_obs, 1].astype(int), minlength=len(points))
    keep_pts = counts >= 2
    keep_obs = good_obs & keep_pts[obs[:, 1].astype(int)]
    return keep_pts, keep_obs


def _collect_obs(
    frames: list[_Frame], rot: list[F64 | None]
) -> list[tuple[int, int, float, float]]:
    out = []
    for fi, fr in enumerate(frames):
        if rot[fi] is None:
            continue
        for kp, pid in fr.point_of.items():
            u, v = fr.kps[kp]
            out.append((fi, pid, float(u), float(v)))
    return out


def _bundle_adjust(
    rot: list[F64 | None],
    trans: list[F64 | None],
    k: F64,
    points: F64,
    obs: F64,
    cfg: SfmConfig,
    max_nfev: int | None = None,
    ref: int = 0,
) -> None:
    """Refine all registered cameras (except the first) and all points in place.

    Gauge: the first camera fixes position and orientation. With `ba_fix_scale`, one extra
    residual also pins the distance from the first camera to the farthest camera, removing the
    free scale direction; without it the solver drifts along that flat direction and settles
    in a slightly sheared solution (measured: ~3 degree orientation bias, DECISIONS D-3a.4)."""
    cams = [f for f in range(len(rot)) if rot[f] is not None and f != ref]
    cam_index = {f: i for i, f in enumerate(cams)}
    n_c, n_p = len(cams), len(points)
    use = np.array([int(f) in cam_index or int(f) == ref for f in obs[:, 0]])
    o = obs[use]
    x0 = np.concatenate(
        [
            np.concatenate([_rodrigues(_pose(rot, trans, f)[0]).ravel(), _pose(rot, trans, f)[1]])
            for f in cams
        ]
        + [points.ravel()]
    )
    fi = o[:, 0].astype(int)
    pi = o[:, 1].astype(int)
    anchor = (
        max(cams, key=lambda f: float(np.linalg.norm(_pose(rot, trans, f)[1]))) if cams else None
    )
    anchor_dist = float(np.linalg.norm(_pose(rot, trans, anchor)[1])) if anchor is not None else 0.0
    use_anchor = cfg.ba_fix_scale and anchor is not None and anchor_dist > 0
    anchor_weight = 100.0

    def residuals(params: F64) -> F64:
        cam_params = params[: n_c * 6].reshape(n_c, 6)
        pts = params[n_c * 6 :].reshape(n_p, 3)
        res = np.empty((len(o), 2))
        for f in np.unique(fi):
            sel = fi == f
            if f == ref:
                r, t = _pose(rot, trans, ref)
            else:
                c = cam_params[cam_index[f]]
                r, t = _rodrigues(c[:3]), c[3:]
            uv, _ = _project(r, t, k, pts[pi[sel]])
            res[sel] = uv - o[sel, 2:4]
        out = res.ravel()
        if use_anchor:
            t_a = cam_params[cam_index[anchor], 3:]  # type: ignore[index]
            out = np.append(out, anchor_weight * (np.linalg.norm(t_a) - anchor_dist))
        return out

    sparsity = lil_matrix((len(o) * 2 + int(use_anchor), n_c * 6 + n_p * 3), dtype=int)
    if use_anchor:
        sparsity[len(o) * 2, cam_index[anchor] * 6 + 3 : cam_index[anchor] * 6 + 6] = 1  # type: ignore[index]
    rows = np.arange(len(o))
    for s in range(6):
        has_cam = np.array([f in cam_index for f in fi])
        cols = np.array([cam_index[f] * 6 + s for f in fi[has_cam]])
        sparsity[2 * rows[has_cam], cols] = 1
        sparsity[2 * rows[has_cam] + 1, cols] = 1
    for s in range(3):
        sparsity[2 * rows, n_c * 6 + pi * 3 + s] = 1
        sparsity[2 * rows + 1, n_c * 6 + pi * 3 + s] = 1
    result = least_squares(
        residuals,
        x0,
        jac_sparsity=sparsity,
        x_scale="jac",
        loss="huber",
        f_scale=1.0,
        method="trf",
        max_nfev=max_nfev or cfg.ba_max_nfev,
    )
    params = result.x
    for f in cams:
        c = params[cam_index[f] * 6 : cam_index[f] * 6 + 6]
        rot[f], trans[f] = _rodrigues(c[:3]), c[3:].copy()
    points[:] = params[n_c * 6 :].reshape(n_p, 3)
