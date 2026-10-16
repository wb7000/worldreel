"""Real `reconstruct` stage v1 (classical, CPU): plane-sweep multi-view stereo + fusion.

For every reference frame (every `ref_stride` frames), fronto-parallel depth planes are swept
through the scene (uniform in inverse depth); neighbouring frames are warped onto each plane with
the plane-induced homography H = K (R + t nᵀ / d) K⁻¹ and compared with zero-mean normalised
cross-correlation (Collins 1996 plane sweep; Gallup et al. 2007). Per pixel the best depth wins,
refined to sub-plane precision with a parabola. Depths are kept only if the match is good, the
reference patch is textured, and the depth agrees with a neighbouring reference frame's depth map
(geometric consistency). Surviving pixels are back-projected and fused on a voxel grid.

All points are `observed` (seen in at least two views); confidence comes from the NCC score.
Requires the classical `camera` stage (poses, intrinsics, sparse points for the depth range).
"""

from __future__ import annotations

from typing import Any, ClassVar

import cv2
import numpy as np
import numpy.typing as npt

from eval.geometry import quat_to_matrix
from formats.wreel import PROVENANCE_CLASSES, PointCloud
from pipeline.io import (
    CAMERA_JSON,
    FRAMES_DIR,
    POINTS_JSON,
    POINTS_PTS,
    read_json,
    save_npy,
    write_json,
)
from pipeline.stage import Impl, StageContext, StageError
from stages.camera.classical import SPARSE_COLORS, SPARSE_POINTS, load_frames

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
OBSERVED = PROVENANCE_CLASSES.index("observed")
_FLIP = np.diag([1.0, -1.0, -1.0])

DEFAULTS: dict[str, Any] = {
    "stereo_width": 240,
    "ref_stride": 3,
    "source_offsets": [-4, -2, 2, 4],
    "planes": 64,
    "window": 7,
    "ncc_min": 0.6,
    "texture_min": 2e-4,  # minimum variance of the reference patch (intensity in [0, 1])
    "consistency_rel": 0.03,
    "max_points": 250_000,
}


def cv_pose(position: list[float], rotation: list[float]) -> tuple[F64, F64]:
    """WorldReel camera pose → OpenCV world-to-camera (R, t) in OpenCV world axes."""
    r_wc = _FLIP @ quat_to_matrix(rotation) @ _FLIP
    c = _FLIP @ np.asarray(position, np.float64)
    r_cw = r_wc.T
    return r_cw, -r_cw @ c


def _box(x: F32, win: int) -> F32:
    out = np.asarray(
        cv2.boxFilter(x, cv2.CV_32F, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT),
        np.float32,
    )
    return out


def sweep(
    ref: F32,
    sources: list[tuple[F32, F64, F64]],
    k: F64,
    r_ref: F64,
    t_ref: F64,
    depths: F64,
    win: int,
) -> tuple[F32, F32]:
    """Returns (cost volume (D, H, W) with 1 - NCC aggregated over the best two sources,
    reference patch variance (H, W))."""
    h, w = ref.shape
    mu_r = _box(ref, win)
    var_r = _box(ref * ref, win) - mu_r * mu_r
    kinv = np.linalg.inv(k)
    n = np.array([[0.0, 0.0, 1.0]])
    volume = np.empty((len(depths), h, w), np.float32)
    ones = np.ones_like(ref)
    for di, d in enumerate(depths):
        costs = []
        for img, r_s, t_s in sources:
            r_sr = r_s @ r_ref.T
            t_sr = t_s - r_sr @ t_ref
            hom = k @ (r_sr + t_sr[:, None] @ n / d) @ kinv
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            warped = np.asarray(
                cv2.warpPerspective(img, hom, (w, h), flags=flags, borderValue=0), np.float32
            )
            valid = np.asarray(
                cv2.warpPerspective(ones, hom, (w, h), flags=flags, borderValue=0), np.float32
            )
            mu_s = _box(warped, win)
            var_s = _box(warped * warped, win) - mu_s * mu_s
            cov = _box(ref * warped, win) - mu_r * mu_s
            ncc = cov / np.sqrt(np.maximum(var_r * var_s, 1e-12))
            cost = np.where(_box(valid, win) > 0.999, 1.0 - ncc, 2.0).astype(np.float32)
            costs.append(cost)
        stack = np.sort(np.stack(costs), axis=0)
        volume[di] = stack[: min(2, len(costs))].mean(axis=0)
    return volume, var_r


def best_depth(volume: F32, depths: F64) -> tuple[F32, F32]:
    """Winner-take-all with parabola refinement in inverse depth. Returns (depth, best cost);
    depth 0 where the minimum sits on the range boundary."""
    d_n = len(depths)
    idx = np.argmin(volume, axis=0)
    rows, cols = np.indices(idx.shape)
    c0 = volume[idx, rows, cols]
    inv = 1.0 / depths
    inner = (idx > 0) & (idx < d_n - 1)
    im, ip = np.clip(idx - 1, 0, d_n - 1), np.clip(idx + 1, 0, d_n - 1)
    cm, cp = volume[im, rows, cols], volume[ip, rows, cols]
    denom = cm - 2 * c0 + cp
    offset = np.where(inner & (denom > 1e-6), 0.5 * (cm - cp) / np.maximum(denom, 1e-6), 0.0)
    step = inv[1] - inv[0] if d_n > 1 else 0.0
    inv_best = inv[idx] + np.clip(offset, -0.5, 0.5) * step
    depth = np.where(inner, 1.0 / inv_best, 0.0).astype(np.float32)
    return depth, c0.astype(np.float32)


def backproject(depth: F32, k: F64, r_cw: F64, t_cw: F64) -> tuple[F64, npt.NDArray[np.bool_]]:
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    valid = depth > 0
    z = depth[valid].astype(np.float64)
    cam = np.stack(
        [(u[valid] - k[0, 2]) / k[0, 0] * z, (v[valid] - k[1, 2]) / k[1, 1] * z, z], axis=1
    )
    return (cam - t_cw) @ r_cw, valid  # camera → world (OpenCV axes)


def consistent(
    depth: F32, k: F64, pose: tuple[F64, F64], other: F32, other_pose: tuple[F64, F64], rel: float
) -> npt.NDArray[np.bool_]:
    """Pixels of `depth` whose 3D point lands on `other`'s depth map within a relative tolerance."""
    world, valid = backproject(depth, k, *pose)
    r2, t2 = other_pose
    cam = world @ r2.T + t2
    z = cam[:, 2]
    h, w = other.shape
    u = np.round(cam[:, 0] / np.maximum(z, 1e-9) * k[0, 0] + k[0, 2]).astype(int)
    v = np.round(cam[:, 1] / np.maximum(z, 1e-9) * k[1, 1] + k[1, 2]).astype(int)
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ok = np.zeros(len(z), bool)
    d_other = other[v[inside], u[inside]]
    ok[inside] = (d_other > 0) & (np.abs(d_other - z[inside]) / z[inside] < rel)
    out = np.zeros(depth.shape, bool)
    out[valid] = ok
    return out


class PlaneSweepReconstructStage:
    name: ClassVar[str] = "reconstruct"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "real"
    depends_on: ClassVar[tuple[str, ...]] = ("frames", "camera")

    def run(self, ctx: StageContext) -> None:
        cfg = {**DEFAULTS, **ctx.config}
        cam = read_json(ctx.inputs["camera"] / CAMERA_JSON)
        sparse_path = ctx.inputs["camera"] / SPARSE_POINTS
        if not sparse_path.exists():
            raise StageError(
                "plane-sweep reconstruction needs the classical camera stage (sparse points)"
            )
        sparse = np.load(sparse_path).astype(np.float64) @ _FLIP.T  # → OpenCV axes
        sparse_colors = np.load(ctx.inputs["camera"] / SPARSE_COLORS)
        images = load_frames(ctx.inputs["frames"] / FRAMES_DIR)
        n = len(images)
        full_h, full_w = images[0].shape[:2]
        sw = int(cfg["stereo_width"])
        sh = round(full_h * sw / full_w)
        s = sw / full_w
        ki = cam["intrinsics"]
        k = np.array(
            [
                [ki["fx"] * s, 0, (ki["cx"] + 0.5) * s - 0.5],
                [0, ki["fy"] * s, (ki["cy"] + 0.5) * s - 0.5],
                [0, 0, 1],
            ]
        )
        small = [cv2.resize(im, (sw, sh), interpolation=cv2.INTER_AREA) for im in images]
        gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for im in small]
        poses = [cv_pose(p["position"], p["rotation"]) for p in cam["poses"]]
        registered = [bool(p.get("registered", True)) for p in cam["poses"]]

        refs = [i for i in range(0, n, int(cfg["ref_stride"])) if registered[i]]
        depth_maps: dict[int, F32] = {}
        scores: dict[int, F32] = {}
        for r in refs:
            srcs = [r + o for o in cfg["source_offsets"] if 0 <= r + o < n and registered[r + o]]
            if len(srcs) < 2:
                continue
            r_ref, t_ref = poses[r]
            z = (sparse @ r_ref.T + t_ref)[:, 2]
            z = z[z > 0]
            if len(z) < 20:
                continue
            near = max(1e-3, float(np.percentile(z, 2)) * 0.7)
            far = float(np.percentile(z, 98)) * 1.3
            depths = 1.0 / np.linspace(1.0 / near, 1.0 / far, int(cfg["planes"]))
            volume, var_r = sweep(
                gray[r],
                [(gray[j], *poses[j]) for j in srcs],
                k,
                r_ref,
                t_ref,
                depths,
                int(cfg["window"]),
            )
            depth, cost = best_depth(volume, depths)
            good = (1.0 - cost > cfg["ncc_min"]) & (var_r > cfg["texture_min"])
            depth_maps[r] = np.where(good, depth, 0.0).astype(np.float32)
            scores[r] = (1.0 - cost).astype(np.float32)
        if not depth_maps:
            raise StageError("no reference frame produced a depth map")

        keys = sorted(depth_maps)
        filtered: dict[int, F32] = {}
        for i, r in enumerate(keys):
            nbrs = [keys[j] for j in (i - 1, i + 1) if 0 <= j < len(keys)]
            keep = np.zeros(depth_maps[r].shape, bool)
            for o in nbrs:
                keep |= consistent(
                    depth_maps[r],
                    k,
                    poses[r],
                    depth_maps[o],
                    poses[o],
                    float(cfg["consistency_rel"]),
                )
            filtered[r] = np.where(keep if nbrs else depth_maps[r] > 0, depth_maps[r], 0.0).astype(
                np.float32
            )

        pts, cols, conf = [], [], []
        for r, depth in filtered.items():
            world, valid = backproject(depth, k, *poses[r])
            pts.append(world)
            cols.append(small[r][valid])
            conf.append(np.clip(scores[r][valid], 0.5, 1.0))
        pts.append(sparse)
        cols.append(sparse_colors)
        conf.append(np.full(len(sparse), 0.9))
        points = np.concatenate(pts)
        colors = np.concatenate(cols).astype(np.uint8)
        confidence = np.concatenate(conf).astype(np.float32)

        # Voxel fusion: one point per cell, cell ≈ one stereo pixel at the median depth.
        median_depth = float(
            np.median([np.median(d[d > 0]) for d in filtered.values() if np.any(d > 0)])
        )
        voxel = median_depth / k[0, 0]
        cells = np.floor(points / voxel).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        first = np.sort(first)
        if len(first) > cfg["max_points"]:
            first = first[np.linspace(0, len(first) - 1, int(cfg["max_points"])).astype(int)]
        points, colors, confidence = points[first], colors[first], confidence[first]

        world_pts = (points @ _FLIP.T).astype(np.float32)  # back to WorldReel axes
        cloud = PointCloud(
            world_pts, colors, np.full(len(world_pts), OBSERVED, np.uint8), confidence
        )
        (ctx.out / POINTS_PTS).write_bytes(cloud.pack())
        depth_dir = ctx.out / "depth"
        for r, d in filtered.items():
            save_npy(depth_dir / f"{r:04d}.npy", d)
        k_out = {
            "fx": k[0, 0],
            "fy": k[1, 1],
            "cx": k[0, 2],
            "cy": k[1, 2],
            "width": sw,
            "height": sh,
        }
        write_json(
            ctx.out / POINTS_JSON,
            {
                "method": "plane-sweep MVS (NCC, best-2 aggregation, geometric consistency)",
                "count": len(world_pts),
                "point_size": round(voxel * 1.8, 5),
                "voxel_m": voxel,
                "reference_frames": keys,
                "depth_maps": {str(r): f"depth/{r:04d}.npy" for r in filtered},
                "depth_intrinsics": k_out,
                "completeness": {str(r): float((filtered[r] > 0).mean()) for r in filtered},
            },
        )
        ctx.log.info("fused %d points from %d reference frames", len(world_pts), len(keys))
