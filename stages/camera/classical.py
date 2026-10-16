"""Real `camera` stage (classical, CPU): incremental SfM over the extracted frames.

Outputs (in WorldReel conventions, spec/wreel.md §2):
    camera.json        intrinsics, per-frame poses, how intrinsics and scale were chosen
    sparse_points.npy  (M, 3) float32 triangulated points
    sparse_colors.npy  (M, 3) uint8 RGB
    observations.npy   (K, 4) float32: frame index, point index, u, v (frame pixels)

Limits (recorded in camera.json): monocular SfM has no absolute scale and no gravity. The world
is the initial camera's frame (its up is world +Y) and the scale is set so the median depth
of the sparse points seen from the first registered camera equals `median_depth_m` (an
assumption, default 5 m).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from eval.geometry import matrix_to_quat
from pipeline.io import (
    CAMERA_JSON,
    FRAMES_DIR,
    FRAMES_JSON,
    read_json,
    read_rgb,
    save_npy,
    write_json,
)
from pipeline.stage import Impl, StageContext, StageError
from stages.camera.sfm import SfmConfig, SfmError, run_sfm

F64 = npt.NDArray[np.float64]
SPARSE_POINTS = "sparse_points.npy"
SPARSE_COLORS = "sparse_colors.npy"
OBSERVATIONS = "observations.npy"

# OpenCV camera/world axes (x right, y down, z forward) → WorldReel (x right, y up, -z forward).
_FLIP = np.diag([1.0, -1.0, -1.0])

DEFAULTS: dict[str, Any] = {
    "hfov_deg": 60.0,  # assumed horizontal field of view when no intrinsics are given
    "median_depth_m": 5.0,
    "min_registered_fraction": 0.6,
}


def load_frames(frames_dir: Path) -> list[npt.NDArray[np.uint8]]:
    paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        raise StageError("no PNG frames found; the camera stage needs the real frames stage")
    return [read_rgb(p) for p in paths]


def intrinsics_for(width: int, height: int, cfg: dict[str, Any]) -> tuple[dict[str, float], str]:
    given = cfg.get("intrinsics")
    if given:
        sx = width / float(given["width"])
        sy = height / float(given["height"])
        k = {
            "fx": given["fx"] * sx,
            "fy": given["fy"] * sy,
            "cx": (given["cx"] + 0.5) * sx - 0.5,
            "cy": (given["cy"] + 0.5) * sy - 0.5,
            "width": width,
            "height": height,
        }
        return k, "given"
    fx = (width / 2) / math.tan(math.radians(float(cfg["hfov_deg"])) / 2)
    k = {
        "fx": fx,
        "fy": fx,
        "cx": width / 2 - 0.5,
        "cy": height / 2 - 0.5,
        "width": width,
        "height": height,
    }
    return k, f"assumed horizontal FOV {cfg['hfov_deg']} deg"


def _fill_missing(
    rot: list[F64 | None], trans: list[F64 | None]
) -> tuple[list[F64], list[F64], list[bool]]:
    """Give unregistered frames the pose of the nearest registered frame (and say which)."""
    registered = [r is not None and t is not None for r, t in zip(rot, trans, strict=True)]
    idx = [i for i, ok in enumerate(registered) if ok]
    out_r: list[F64] = []
    out_t: list[F64] = []
    for i in range(len(rot)):
        j = i if registered[i] else min(idx, key=lambda j: abs(j - i))
        r, t = rot[j], trans[j]
        assert r is not None and t is not None
        out_r.append(r)
        out_t.append(t)
    return out_r, out_t, registered


class ClassicalCameraStage:
    name: ClassVar[str] = "camera"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "real"
    depends_on: ClassVar[tuple[str, ...]] = ("frames",)

    def run(self, ctx: StageContext) -> None:
        cfg = {**DEFAULTS, **ctx.config}
        meta = read_json(ctx.inputs["frames"] / FRAMES_JSON)
        images = load_frames(ctx.inputs["frames"] / FRAMES_DIR)
        h, w = images[0].shape[:2]
        k, k_source = intrinsics_for(w, h, cfg)
        kmat = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
        sfm_cfg = SfmConfig(**{f: cfg[f] for f in SfmConfig.__dataclass_fields__ if f in cfg})
        try:
            res = run_sfm(images, kmat, sfm_cfg)
        except SfmError as exc:
            raise StageError(f"structure-from-motion failed: {exc}") from exc
        n = len(images)
        if not any(r is not None for r in res.rotations):
            raise StageError("no frame could be registered")
        rotations, translations, registered = _fill_missing(res.rotations, res.translations)
        if sum(registered) < cfg["min_registered_fraction"] * n:
            raise StageError(f"only {sum(registered)} of {n} frames could be registered")

        # Scale: median depth of the sparse points in the first camera.
        first = registered.index(True)
        r0, t0 = rotations[first], translations[first]
        depth0 = (res.points @ r0.T + t0)[:, 2]
        median = float(np.median(depth0[depth0 > 0])) if np.any(depth0 > 0) else 1.0
        scale = float(cfg["median_depth_m"]) / median

        poses = []
        fps = float(meta["fps"])
        for i in range(n):
            r_wc = rotations[i].T
            center = -r_wc @ translations[i]
            poses.append(
                {
                    "t": round(i / fps, 6),
                    "position": [float(c) for c in _FLIP @ center * scale],
                    "rotation": matrix_to_quat(_FLIP @ r_wc @ _FLIP),
                    "registered": registered[i],
                }
            )
        points = (res.points @ _FLIP.T) * scale
        save_npy(ctx.out / SPARSE_POINTS, points.astype(np.float32))
        save_npy(ctx.out / SPARSE_COLORS, res.colors.astype(np.uint8))
        save_npy(ctx.out / OBSERVATIONS, res.observations.astype(np.float32))
        write_json(
            ctx.out / CAMERA_JSON,
            {
                "method": "classical-sfm (SIFT, essential matrix, PnP, bundle adjustment)",
                "intrinsics": k,
                "intrinsics_source": k_source,
                "fps": fps,
                "scale_m_per_unit": scale,
                "scale_source": (
                    f"assumed median depth {cfg['median_depth_m']} m in the first registered frame"
                ),
                "up_source": "first camera's up direction (no gravity estimate)",
                "poses": poses,
                "stats": res.stats,
            },
        )
        ctx.log.info("registered %d/%d frames, %d points", sum(registered), n, len(points))
