"""Reconstruction benchmark on a synthetic scene with ground truth (tools/synth.py).

    python -m eval.benchmark data/synthetic/seed0011 [--calibrated] [--report out.json]

Runs the real pipeline (image-sequence frames → classical camera → plane-sweep reconstruct →
package) on the scene's rendered frames, then measures:
- trajectory: ATE / orientation / RPE after Sim(3) alignment (eval/geometry.py)
- depth: AbsRel, δ<1.25, completeness of the reference-frame depth maps
- novel views: PSNR / SSIM / coverage when rendering the fused points into held-out eval cameras,
  compared with an oracle (ground-truth geometry from the same reference frames and density,
  same renderer). Two protocols: `aligned` (eval camera mapped through the global Sim(3)) and
  `localized` (eval camera located in the reconstruction by PnP, eval/localize.py); the
  thresholds use `localized`, because pose error is already scored by the trajectory metrics
and checks them against eval/thresholds.toml. Exit code 1 if any threshold fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from eval.depth import depth_errors
from eval.geometry import quat_to_matrix, trajectory_errors
from eval.image import psnr, ssim
from eval.localize import localize
from eval.render import render_points
from formats.wreel import PointCloud, validate_wreel
from pipeline.io import (
    CAMERA_JSON,
    FRAMES_DIR,
    POINTS_JSON,
    POINTS_PTS,
    WORLD_WREEL,
    read_json,
    read_rgb,
)
from pipeline.registry import VIDEO_PIPELINE, build_stages
from pipeline.runner import Runner
from pipeline.stage import Item
from stages.camera.classical import OBSERVATIONS, SPARSE_POINTS, load_frames
from stages.safety.hook import PassThroughSafety
from tools.synth import backproject

THRESHOLDS = Path(__file__).with_name("thresholds.toml")
IMPL = {"frames": "images", "camera": "real", "reconstruct": "real", "package": "real"}


def load_thresholds(section: str = "static") -> dict[str, float]:
    with THRESHOLDS.open("rb") as fh:
        return {k: float(v) for k, v in tomllib.load(fh)[section].items()}


def check_thresholds(
    metrics: dict[str, float], limits: dict[str, float]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, limit in limits.items():
        name, kind = key.rsplit("_", 1)
        value = metrics.get(name)
        ok = (
            value is not None
            and np.isfinite(value)
            and (value <= limit if kind == "max" else value >= limit)
        )
        out[name] = {"value": value, kind: limit, "pass": bool(ok)}
    return out


def oracle_cloud(
    scene: Path, gt: dict[str, Any], refs: list[int], k_small: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth points from the reconstruction's reference frames, at its stereo resolution."""
    w, h = int(k_small["width"]), int(k_small["height"])
    pts, cols = [], []
    for r in refs:
        frame = gt["frames"][r]
        depth = cv2.resize(np.load(scene / frame["depth"]), (w, h), interpolation=cv2.INTER_NEAREST)
        rgb = np.asarray(
            cv2.resize(read_rgb(scene / frame["rgb"]), (w, h), interpolation=cv2.INTER_AREA),
            np.uint8,
        )
        pts.append(
            backproject(depth, k_small, frame["camera"]["position"], frame["camera"]["rotation"])
        )
        cols.append(rgb[depth > 0])
    return np.concatenate(pts), np.concatenate(cols).astype(np.uint8)


def run(
    scene: Path,
    work: Path,
    *,
    calibrated: bool,
    stage_config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gt = json.loads((scene / "gt.json").read_text(encoding="utf-8"))
    if not gt.get("eval_views"):
        raise SystemExit("scene has no eval views; render it with --eval-views 4")
    config: dict[str, dict[str, Any]] = {"frames": {"fps": float(gt["fps"])}}
    if calibrated:
        config["camera"] = {"intrinsics": gt["intrinsics"]}
    for name, opts in (stage_config or {}).items():
        config.setdefault(name, {}).update(opts)
    item = Item(
        id=f"bench-seed{gt['seed']:04d}",
        sha256=hashlib.sha256((scene / "gt.json").read_bytes()).hexdigest(),
        path=scene / "rgb",
        source="synthetic benchmark",
        license="CC0-1.0",
        upload_rights="synthetic",
    )
    stages = build_stages(VIDEO_PIPELINE, {name: IMPL.get(name, "mock") for name in VIDEO_PIPELINE})
    result = Runner(work, stages, config=config, safety=PassThroughSafety(work)).run(item)
    if not result.ok:
        failed = {n: r.error for n, r in result.stages.items() if r.status != "done"}
        raise RuntimeError(f"pipeline failed: {failed}")
    out = work / item.id
    timings = {n: read_json(out / n / "_stage.json").get("seconds") for n in result.stages}

    # ---- trajectory
    cam = read_json(out / "camera" / CAMERA_JSON)
    est_pos = np.array([p["position"] for p in cam["poses"]], np.float64)
    est_rot = np.stack([quat_to_matrix(p["rotation"]) for p in cam["poses"]])
    gt_pos = np.array([f["camera"]["position"] for f in gt["frames"]], np.float64)
    gt_rot = np.stack([quat_to_matrix(f["camera"]["rotation"]) for f in gt["frames"]])
    if len(est_pos) != len(gt_pos):
        raise RuntimeError(f"{len(est_pos)} estimated poses for {len(gt_pos)} frames")
    traj, sim = trajectory_errors(est_pos, est_rot, gt_pos, gt_rot)

    # ---- depth (scale estimated depths into metres with the alignment scale)
    pinfo = read_json(out / "reconstruct" / POINTS_JSON)
    abs_rel, delta1, comp, counts = [], [], [], []
    for frame_idx, rel_path in pinfo["depth_maps"].items():
        est = np.load(out / "reconstruct" / rel_path) * sim.scale
        gt_depth = np.load(scene / gt["frames"][int(frame_idx)]["depth"])
        gt_small = cv2.resize(
            gt_depth, (est.shape[1], est.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        d = depth_errors(est, gt_small)
        abs_rel.append(d.abs_rel * d.count)
        delta1.append(d.delta1 * d.count)
        counts.append(d.count)
        comp.append(d.completeness)
    total = max(1, sum(counts))

    # ---- novel views, and the oracle ceiling for the same renderer
    cloud = PointCloud.unpack((out / "reconstruct" / POINTS_PTS).read_bytes())
    world = sim.apply(cloud.positions.astype(np.float64))
    k = gt["intrinsics"]
    # The oracle always uses the TRUE intrinsics, scaled to the reconstruction's stereo size.
    small_w, small_h = (
        int(pinfo["depth_intrinsics"]["width"]),
        int(pinfo["depth_intrinsics"]["height"]),
    )
    sx, sy = small_w / gt["width"], small_h / gt["height"]
    k_true_small = {
        "fx": k["fx"] * sx,
        "fy": k["fy"] * sy,
        "cx": (k["cx"] + 0.5) * sx - 0.5,
        "cy": (k["cy"] + 0.5) * sy - 0.5,
        "width": small_w,
        "height": small_h,
    }
    oracle_pts, oracle_cols = oracle_cloud(
        scene, gt, [int(r) for r in pinfo["reference_frames"]], k_true_small
    )
    sparse = np.load(out / "camera" / SPARSE_POINTS)
    observations = np.load(out / "camera" / OBSERVATIONS)
    train = load_frames(out / "frames" / FRAMES_DIR)
    views = []
    for view in gt["eval_views"]:
        truth = read_rgb(scene / view["rgb"])
        gt_valid = np.load(scene / view["depth"]) > 0
        pose = (view["camera"]["position"], view["camera"]["rotation"])
        image, depth = render_points(world, cloud.colors, k, *pose, radius_px=1)
        o_image, o_depth = render_points(oracle_pts, oracle_cols, k, *pose, radius_px=1)
        covered = (depth > 0) & gt_valid
        o_covered = (o_depth > 0) & gt_valid
        # Localise the eval camera in the reconstruction, matching against the 3 training frames
        # whose true cameras are nearest (ground truth only picks which images to match; the
        # pose comes from the reconstruction alone).
        dists = [
            np.linalg.norm(np.subtract(f["camera"]["position"], pose[0])) for f in gt["frames"]
        ]
        near = [int(i) for i in np.argsort(dists)[:3]]
        loc = localize(
            truth, [train[i] for i in near], near, sparse, observations, cam["intrinsics"]
        )
        if loc is None:
            l_psnr, l_ssim, l_cov, inliers = float("nan"), float("nan"), 0.0, 0
        else:
            l_image, l_depth = render_points(
                cloud.positions, cloud.colors, cam["intrinsics"], loc[0], loc[1], radius_px=1
            )
            l_covered = (l_depth > 0) & gt_valid
            l_psnr, l_ssim = psnr(l_image, truth, l_covered), ssim(l_image, truth, l_covered)
            l_cov, inliers = float(l_covered.sum() / max(1, gt_valid.sum())), loc[2]
        views.append(
            {
                "id": view["id"],
                "psnr": l_psnr,
                "ssim": l_ssim,
                "coverage": l_cov,
                "localization_inliers": inliers,
                "aligned_psnr": psnr(image, truth, covered),
                "aligned_ssim": ssim(image, truth, covered),
                "aligned_coverage": float(covered.sum() / max(1, gt_valid.sum())),
                "oracle_psnr": psnr(o_image, truth, o_covered),
                "oracle_ssim": ssim(o_image, truth, o_covered),
            }
        )
    mean = {key: float(np.mean([v[key] for v in views])) for key in views[0] if key != "id"}

    metrics = {
        "ate_rel": traj.ate_rel,
        "rot_mean_deg": traj.rot_mean_deg,
        "rpe_rot_deg": traj.rpe_rot_deg,
        "depth_abs_rel": sum(abs_rel) / total,
        "depth_delta1": sum(delta1) / total,
        "depth_completeness": float(np.mean(comp)) if comp else 0.0,
        "novel_psnr_gap": mean["oracle_psnr"] - mean["psnr"],
        "novel_ssim_ratio": mean["ssim"] / mean["oracle_ssim"] if mean["oracle_ssim"] > 0 else 0.0,
        "novel_coverage": mean["coverage"],
        # reported, not thresholded:
        "novel_psnr": mean["psnr"],
        "novel_ssim": mean["ssim"],
        "novel_aligned_psnr": mean["aligned_psnr"],
        "novel_aligned_ssim": mean["aligned_ssim"],
        "oracle_psnr": mean["oracle_psnr"],
        "oracle_ssim": mean["oracle_ssim"],
    }
    checks = check_thresholds(metrics, load_thresholds())
    world_report = validate_wreel(out / "package" / WORLD_WREEL)
    return {
        "scene": str(scene),
        "seed": gt["seed"],
        "calibrated": calibrated,
        "intrinsics_source": cam["intrinsics_source"],
        "frames": len(gt["frames"]),
        "resolution": [gt["width"], gt["height"]],
        "trajectory": asdict(traj),
        "metrics": metrics,
        "checks": checks,
        "passed": all(c["pass"] for c in checks.values()),
        "novel_views": views,
        "points": len(cloud),
        "sfm": cam.get("stats", {}),
        "stage_seconds": timings,
        "wreel_valid": world_report.ok,
        "wreel": str(out / "package" / WORLD_WREEL),
    }


def format_report(r: dict[str, Any]) -> str:
    lines = [
        f"Benchmark seed {r['seed']} ({r['frames']} frames, "
        f"{r['resolution'][0]}x{r['resolution'][1]}, intrinsics {r['intrinsics_source']})"
    ]
    for name, c in r["checks"].items():
        bound = f"<= {c['max']}" if "max" in c else f">= {c['min']}"
        lines.append(f"  {'PASS' if c['pass'] else 'FAIL'}  {name:<20} {c['value']:.4f}  ({bound})")
    m = r["metrics"]
    lines.append(
        f"  novel views: localized {m['novel_psnr']:.2f} dB / SSIM {m['novel_ssim']:.3f}; "
        f"aligned {m['novel_aligned_psnr']:.2f} dB / SSIM {m['novel_aligned_ssim']:.3f}; "
        f"oracle {m['oracle_psnr']:.2f} dB / SSIM {m['oracle_ssim']:.3f}"
    )
    lines.append(f"  points {r['points']:,}; .wreel valid: {r['wreel_valid']}")
    lines.append(f"  {'ALL PASSED' if r['passed'] else 'NOT ALL PASSED'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.benchmark")
    parser.add_argument("scene", type=Path)
    parser.add_argument(
        "--calibrated", action="store_true", help="give the camera stage the true intrinsics"
    )
    parser.add_argument("--work", type=Path, default=None, help="keep pipeline outputs here")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.work:
        report = run(args.scene, args.work, calibrated=args.calibrated)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            report = run(args.scene, Path(tmp), calibrated=args.calibrated)
    print(format_report(report))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
