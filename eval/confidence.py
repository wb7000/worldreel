"""Fill-honesty / confidence calibration for splat worlds against synthetic ground truth.

Per foreground Gaussian (opacity >= 0.05): error = distance from its centre to the nearest
ground-truth surface point (GT depth of all frames + eval views, back-projected), after the
camera-trajectory Sim(3) alignment. A Gaussian is "bad" when error > `bad_frac` x scene extent.

Reported per labelling (baseline constant vs algorithm A-1 variants):
  * observed_fraction       share of Gaussians labelled observed
  * honesty_recall_bad      of bad Gaussians, share NOT labelled observed   (higher = more honest)
  * precision_observed      of observed Gaussians, share that are good      (higher = trustworthy)
  * spearman                rank correlation of confidence vs error         (negative = calibrated)
  * error_by_decile         mean error (m) per confidence decile, low -> high

Usage: python -m eval.confidence <splat export dir> <scene dir>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from eval.geometry import quat_to_matrix, trajectory_errors
from formats.wreel import PROVENANCE_CLASSES, read_wreel
from stages.package.confidence import (
    SupportConfig,
    classify,
    far_from_points,
    free_space_floaters,
    stereo_evidence,
    view_support,
    views_from_camera_json,
)
from stages.package.gaussians import BACKGROUND_CONFIDENCE, splats_to_gaussians
from tools.synth import backproject

F64 = npt.NDArray[np.float64]
OBSERVED = PROVENANCE_CLASSES.index("observed")


def gt_surface(scene: Path, stride: int = 2) -> F64:
    gt = json.loads((scene / "gt.json").read_text(encoding="utf-8"))
    pts = []
    for item in gt["frames"] + gt.get("eval_views", []):
        depth = np.load(scene / item["depth"])[::stride, ::stride]
        k = dict(gt["intrinsics"])
        k.update(fx=k["fx"] / stride, fy=k["fy"] / stride, cx=k["cx"] / stride, cy=k["cy"] / stride)
        p = backproject(depth, k, item["camera"]["position"], item["camera"]["rotation"])
        pts.append(p)  # backproject already drops pixels without depth
    return np.concatenate(pts)


def score(
    error: F64, cls: npt.NDArray[np.uint8], conf: npt.NDArray[np.float32], bad_m: float
) -> dict[str, Any]:
    bad = error > bad_m
    observed = cls == OBSERVED
    rho = float(spearmanr(conf, error).statistic) if np.ptp(conf) > 0 else float("nan")
    order = np.argsort(conf, kind="stable")
    deciles = [float(error[chunk].mean()) for chunk in np.array_split(order, 10)]
    return {
        "observed_fraction": round(float(observed.mean()), 3),
        "honesty_recall_bad": round(float((~observed[bad]).mean()) if bad.any() else 1.0, 3),
        "precision_observed": round(float((~bad[observed]).mean()) if observed.any() else 1.0, 3),
        "spearman": round(rho, 3),
        "error_by_decile": [round(d, 3) for d in deciles],
    }


def run(export: Path, scene: Path, bad_frac: float = 0.05) -> dict[str, Any]:
    gt = json.loads((scene / "gt.json").read_text(encoding="utf-8"))
    meta = json.loads((export / "meta.json").read_text(encoding="utf-8"))
    with np.load(export / "splats.npz") as npz:
        params = {k: npz[k] for k in npz.files}
    beyond = 0.5 * meta["background_radius"] * meta["extent"]
    g = splats_to_gaussians(
        params, centre_cv=np.asarray(meta["centre_cv"]), background_beyond=beyond
    )
    keep = (g.confidence != np.float32(BACKGROUND_CONFIDENCE)) & (g.opacity >= 0.05)
    pos = g.positions[keep].astype(np.float64)
    opacity = g.opacity[keep]

    template = read_wreel(export / "template.wreel")
    keys = template.manifest["cameras"][0]["keyframes"]
    cam = {
        "intrinsics": gt["intrinsics"],
        "poses": [{"position": k["position"], "rotation": k["rotation"]} for k in keys],
    }
    est_pos = np.array([k["position"] for k in keys])
    est_rot = np.stack([quat_to_matrix(k["rotation"]) for k in keys])
    gt_pos = np.array([f["camera"]["position"] for f in gt["frames"]])
    gt_rot = np.stack([quat_to_matrix(f["camera"]["rotation"]) for f in gt["frames"]])
    _, sim = trajectory_errors(est_pos, est_rot, gt_pos, gt_rot)
    dist, _ = cKDTree(gt_surface(scene)).query(sim.apply(pos), workers=-1)
    error: F64 = np.asarray(dist, np.float64)
    bad_m = bad_frac * meta["extent"] * sim.scale

    views = views_from_camera_json(cam)
    base_cls = np.full(len(pos), OBSERVED, np.uint8)
    base_conf = np.full(len(pos), 0.9, np.float32)
    report: dict[str, Any] = {
        "gaussians": len(pos),
        "views": len(views),
        "bad_threshold_m": round(bad_m, 3),
        "bad_fraction": round(float((error > bad_m).mean()), 3),
        "baseline_constant": score(error, base_cls, base_conf, bad_m),
    }
    cfg = SupportConfig()
    n, spread = view_support(pos, opacity, views, cfg)
    for name, variant in (
        ("a1_count_only", replace(cfg, use_spread=False)),
        ("a1_count_spread", cfg),
    ):
        cls, conf = classify(n, spread, variant)
        report[name] = score(error, cls, conf, bad_m)
    layer = template.manifest["scene"]["layers"][0]
    mvs = template.points(layer["lods"][-1]["path"]).positions.astype(np.float64)
    for frac in (0.01, 0.02, 0.04):
        far = far_from_points(pos, mvs, frac * meta["extent"])
        cls, conf = classify(n, spread, cfg, far_from_mvs=far)
        report[f"a1_spread_mvs{frac}"] = score(error, cls, conf, bad_m) | {
            "far_fraction": round(float(far.mean()), 3)
        }
    covered, in_front = stereo_evidence(pos, views, mvs)
    floaters = free_space_floaters(covered, in_front)
    far = far_from_points(pos, mvs, 0.02 * meta["extent"])
    for name, flag in (
        ("a1_spread_freespace", floaters),
        ("a1_spread_mvs0.02_covered", far & (covered >= 3)),
    ):
        cls, conf = classify(n, spread, cfg, far_from_mvs=flag)
        report[name] = score(error, cls, conf, bad_m) | {
            "far_fraction": round(float(flag.mean()), 3)
        }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", type=Path)
    ap.add_argument("scene", type=Path)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.export, a.scene), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
