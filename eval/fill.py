"""Measure the plane-completion fill (algorithm A-3) on a synthetic splat world.

Observed geometry = Gaussian centres labelled `observed` by A-1. Ground truth = surface seen from
the held-out eval views (GT depth + RGB), aligned by the camera-trajectory Sim(3).
r = 2 % of the scene extent.

  coverage_observed / coverage_with_fill   share of eval-view GT points within r of geometry
  fill_precision                           share of fill elements within r of the GT surface
  fill_colour_error / flat_colour_error    mean |RGB| of fill vs the mean GT colour within r of
                                           it (a fill element stands for a cell, so it is
                                           compared with the local average, not one textured
                                           pixel), and the same for a flat mean-colour fill
  tier1_s / tier2_s, tier1_complete        time per tier, whether tier 1 met its budget

Usage: python -m eval.fill <splat export dir> <scene dir> [--budget 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from eval.geometry import quat_to_matrix, trajectory_errors
from formats.wreel import read_wreel
from pipeline.io import read_rgb
from stages.fill.planes import drop_seen_through, progressive_fill
from stages.package.confidence import OBSERVED, relabel_gaussians, views_from_camera_json
from stages.package.gaussians import BACKGROUND_CONFIDENCE, splats_to_gaussians
from tools.synth import backproject


def eval_surface(scene: Path, gt: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pts, cols = [], []
    for view in gt.get("eval_views", []):
        depth = np.load(scene / view["depth"])
        rgb = read_rgb(scene / view["rgb"])
        p = backproject(
            depth, gt["intrinsics"], view["camera"]["position"], view["camera"]["rotation"]
        )
        pts.append(p)
        cols.append(rgb.reshape(-1, 3)[depth.reshape(-1) > 0])
    return np.concatenate(pts), np.concatenate(cols).astype(np.float64)


def exact_surface_distance(
    points: np.ndarray, objects: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """(distance to the nearest true surface, inside-a-box flag) from the generator's exact
    objects (gt.json "objects", generator >= 0.2.0): a ground plane and axis-aligned boxes."""
    dist = np.full(len(points), np.inf)
    inside = np.zeros(len(points), bool)
    for obj in objects:
        if obj["kind"] == "plane":
            n = np.asarray(obj["normal"], np.float64)
            d = np.abs(points @ n - obj["offset"])
            within = np.all(np.abs(points[:, [0, 2]]) <= obj["half_size"], axis=1)
            dist = np.where(within, np.minimum(dist, d), dist)
        elif obj["kind"] == "box":
            q = np.abs(points - np.asarray(obj["centre"])) - np.asarray(obj["size"]) / 2
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inner = q.max(axis=1)
            is_in = inner < 0
            dist = np.minimum(dist, np.where(is_in, -inner, outside))
            inside |= is_in
    return dist, inside


def colour_error(a: np.ndarray, b: np.ndarray) -> float | None:
    return round(float(np.abs(a.astype(np.float64) - b).mean()), 1) if len(a) else None


def run(
    export: Path, scene: Path, budget_s: float = 5.0, colour_mode: str = "plane_mean"
) -> dict[str, Any]:
    gt = json.loads((scene / "gt.json").read_text(encoding="utf-8"))
    meta = json.loads((export / "meta.json").read_text(encoding="utf-8"))
    with np.load(export / "splats.npz") as npz:
        params = {k: npz[k] for k in npz.files}
    beyond = 0.5 * meta["background_radius"] * meta["extent"]
    g = splats_to_gaussians(
        params, centre_cv=np.asarray(meta["centre_cv"]), background_beyond=beyond
    )
    template = read_wreel(export / "template.wreel")
    keys = template.manifest["cameras"][0]["keyframes"]
    cam = {
        "intrinsics": gt["intrinsics"],
        "poses": [{"position": k["position"], "rotation": k["rotation"]} for k in keys],
    }
    layer = template.manifest["scene"]["layers"][0]
    mvs = template.points(layer["lods"][-1]["path"]).positions.astype(np.float64)
    fg = (g.confidence != np.float32(BACKGROUND_CONFIDENCE)) & (g.opacity >= 0.05)
    cls, _, _ = relabel_gaussians(
        g.positions.astype(np.float64),
        g.opacity,
        fg,
        views_from_camera_json(cam),
        mvs,
        float(meta["extent"]),
    )
    obs_pos = g.positions[fg][cls == OBSERVED].astype(np.float64)
    obs_col = g.colors[fg][cls == OBSERVED]
    obs_w = g.opacity[fg][cls == OBSERVED].astype(np.float64)
    _, tier1, tier2 = progressive_fill(
        obs_pos,
        obs_col,
        float(meta["extent"]),
        budget_s=budget_s,
        weights=obs_w,
        colour_mode=colour_mode,
    )
    views = views_from_camera_json(cam)
    seen_through = [drop_seen_through(t, views, obs_pos) for t in (tier1, tier2)]

    est_pos = np.array([k["position"] for k in keys])
    est_rot = np.stack([quat_to_matrix(k["rotation"]) for k in keys])
    gt_pos = np.array([f["camera"]["position"] for f in gt["frames"]])
    gt_rot = np.stack([quat_to_matrix(f["camera"]["rotation"]) for f in gt["frames"]])
    _, sim = trajectory_errors(est_pos, est_rot, gt_pos, gt_rot)
    r = 0.02 * meta["extent"] * sim.scale
    surf, _ = eval_surface(scene, gt)
    # Full GT surface (all frames + eval views) with colours, for precision and colour error.
    all_pts, all_col = [], []
    for item in gt["frames"] + gt.get("eval_views", []):
        depth = np.load(scene / item["depth"])[::2, ::2]
        k = dict(gt["intrinsics"])
        k.update(fx=k["fx"] / 2, fy=k["fy"] / 2, cx=k["cx"] / 2, cy=k["cy"] / 2)
        all_pts.append(
            backproject(depth, k, item["camera"]["position"], item["camera"]["rotation"])
        )
        all_col.append(
            read_rgb(scene / item["rgb"])[::2, ::2].reshape(-1, 3)[depth.reshape(-1) > 0]
        )
    full_tree = cKDTree(np.concatenate(all_pts))
    full_col = np.concatenate(all_col).astype(np.float64)
    obs_tree = cKDTree(sim.apply(obs_pos))
    covered_obs = obs_tree.query(surf, distance_upper_bound=r, workers=-1)[0] < np.inf
    report: dict[str, Any] = {
        "observed_gaussians": len(obs_pos),
        "eval_points": len(surf),
        "radius_m": round(r, 3),
        "coverage_observed": round(float(covered_obs.mean()), 3),
        "dropped_seen_through": seen_through,
    }
    flat = obs_col.astype(np.float64).mean(0)
    for name, tier in (("tier1", tier1), ("tier2", tier2)):
        fill_w = sim.apply(tier.positions)
        if not len(fill_w):
            report[name] = {"elements": 0, "seconds": round(tier.seconds, 2)}
            continue
        dist_raw, _ = full_tree.query(fill_w, workers=-1)
        dist = np.asarray(dist_raw, np.float64)
        on = dist <= r
        local_mean = np.zeros((len(fill_w), 3))
        for j in np.nonzero(on)[0]:
            local_mean[j] = full_col[full_tree.query_ball_point(fill_w[j], r)].mean(0)
        covered_fill = cKDTree(fill_w).query(surf, distance_upper_bound=r, workers=-1)[0] < np.inf
        exact: dict[str, Any] = {}
        if gt.get("objects"):
            d_exact, in_box = exact_surface_distance(fill_w, gt["objects"])
            exact = {
                "fill_precision_exact": round(float((d_exact <= r).mean()), 3),
                "fill_inside_objects": round(float((in_box & (d_exact > r)).mean()), 3),
            }
        report[name] = {
            **exact,
            "elements": len(tier.positions),
            "cell_m": round(tier.cell * sim.scale, 3),
            "seconds": round(tier.seconds, 2),
            "complete": tier.complete,
            "coverage_with_fill": round(float((covered_obs | covered_fill).mean()), 3),
            "fill_precision": round(float(on.mean()), 3),
            "fill_colour_error": colour_error(tier.colors[on], local_mean[on]),
            "flat_colour_error": colour_error(np.tile(flat, (int(on.sum()), 1)), local_mean[on]),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", type=Path)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--colour", choices=["diffuse", "plane_mean"], default="plane_mean")
    a = ap.parse_args(argv)
    print(json.dumps(run(a.export, a.scene, a.budget, a.colour), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
