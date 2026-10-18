"""Measure reference-image verification (algorithm A-4) on synthetic scenes, offline.

Positives: the held-out eval views of the scene itself, used as owner-provided reference photos.
Negatives: eval views of other scenes (same generator, similar look: hard negatives).
For accepted positives: camera position / rotation error vs ground truth (after the trajectory
Sim(3)), and the colour of the scene's fill elements (A-3) taken from the photo vs the plane-mean
baseline, both against the local mean ground-truth colour (as in eval.fill).

Usage: python -m eval.retrieve <splat export> <scene> --negatives <scene> [<scene> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from eval.fill import colour_error
from eval.geometry import quat_to_matrix, rotation_angle_deg, trajectory_errors
from formats.wreel import read_wreel
from pipeline.io import read_rgb
from stages.fill.planes import progressive_fill
from stages.fill.retrieve import FrameView, colours_from, verify
from stages.package.confidence import OBSERVED, relabel_gaussians, views_from_camera_json
from stages.package.gaussians import BACKGROUND_CONFIDENCE, splats_to_gaussians
from tools.synth import backproject

_FLIP = np.diag([1.0, -1.0, -1.0])


def run(export: Path, scene: Path, negatives: list[Path]) -> dict[str, Any]:
    gt = json.loads((scene / "gt.json").read_text(encoding="utf-8"))
    meta = json.loads((export / "meta.json").read_text(encoding="utf-8"))
    with np.load(export / "splats.npz") as npz:
        params = {k: npz[k] for k in npz.files}
    ext = float(meta["extent"])
    g = splats_to_gaussians(
        params,
        centre_cv=np.asarray(meta["centre_cv"]),
        background_beyond=0.5 * meta["background_radius"] * ext,
    )
    template = read_wreel(export / "template.wreel")
    keys = template.manifest["cameras"][0]["keyframes"]
    cam = {
        "intrinsics": gt["intrinsics"],
        "poses": [{"position": k["position"], "rotation": k["rotation"]} for k in keys],
    }
    views = views_from_camera_json(cam)
    mvs = template.points(
        template.manifest["scene"]["layers"][0]["lods"][-1]["path"]
    ).positions.astype(np.float64)
    fg = (g.confidence != np.float32(BACKGROUND_CONFIDENCE)) & (g.opacity >= 0.05)
    cls, _, _ = relabel_gaussians(g.positions.astype(np.float64), g.opacity, fg, views, mvs, ext)
    obs = g.positions[fg][cls == OBSERVED].astype(np.float64)
    obs_col = g.colors[fg][cls == OBSERVED]
    frames = [
        FrameView(read_rgb(scene / f["rgb"]), v.r, v.t, v.k)
        for f, v in zip(gt["frames"], views, strict=True)
    ]
    est_pos = np.array([k["position"] for k in keys])
    est_rot = np.stack([quat_to_matrix(k["rotation"]) for k in keys])
    gt_pos = np.array([f["camera"]["position"] for f in gt["frames"]])
    gt_rot = np.stack([quat_to_matrix(f["camera"]["rotation"]) for f in gt["frames"]])
    traj, sim = trajectory_errors(est_pos, est_rot, gt_pos, gt_rot)

    _, _, fill = progressive_fill(
        obs, obs_col, ext, weights=g.opacity[fg][cls == OBSERVED].astype(np.float64)
    )
    report: dict[str, Any] = {
        "scene": scene.name,
        "camera_rot_deg": round(traj.rot_mean_deg, 2),
        "positives": [],
        "negatives": [],
    }
    recoloured = np.zeros(len(fill.positions), bool)
    colours = fill.colors.copy()
    for view in gt.get("eval_views", []):
        img = read_rgb(scene / view["rgb"])
        v = verify(img, frames, obs, observed_colors=obs_col)
        entry: dict[str, Any] = {
            "id": view["id"],
            "accepted": v.ok,
            "reason": v.reason,
            "inliers": v.inliers,
            "median_px": round(v.median_px, 2),
            "second": v.second_frame_inliers,
            "off_plane": round(v.off_plane_frac, 3),
            "photo_agree": round(v.photo_agree, 3),
        }
        if v.ok:
            # Estimated camera (OpenCV world-to-camera, OpenCV axes) -> WorldReel pose -> GT frame.
            c_wr = _FLIP @ (-v.r.T @ v.t)
            r_wr = _FLIP @ v.r.T @ _FLIP
            pos_gt = sim.apply(c_wr[None])[0]
            rot_gt = sim.apply_rotation(r_wr[None])[0]
            true_r = quat_to_matrix(view["camera"]["rotation"])
            entry["position_error_m"] = round(
                float(np.linalg.norm(pos_gt - np.asarray(view["camera"]["position"]))), 3
            )
            entry["rotation_error_deg"] = round(rotation_angle_deg(true_r.T @ rot_gt), 2)
            vis, col = colours_from(img, v, fill.positions, obs, footprint=fill.cell)
            new = vis & ~recoloured
            colours[new] = col[new]
            recoloured |= vis
        report["positives"].append(entry)
    for neg in negatives:
        ngt = json.loads((neg / "gt.json").read_text(encoding="utf-8"))
        for view in ngt.get("eval_views", []):
            v = verify(read_rgb(neg / view["rgb"]), frames, obs, observed_colors=obs_col)
            report["negatives"].append(
                {
                    "scene": neg.name,
                    "id": view["id"],
                    "accepted": v.ok,
                    "reason": v.reason,
                    "inliers": v.inliers,
                    "off_plane": round(v.off_plane_frac, 3),
                    "photo_agree": round(v.photo_agree, 3),
                }
            )

    # Colour check on the recoloured fill elements: local mean GT colour within r (as eval.fill).
    r = 0.02 * ext * sim.scale
    pts, cols = [], []
    for item in gt["frames"] + gt.get("eval_views", []):
        depth = np.load(scene / item["depth"])[::2, ::2]
        k = dict(gt["intrinsics"])
        k.update(fx=k["fx"] / 2, fy=k["fy"] / 2, cx=k["cx"] / 2, cy=k["cy"] / 2)
        pts.append(backproject(depth, k, item["camera"]["position"], item["camera"]["rotation"]))
        cols.append(read_rgb(scene / item["rgb"])[::2, ::2].reshape(-1, 3)[depth.reshape(-1) > 0])
    tree = cKDTree(np.concatenate(pts))
    full_col = np.concatenate(cols).astype(np.float64)
    fill_w = sim.apply(fill.positions)
    use = np.nonzero(recoloured)[0]
    local = []
    keep = []
    for j in use:
        nb = tree.query_ball_point(fill_w[j], r)
        if nb:
            local.append(full_col[nb].mean(0))
            keep.append(j)
    keep_a = np.array(keep, np.int64)
    local_a = np.array(local) if local else np.zeros((0, 3))
    report["fill_elements"] = len(fill.positions)
    report["recoloured"] = int(recoloured.sum())
    report["recoloured_on_surface"] = len(keep_a)
    report["colour_error_retrieved"] = colour_error(colours[keep_a], local_a)
    report["colour_error_plane_mean"] = colour_error(fill.colors[keep_a], local_a)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", type=Path)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--negatives", type=Path, nargs="*", default=[])
    a = ap.parse_args(argv)
    print(json.dumps(run(a.export, a.scene, a.negatives), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
