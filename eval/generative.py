"""Compare plane-mean fill colour (A-3) with LaMa-generated texture (A-5) on a synthetic scene.

For refined-tier fill elements within r (2 % of extent) of the true surface:
  local_error    |colour - mean GT colour within r|     (accuracy at cell level)
  pixel_error    |colour - nearest GT pixel colour|      (accuracy at texture level)
  spread_ratio   mean colour spread of fill neighbours / mean spread of GT within r
                 (0 = flat; 1 = as much texture as the real surface)

Usage: python -m eval.generative <splat export> <scene>
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
from stages.fill.generative import generative_colours, lama_inpainter
from stages.fill.planes import drop_seen_through, progressive_fill
from stages.package.confidence import OBSERVED, relabel_gaussians, views_from_camera_json
from stages.package.gaussians import BACKGROUND_CONFIDENCE, splats_to_gaussians
from tools.synth import backproject


def run(export: Path, scene: Path) -> dict[str, Any]:
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
    sel = cls == OBSERVED
    pos, col = g.positions[fg][sel].astype(np.float64), g.colors[fg][sel]
    wts = g.opacity[fg][sel].astype(np.float64)
    planes, _, tier = progressive_fill(pos, col, ext, weights=wts)
    drop_seen_through(tier, views, pos)
    gen, seconds = generative_colours(planes, pos, col, wts, tier, lama_inpainter())

    est_pos = np.array([k["position"] for k in keys])
    est_rot = np.stack([quat_to_matrix(k["rotation"]) for k in keys])
    gt_pos = np.array([f["camera"]["position"] for f in gt["frames"]])
    gt_rot = np.stack([quat_to_matrix(f["camera"]["rotation"]) for f in gt["frames"]])
    _, sim = trajectory_errors(est_pos, est_rot, gt_pos, gt_rot)
    r = 0.02 * ext * sim.scale
    pts, cols = [], []
    for item in gt["frames"] + gt.get("eval_views", []):
        depth = np.load(scene / item["depth"])[::2, ::2]
        k = dict(gt["intrinsics"])
        k.update(fx=k["fx"] / 2, fy=k["fy"] / 2, cx=k["cx"] / 2, cy=k["cy"] / 2)
        pts.append(backproject(depth, k, item["camera"]["position"], item["camera"]["rotation"]))
        cols.append(read_rgb(scene / item["rgb"])[::2, ::2].reshape(-1, 3)[depth.reshape(-1) > 0])
    tree = cKDTree(np.concatenate(pts))
    gcol = np.concatenate(cols).astype(np.float64)
    fw = sim.apply(tier.positions)
    dist, nearest = tree.query(fw, workers=-1)
    on = np.nonzero(np.asarray(dist) <= r)[0]
    local_mean = np.zeros((len(fw), 3))
    local_std = np.zeros(len(fw))
    for j in on:
        nb = tree.query_ball_point(fw[j], r)
        local_mean[j] = gcol[nb].mean(0)
        local_std[j] = gcol[nb].std(0).mean()
    fill_tree = cKDTree(fw)
    report: dict[str, Any] = {
        "scene": scene.name,
        "elements": len(fw),
        "on_surface": len(on),
        "lama_seconds": round(seconds, 1),
    }
    near = np.asarray(nearest)
    for name, c in (("plane_mean", tier.colors), ("generated", gen)):
        cf = c.astype(np.float64)
        spread = np.array([cf[fill_tree.query_ball_point(fw[j], r)].std(0).mean() for j in on])
        report[name] = {
            "local_error": round(float(np.abs(cf[on] - local_mean[on]).mean()), 1),
            "pixel_error": round(float(np.abs(cf[on] - gcol[near[on]]).mean()), 1),
            "spread_ratio": round(float(spread.mean() / max(local_std[on].mean(), 1e-9)), 2),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", type=Path)
    ap.add_argument("scene", type=Path)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.export, a.scene)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
