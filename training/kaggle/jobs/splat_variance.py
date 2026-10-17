# Job: run-to-run variance of splat training. For each attached scene: one classical
# reconstruction, then every (variant, training seed) pair is trained and scored on the eval
# views (PnP-localised). The protocol was fixed before results were seen (DECISIONS D-3b.6):
# - views with >= 1000 localisation inliers are "well localised" and are the headline numbers;
# - all views are reported too;
# - mean and spread over the 3 training seeds are reported per variant.
# Runs after bootstrap.py.
import os
import subprocess
import sys

for module, package in (("gsplat", "gsplat==1.5.3"), ("jsonschema", "jsonschema")):
    try:
        __import__(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
os.environ["WORLDREEL_SIGNING_KEY"] = str(WORK / "tmp_signing_key.pem")

import json

import numpy as np

from eval.benchmark import run as classical_benchmark
from eval.image import psnr, ssim
from eval.localize import localize
from pipeline.io import read_rgb
from stages.camera.classical import OBSERVATIONS, SPARSE_COLORS, SPARSE_POINTS, load_frames
from stages.reconstruct.planesweep import cv_pose
from stages.reconstruct.splat_train import SplatConfig, render_splats, train_splats

FLIP = np.diag([1.0, -1.0, -1.0])
WELL_LOCALISED = 1000
SEEDS = (0, 1, 2)
VARIANTS = {
    "no_background": dict(background_points=0),
    "background_small": dict(),
}


def viewmat(position, rotation):
    r, t = cv_pose(position, rotation)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


results: dict = {}
for scene in find_scenes():
    name = scene.name.replace("worldreel-scene-", "")
    gt = json.loads((scene / "gt.json").read_text())
    work = WORK / "bench" / name
    classical = classical_benchmark(scene, work, calibrated=True)
    item = next(p for p in work.iterdir() if p.is_dir())
    cam = json.loads((item / "camera" / "camera.json").read_text())
    k = cam["intrinsics"]
    kmat = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    frames = load_frames(item / "frames" / "frames")
    views = np.stack([viewmat(p["position"], p["rotation"]) for p in cam["poses"]])
    sparse = np.load(item / "camera" / SPARSE_POINTS).astype(np.float64)
    colors = np.load(item / "camera" / SPARSE_COLORS)
    obs = np.load(item / "camera" / OBSERVATIONS)

    eval_views = []
    for view in gt["eval_views"]:
        truth = read_rgb(scene / view["rgb"])
        valid = np.load(scene / view["depth"]) > 0
        d = [
            np.linalg.norm(np.subtract(f["camera"]["position"], view["camera"]["position"]))
            for f in gt["frames"]
        ]
        near = [int(i) for i in np.argsort(d)[:3]]
        loc = localize(truth, [frames[i] for i in near], near, sparse, obs, k)
        eval_views.append((view["id"], truth, valid, loc))
    cv = {v["id"]: v for v in classical["novel_views"]}
    scene_res: dict = {
        "localisation_inliers": {vid: (loc[2] if loc else 0) for vid, _, _, loc in eval_views},
        "classical": {
            vid: {"psnr": cv[vid]["psnr"], "ssim": cv[vid]["ssim"]} for vid, *_ in eval_views
        },
        "runs": {},
    }
    for variant, overrides in VARIANTS.items():
        for seed in SEEDS:
            cfg = SplatConfig(seed=seed, **overrides)
            res = train_splats(
                frames, kmat, views, sparse @ FLIP.T, colors, cfg, log=lambda m: None
            )
            per_view = {}
            for vid, truth, valid, loc in eval_views:
                if loc is None:
                    continue
                img, _ = render_splats(
                    res.params, kmat, viewmat(loc[0], loc[1]), truth.shape[1], truth.shape[0]
                )
                per_view[vid] = {"psnr": psnr(img, truth, valid), "ssim": ssim(img, truth, valid)}
            scene_res["runs"][f"{variant}/seed{seed}"] = {
                "views": per_view,
                "gaussians": res.stats["gaussians"],
                "seconds": res.stats["seconds"],
                "extent": res.stats["scene_extent"],
            }
            good = [
                v
                for vid, v in per_view.items()
                if scene_res["localisation_inliers"][vid] >= WELL_LOCALISED
            ]
            log(
                f"{name} {variant} seed{seed}: well-localised mean {np.mean([g['psnr'] for g in good]):.2f} dB "
                f"/ {np.mean([g['ssim'] for g in good]):.3f} ({len(good)} views), {res.stats['seconds']} s"
            )
            results[name] = scene_res
            save_json("variance.json", results)
log("done")
