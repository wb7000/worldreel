# Job: Phase 3b experiment 1. For each attached synthetic scene: run the classical CPU pipeline
# and its benchmark (eval.benchmark.run), then train 3D Gaussian splats on the GPU from the
# classical camera poses and sparse points, and score both on the same held-out eval views
# (cameras located in the reconstruction by PnP, non-sky pixels). Ground truth is used only for
# scoring. Runs after bootstrap.py (log, save_json, find_scenes, OUT, WORK are defined there).
import os
import subprocess
import sys

for module, package in (
    ("gsplat", "gsplat==1.5.3"),
    ("jsonschema", "jsonschema"),
    ("cv2", "opencv-python-headless"),
):
    try:
        __import__(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
os.environ["WORLDREEL_SIGNING_KEY"] = str(WORK / "tmp_signing_key.pem")  # throwaway key on Kaggle

import json

import cv2
import numpy as np

from eval.benchmark import run as classical_benchmark
from eval.image import psnr, ssim
from eval.localize import localize
from pipeline.io import read_rgb
from stages.camera.classical import OBSERVATIONS, SPARSE_COLORS, SPARSE_POINTS, load_frames
from stages.reconstruct.planesweep import cv_pose
from stages.reconstruct.splat_train import SplatConfig, render_splats, train_splats

FLIP = np.diag([1.0, -1.0, -1.0])
STEPS = int(os.environ.get("WORLDREEL_SPLAT_STEPS", "7000"))


def viewmat(position: list[float], rotation: list[float]) -> np.ndarray:
    r, t = cv_pose(position, rotation)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


results = {}
for scene in find_scenes():
    name = scene.name
    log(f"=== scene {name}")
    gt = json.loads((scene / "gt.json").read_text())
    work = WORK / "bench" / name
    classical = classical_benchmark(scene, work, calibrated=True)
    log(f"classical: {classical['metrics']}")
    item = next(p for p in work.iterdir() if p.is_dir())
    cam = json.loads((item / "camera" / "camera.json").read_text())
    k = cam["intrinsics"]
    kmat = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    frames = load_frames(item / "frames" / "frames")
    views = np.stack([viewmat(p["position"], p["rotation"]) for p in cam["poses"]])
    sparse = np.load(item / "camera" / SPARSE_POINTS).astype(np.float64)
    colors = np.load(item / "camera" / SPARSE_COLORS)
    observations = np.load(item / "camera" / OBSERVATIONS)

    res = train_splats(
        frames,
        kmat,
        views,
        sparse @ FLIP.T,
        colors,
        SplatConfig(steps=STEPS),
        checkpoint_dir=OUT / "checkpoints" / name,
        log=log,
    )
    log(f"splats: {res.stats['gaussians']} gaussians, {res.stats['seconds']} s")

    per_view = []
    for vi, view in enumerate(gt["eval_views"]):
        truth = read_rgb(scene / view["rgb"])
        valid = np.load(scene / view["depth"]) > 0
        dists = [
            np.linalg.norm(np.subtract(f["camera"]["position"], view["camera"]["position"]))
            for f in gt["frames"]
        ]
        near = [int(i) for i in np.argsort(dists)[:3]]
        loc = localize(truth, [frames[i] for i in near], near, sparse, observations, k)
        if loc is None:
            per_view.append({"id": view["id"], "localized": False})
            continue
        img, alpha = render_splats(
            res.params, kmat, viewmat(loc[0], loc[1]), truth.shape[1], truth.shape[0]
        )
        per_view.append(
            {
                "id": view["id"],
                "localized": True,
                "inliers": loc[2],
                "psnr_nonsky": psnr(img, truth, valid),
                "ssim_nonsky": ssim(img, truth, valid),
                "coverage_nonsky": float(((alpha > 0.5) & valid).sum() / max(1, valid.sum())),
            }
        )
        if vi < 2:
            side = np.concatenate([truth, img], 1)
            cv2.imwrite(
                str(OUT / f"{name}_{view['id']}_truth_vs_splat.png"),
                cv2.cvtColor(side, cv2.COLOR_RGB2BGR),
            )
    train_psnr = []
    for i in range(0, len(frames), 5):
        img, _ = render_splats(res.params, kmat, views[i], frames[i].shape[1], frames[i].shape[0])
        train_psnr.append(psnr(img, frames[i]))
    ok = [v for v in per_view if v.get("localized")]
    results[name] = {
        "classical": classical["metrics"],
        "classical_novel_views": classical["novel_views"],
        "splat": {
            "novel_psnr_nonsky": float(np.mean([v["psnr_nonsky"] for v in ok])) if ok else None,
            "novel_ssim_nonsky": float(np.mean([v["ssim_nonsky"] for v in ok])) if ok else None,
            "novel_coverage_nonsky": float(np.mean([v["coverage_nonsky"] for v in ok]))
            if ok
            else None,
            "train_view_psnr_full": float(np.mean(train_psnr)),
            "views": per_view,
            "training": res.stats,
        },
    }
    save_json("results.json", results)
    log(
        f"splat novel views: {results[name]['splat']['novel_psnr_nonsky']} dB / SSIM {results[name]['splat']['novel_ssim_nonsky']}"
    )
log("done")
