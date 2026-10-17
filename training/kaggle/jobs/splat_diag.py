# Job: diagnose the fog on seed 13's novel views. Trains three splat variants from the same
# classical reconstruction and renders the first eval view with colour, alpha and expected depth
# for each, plus statistics on where semi-transparent Gaussians sit. Runs after bootstrap.py.
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

import cv2
import numpy as np
import torch
from gsplat import rasterization

from eval.benchmark import run as classical_benchmark
from eval.image import psnr
from eval.localize import localize
from pipeline.io import read_rgb
from stages.camera.classical import OBSERVATIONS, SPARSE_COLORS, SPARSE_POINTS, load_frames
from stages.reconstruct.planesweep import cv_pose
from stages.reconstruct.splat_train import SplatConfig, train_splats

FLIP = np.diag([1.0, -1.0, -1.0])
VARIANTS = {
    "no_background": SplatConfig(background_points=0, prune_scale3d=0.1),
    "bg20k_prune1.0": SplatConfig(
        background_points=20000, prune_scale3d=1.0, background_max_scale=1e9
    ),
    "bg100k_small": SplatConfig(),
}


def viewmat(position, rotation):
    r, t = cv_pose(position, rotation)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


scene = find_scenes()[0]
gt = json.loads((scene / "gt.json").read_text())
work = WORK / "bench" / scene.name
classical_benchmark(scene, work, calibrated=True)
item = next(p for p in work.iterdir() if p.is_dir())
cam = json.loads((item / "camera" / "camera.json").read_text())
k = cam["intrinsics"]
kmat = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
frames = load_frames(item / "frames" / "frames")
views = np.stack([viewmat(p["position"], p["rotation"]) for p in cam["poses"]])
sparse = np.load(item / "camera" / SPARSE_POINTS).astype(np.float64)
colors = np.load(item / "camera" / SPARSE_COLORS)
obs = np.load(item / "camera" / OBSERVATIONS)
centers = -np.einsum("nji,nj->ni", views[:, :3, :3], views[:, :3, 3])
sparse_cv = sparse @ FLIP.T

view = gt["eval_views"][0]
truth = read_rgb(scene / view["rgb"])
valid = np.load(scene / view["depth"]) > 0
dists = [
    np.linalg.norm(np.subtract(f["camera"]["position"], view["camera"]["position"]))
    for f in gt["frames"]
]
near = [int(i) for i in np.argsort(dists)[:3]]
loc = localize(truth, [frames[i] for i in near], near, sparse, obs, k)
vm_eval = viewmat(loc[0], loc[1])
eval_center = -vm_eval[:3, :3].T @ vm_eval[:3, 3]
sparse_depth_eval = (sparse_cv @ vm_eval[:3, :3].T + vm_eval[:3, 3])[:, 2]
report = {
    "localization_inliers": loc[2],
    "eval_center": eval_center.tolist(),
    "train_centers_min": centers.min(0).tolist(),
    "train_centers_max": centers.max(0).tolist(),
    "eval_dist_to_nearest_train_center": float(np.linalg.norm(centers - eval_center, axis=1).min()),
    "sparse_depth_in_eval_view_p5_p50_p95": np.percentile(
        sparse_depth_eval[sparse_depth_eval > 0], [5, 50, 95]
    ).tolist(),
    "variants": {},
}
dev = torch.device("cuda")
for name, cfg in VARIANTS.items():
    res = train_splats(frames, kmat, views, sparse_cv, colors, cfg, log=log)
    p = {key: torch.as_tensor(v, device=dev) for key, v in res.params.items()}
    with torch.no_grad():
        img, alpha, _ = rasterization(
            p["means"],
            p["quats"],
            torch.exp(p["scales"]),
            torch.sigmoid(p["opacities"]),
            torch.sigmoid(p["colors"]),
            torch.as_tensor(vm_eval, dtype=torch.float32, device=dev)[None],
            torch.as_tensor(kmat, dtype=torch.float32, device=dev)[None],
            truth.shape[1],
            truth.shape[0],
            packed=False,
            render_mode="RGB+ED",
        )
    rgb = (img[0, :, :, :3].clamp(0, 1) * 255).byte().cpu().numpy()
    depth = img[0, :, :, 3].cpu().numpy()
    a = alpha[0, :, :, 0].cpu().numpy()
    # Gaussians in front of the eval camera, closer than the nearest sparse point (likely fog)
    means = res.params["means"].astype(np.float64)
    opac = 1 / (1 + np.exp(-res.params["opacities"]))
    scale = np.exp(res.params["scales"]).max(1)
    cam_z = (means @ vm_eval[:3, :3].T + vm_eval[:3, 3])[:, 2]
    near_limit = report["sparse_depth_in_eval_view_p5_p50_p95"][0]
    fog = (cam_z > 0) & (cam_z < 0.8 * near_limit)
    report["variants"][name] = {
        "psnr_nonsky": psnr(rgb, truth, valid),
        "train_view_psnr": float(np.mean([psnr(render, frames[i]) for i, render in []] or [0])),
        "gaussians": res.stats["gaussians"],
        "eval_expected_depth_median": float(np.median(depth[valid])),
        "eval_alpha_mean": float(a.mean()),
        "gaussians_in_front_of_scene": int(fog.sum()),
        "their_opacity_sum": float(opac[fog].sum()),
        "their_scale_median": float(np.median(scale[fog])) if fog.any() else None,
        "their_distance_to_nearest_train_camera_median": float(
            np.median(np.min(np.linalg.norm(means[fog][:, None] - centers[None], axis=2), axis=1))
        )
        if fog.any()
        else None,
        "scene_extent": res.stats["scene_extent"],
    }
    dvis = cv2.applyColorMap(
        np.clip(
            depth / (2 * report["sparse_depth_in_eval_view_p5_p50_p95"][1]) * 255, 0, 255
        ).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    row = np.concatenate(
        [cv2.cvtColor(truth, cv2.COLOR_RGB2BGR), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dvis], 1
    )
    cv2.imwrite(str(OUT / f"diag_{name}.png"), row)
    log(f"{name}: {report['variants'][name]}")
    save_json("diag.json", report)
log("done")
