# Job: train Gaussian splats for each attached scene and export what the laptop needs to build
# a signed .wreel (tools/splat_wreel.py): the trained parameters, the classical reconstruction
# .wreel as a template, and the alignment metadata. No signing key leaves the laptop: the
# template is signed with a throwaway key here and re-signed locally. Runs after bootstrap.py.
import os
import shutil
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
from stages.camera.classical import SPARSE_COLORS, SPARSE_POINTS, load_frames
from stages.reconstruct.planesweep import cv_pose
from stages.reconstruct.splat_train import SplatConfig, train_splats

FLIP = np.diag([1.0, -1.0, -1.0])
SEED = int(os.environ.get("WORLDREEL_SPLAT_SEED", "1"))


def viewmat(position, rotation):
    r, t = cv_pose(position, rotation)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


for scene in find_scenes():
    name = scene.name.replace("worldreel-scene-", "")
    work = WORK / "bench" / name
    classical_benchmark(scene, work, calibrated=True)
    item = next(p for p in work.iterdir() if p.is_dir())
    cam = json.loads((item / "camera" / "camera.json").read_text())
    k = cam["intrinsics"]
    kmat = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    frames = load_frames(item / "frames" / "frames")
    views = np.stack([viewmat(p["position"], p["rotation"]) for p in cam["poses"]])
    sparse_cv = np.load(item / "camera" / SPARSE_POINTS).astype(np.float64) @ FLIP.T
    colors = np.load(item / "camera" / SPARSE_COLORS)
    cfg = SplatConfig(seed=SEED)
    res = train_splats(frames, kmat, views, sparse_cv, colors, cfg, log=log)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "splats.npz", **res.params)
    shutil.copy(item / "package" / "world.wreel", out / "template.wreel")
    centre = np.median(sparse_cv, axis=0)
    save_json(
        f"{name}/meta.json",
        {
            "centre_cv": centre.tolist(),
            "extent": res.stats["scene_extent"],
            "background_radius": cfg.background_radius,
            "training": res.stats,
            "seed": SEED,
        },
    )
    log(f"{name}: {res.stats['gaussians']} gaussians exported")
log("done")
