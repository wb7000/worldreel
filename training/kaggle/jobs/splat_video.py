# Job: reconstruct an uploaded real video with Gaussian splats. Runs the WorldReel pipeline with
# the uploaded pipeline.toml (ffmpeg frames, classical camera + plane sweep, package), then trains
# splats on the frames the camera stage could register, and exports the parameters plus the
# packaged reconstruction as a template for tools/splat_wreel.py. Runs after bootstrap.py.
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
log(f"ffmpeg: {shutil.which('ffmpeg')}, ffprobe: {shutil.which('ffprobe')}")

import json

import numpy as np

from pipeline.registry import VIDEO_PIPELINE, PipelineConfig, build_stages
from pipeline.runner import Runner
from pipeline.stage import Item
from stages.actors.common import static_masks
from stages.camera.classical import SPARSE_COLORS, SPARSE_POINTS, load_frames
from stages.reconstruct.planesweep import cv_pose
from stages.reconstruct.splat_train import SplatConfig, train_splats
from stages.safety.hook import PassThroughSafety

FLIP = np.diag([1.0, -1.0, -1.0])
SEED = int(os.environ.get("WORLDREEL_SPLAT_SEED", "1"))


SAM2_CKPT = WORK / "sam2.1_hiera_small.pt"


def ensure_sam2():
    """SAM 2.1 (Apache-2.0) for the real actors stage; checkpoint URL from the official repo."""
    try:
        __import__("sam2")
    except ImportError:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "git+https://github.com/facebookresearch/sam2.git",
            ],
            check=True,
        )
    if not SAM2_CKPT.exists():
        subprocess.run(
            [
                "wget",
                "-q",
                "-O",
                str(SAM2_CKPT),
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            ],
            check=True,
        )
    log("sam2 ready")


def viewmat(position, rotation):
    r, t = cv_pose(position, rotation)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


for meta_file in sorted(INPUT.rglob("item.json")):
    folder = meta_file.parent
    name = next(
        (p.name for p in folder.parents if p.name.startswith("worldreel-video")), folder.name
    )
    name = folder.name if folder.name.startswith("worldreel-video") else name
    name = name.replace("worldreel-video-", "")
    meta = json.loads(meta_file.read_text())
    video = next(p for p in folder.iterdir() if p.name.startswith("video."))
    item = Item(
        id=meta["id"],
        sha256=meta["sha256"],
        path=video,
        source=meta["source"],
        license=meta["license"],
        upload_rights=meta["upload_rights"],
    )
    cfg = PipelineConfig.load(folder / "pipeline.toml")
    if cfg.impl.get("actors") == "real":
        ensure_sam2()
        cfg.stage.setdefault("actors", {})["sam2_checkpoint"] = str(SAM2_CKPT)
    work = WORK / "processed"
    runner = Runner(
        work,
        build_stages(VIDEO_PIPELINE, cfg.impl),
        config=cfg.stage,
        seed=cfg.seed,
        max_workers=cfg.max_workers,
        safety=PassThroughSafety(work),
    )
    result = runner.run(item)
    log(
        f"{name}: pipeline {'ok' if result.ok else 'FAILED'}: "
        + ", ".join(f"{n}={r.status}" for n, r in result.stages.items())
    )
    if result.stages["camera"].status != "done":
        log(f"camera failed: {result.stages['camera'].error}")
        continue
    out_item = work / item.id
    (OUT / name).mkdir(parents=True, exist_ok=True)
    if (out_item / "actors" / "actors.json").exists():
        actors_meta = json.loads((out_item / "actors" / "actors.json").read_text())
        save_json(
            f"{name}/actors.json",
            {
                "backend": actors_meta.get("backend"),
                "tracks": [
                    {"id": t["id"], "frames": len(t["frames"]), "max_pixels": max(t["pixels"])}
                    for t in actors_meta.get("tracks", [])
                ],
            },
        )
        log(f"{name}: {len(actors_meta.get('tracks', []))} person tracks")
    for small_file in ("masks.npz", "objects_small.npz"):
        if (out_item / "actors" / small_file).exists():
            shutil.copy(out_item / "actors" / small_file, OUT / name / small_file)
    if (out_item / "package" / "actors_report.json").exists():
        shutil.copy(out_item / "package" / "actors_report.json", OUT / name / "actors_report.json")
    cam = json.loads((out_item / "camera" / "camera.json").read_text())
    reg = [i for i, p in enumerate(cam["poses"]) if p.get("registered", True)]
    k = cam["intrinsics"]
    kmat = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    frames_all = load_frames(out_item / "frames" / "frames")
    frames = [frames_all[i] for i in reg]
    views = np.stack(
        [viewmat(cam["poses"][i]["position"], cam["poses"][i]["rotation"]) for i in reg]
    )
    sparse_cv = np.load(out_item / "camera" / SPARSE_POINTS).astype(np.float64) @ FLIP.T
    colors = np.load(out_item / "camera" / SPARSE_COLORS)
    log(
        f"{name}: training on {len(frames)} registered frames of {len(frames_all)}, {len(sparse_cv)} SfM points"
    )
    splat_cfg = SplatConfig(seed=SEED)
    people = static_masks(out_item / "actors", len(frames_all), frames_all[0].shape[:2])
    ignore = [people[i] for i in reg] if people is not None else None
    res = train_splats(frames, kmat, views, sparse_cv, colors, splat_cfg, log=log, ignore=ignore)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "splats.npz", **res.params)
    if (out_item / "package" / "world.wreel").exists():
        shutil.copy(out_item / "package" / "world.wreel", out / "template.wreel")
    save_json(
        f"{name}/meta.json",
        {
            "centre_cv": np.median(sparse_cv, axis=0).tolist(),
            "extent": res.stats["scene_extent"],
            "background_radius": splat_cfg.background_radius,
            "training": res.stats,
            "seed": SEED,
            "registered_frames": len(reg),
            "total_frames": len(frames_all),
            "camera_stats": cam.get("stats", {}),
        },
    )
    log(f"{name}: {res.stats['gaussians']} gaussians exported")
log("done")
