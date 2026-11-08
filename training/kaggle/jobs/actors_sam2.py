# Job: run the real actors stage (DETR + SAM 2.1) inside the full pipeline on a synthetic scene
# and score masks and 3D tracks against ground truth. Runs after bootstrap.py.
# NOTE: DETR labels the synthetic cylinder-and-sphere people as "vase" (checked by actors_env),
# so this TEST accepts "vase" boxes too. It measures tracking + masks + lifting, not detection
# of real humans; that needs real footage.
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np

report: dict = {}
t = time.time()
r = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "git+https://github.com/facebookresearch/sam2.git",
    ],
    capture_output=True,
    text=True,
)
log(f"sam2 install rc={r.returncode} in {time.time() - t:.0f}s")
ckpt = WORK / "sam2.1_hiera_small.pt"
subprocess.run(
    [
        "wget",
        "-q",
        "-O",
        str(ckpt),
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    ],
    check=True,
)

from eval.actors import mask_metrics, track_metrics
from pipeline.registry import VIDEO_PIPELINE, build_stages
from pipeline.runner import Runner
from pipeline.stage import Item
from stages.actors.common import read_tracks
from stages.safety.hook import EnforcedSafety
from tools.synth import actor_masks

# Stage crashes are logged with their traceback at DEBUG; show them in the kernel log.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("worldreel.pipeline").setLevel(logging.DEBUG)

scene = find_scenes()[0]
gt = json.loads((scene / "gt.json").read_text())
work = WORK / "pipe"
impl = {
    "frames": "images",
    "actors": "real",
    "camera": "real",
    "reconstruct": "real",
    "package": "real",
}
cfg = {
    "frames": {"fps": float(gt["fps"])},
    "actors": {
        "sam2_checkpoint": str(ckpt),
        "labels": ["person", "vase"],  # test-only, see NOTE above
        "score": 0.5,
        "detect_every": 5,
    },
    "camera": {"intrinsics": gt["intrinsics"]},
}
item = Item(
    id="actors-real",
    sha256=hashlib.sha256((scene / "gt.json").read_bytes()).hexdigest(),
    path=scene / "rgb",
    source="synthetic",
    license="CC0-1.0",
    upload_rights="synthetic",
)
t = time.time()
res = Runner(
    work,
    build_stages(VIDEO_PIPELINE, {n: impl.get(n, "mock") for n in VIDEO_PIPELINE}),
    config=cfg,
    safety=EnforcedSafety(work, cfg.get("safety", {})),
).run(item)
report["pipeline_s"] = round(time.time() - t, 1)
report["stages"] = {n: (r.status, r.error) for n, r in res.stages.items()}
log(f"pipeline {report['stages']}")
meta, pred = read_tracks(work / item.id / "actors")
report["tracks_found"] = len(meta["tracks"])
truth = np.stack([actor_masks(scene, i) for i in range(len(gt["frames"]))])
if pred is not None:
    report["masks"] = asdict(mask_metrics(pred, truth))
    log(f"masks {report['masks']}")
if res.stages["package"].status == "done":
    report["tracks"] = track_metrics(scene, work / item.id)
    log(f"tracks {report['tracks']}")
save_json("actors_sam2.json", report)
log("done")
