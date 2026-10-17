# Job: verify the person-detection and tracking models before building on them.
# DETR-ResNet-50 (facebook/detr-resnet-50, Apache-2.0) and SAM 2.1 hiera_small (Apache-2.0,
# github.com/facebookresearch/sam2). Prints real signatures/labels; runs both on a scene.
# Runs after bootstrap.py.
import inspect
import subprocess
import sys
import time

report: dict = {}


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


t = time.time()
r = sh(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "git+https://github.com/facebookresearch/sam2.git",
    ]
)
report["sam2_install"] = {
    "rc": r.returncode,
    "s": round(time.time() - t, 1),
    "err": r.stderr[-1200:],
}
log(f"sam2 install rc={r.returncode} in {report['sam2_install']['s']}s")
ckpt = WORK / "sam2.1_hiera_small.pt"
r = sh(
    [
        "wget",
        "-q",
        "-O",
        str(ckpt),
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    ]
)
report["ckpt_bytes"] = ckpt.stat().st_size if ckpt.exists() else 0
log(f"checkpoint {report['ckpt_bytes'] / 1e6:.1f} MB")

import numpy as np
import torch

try:
    import sam2
    from sam2 import build_sam

    pkg = __import__("pathlib").Path(sam2.__file__).parent
    report["sam2_configs"] = sorted(
        str(p.relative_to(pkg)) for p in pkg.rglob("sam2.1_hiera_s*.yaml")
    )
    report["build_video_predictor_sig"] = str(
        inspect.signature(build_sam.build_sam2_video_predictor)
    )
    log(f"configs {report['sam2_configs']}")
except Exception as exc:
    report["sam2_import_error"] = repr(exc)
    log(f"sam2 import failed: {exc!r}")

import transformers
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

report["transformers_version"] = transformers.__version__
frames: list = []
scene = find_scenes()[0]
try:
    import json

    from pipeline.io import read_rgb

    gt = json.loads((scene / "gt.json").read_text())
    frames = [read_rgb(scene / f["rgb"]) for f in gt["frames"][:8]]
    proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    # The no_timm config stores "dilation": null, which strict config validation in recent
    # transformers rejects (seen on Kaggle). False is the plain ResNet-50 backbone.
    cfg_dict, _ = DetrConfig.get_config_dict("facebook/detr-resnet-50", revision="no_timm")
    report["detr_config_dilation_raw"] = cfg_dict.get("dilation")
    if cfg_dict.get("dilation") is None:
        cfg_dict["dilation"] = False
    detr = (
        DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm", config=DetrConfig.from_dict(cfg_dict)
        )
        .cuda()
        .eval()
    )
    labels = detr.config.id2label
    report["detr_has_person"] = [i for i, n in labels.items() if n == "person"]
    log(f"DETR person label ids: {report['detr_has_person']}")
    dets = []
    for img in frames:
        inputs = proc(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = detr(**inputs)
        res = proc.post_process_object_detection(out, target_sizes=[img.shape[:2]], threshold=0.5)[
            0
        ]
        dets.append(
            [
                (labels[int(lab)], round(float(s), 3), [round(float(x), 1) for x in b])
                for s, lab, b in zip(res["scores"], res["labels"], res["boxes"], strict=False)
            ]
        )
    report["detr_on_synthetic"] = dets[:3]
    log(f"DETR detections frame 0: {dets[0]}")
except Exception as exc:
    import traceback

    report["detr_error"] = traceback.format_exc()[-2000:]
    log(f"DETR failed: {exc!r}")

# SAM2 video tracking from a ground-truth box of actor 1 on frame 0.
try:
    from sam2.build_sam import build_sam2_video_predictor

    from tools.synth import actor_masks

    jpeg = WORK / "jpeg"
    jpeg.mkdir(exist_ok=True)
    import cv2

    for i, img in enumerate(frames):
        cv2.imwrite(str(jpeg / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cfg = next(c for c in report["sam2_configs"] if c.endswith("sam2.1_hiera_s.yaml"))
    predictor = build_sam2_video_predictor(
        "configs/" + cfg.split("configs/", 1)[1] if "configs/" in cfg else cfg,
        str(ckpt),
        device="cuda",
    )
    report["predictor_methods"] = {
        m: str(inspect.signature(getattr(predictor, m)))
        for m in ("init_state", "add_new_points_or_box", "propagate_in_video")
    }
    lab0 = actor_masks(scene, 0)
    ys, xs = np.nonzero(lab0 == 1)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
    state = predictor.init_state(video_path=str(jpeg))
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, box=box)
    ious = []
    for fidx, _obj_ids, logits in predictor.propagate_in_video(state):
        pred = (logits[0, 0] > 0).cpu().numpy()
        truth = actor_masks(scene, fidx) == 1
        inter, union = (pred & truth).sum(), (pred | truth).sum()
        ious.append(round(float(inter / union), 3) if union else None)
    report["sam2_iou_vs_gt_actor1"] = ious
    log(f"SAM2 IoU per frame (actor 1): {ious}")
except Exception as exc:
    import traceback

    report["sam2_run_error"] = traceback.format_exc()[-2000:]
    log(f"SAM2 run failed: {exc!r}")
save_json("actors_env.json", report)
log("done")
