"""`actors` (real): DETR person boxes on keyframes, linked into tracks, then SAM 2.1 video masks.

Models (both Apache-2.0, interfaces checked on Kaggle, DECISIONS D-3b.10):
  * facebook/detr-resnet-50, revision "no_timm" (Hugging Face transformers); "person" = id 1.
  * SAM 2.1 hiera_small (github.com/facebookresearch/sam2), video predictor.

Steps:
  1. DETR on every `detect_every`-th frame; keep boxes with label in `labels` and score >= `score`.
  2. Link keyframe boxes into tracks by box IoU (Hungarian per keyframe); drop tracks seen on
     fewer than `min_hits` keyframes (one-off false positives).
  3. Prompt SAM 2.1 with each track's first box at its frame; propagate through the video.
  4. Merge per-object logits into one label image per frame (highest logit wins, > 0 only).
  5. Merge objects that segment the same person (`duplicate_map`, `merge_iou` 0.5).

Config: detect_every (5), score (0.8), labels (["person"]), min_hits (2), link_iou (0.3),
max_gap (3 keyframes), sam2_checkpoint (path, required), sam2_config
("configs/sam2.1/sam2.1_hiera_s.yaml"), device ("cuda"), max_people (32).
Needs a GPU; the laptop runs `mock` or `oracle` instead.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from pipeline.io import FRAMES_DIR
from pipeline.stage import Impl, StageContext, StageError
from stages.actors.common import write_tracks
from stages.camera.classical import load_frames

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
F32 = npt.NDArray[np.float32]


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class BoxTrack:
    boxes: dict[int, Box] = field(default_factory=dict)  # frame -> box

    @property
    def first(self) -> tuple[int, Box]:
        f = min(self.boxes)
        return f, self.boxes[f]

    @property
    def last(self) -> tuple[int, Box]:
        f = max(self.boxes)
        return f, self.boxes[f]


def link_detections(
    detections: dict[int, list[Box]],
    *,
    link_iou: float = 0.3,
    max_gap: int = 3,
    min_hits: int = 2,
) -> list[BoxTrack]:
    """Greedy-optimal (Hungarian) IoU linking of keyframe boxes. `max_gap` is in keyframes."""
    keyframes = sorted(detections)
    order = {f: i for i, f in enumerate(keyframes)}
    live: list[BoxTrack] = []
    done: list[BoxTrack] = []
    for f in keyframes:
        boxes = detections[f]
        expired = [t for t in live if order[f] - order[t.last[0]] > max_gap]
        done += expired
        live = [t for t in live if t not in expired]
        cost = np.array([[box_iou(t.last[1], b) for b in boxes] for t in live]).reshape(
            len(live), len(boxes)
        )
        used: set[int] = set()
        if cost.size:
            rows, cols = linear_sum_assignment(-cost)
            for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                if cost[r, c] >= link_iou:
                    live[r].boxes[f] = boxes[c]
                    used.add(c)
        live += [BoxTrack({f: b}) for c, b in enumerate(boxes) if c not in used]
    tracks = done + live
    tracks = [t for t in tracks if len(t.boxes) >= min_hits]
    return sorted(tracks, key=lambda t: t.first[0])


def merge_logits(per_object: dict[int, F32], shape: tuple[int, int]) -> npt.NDArray[np.uint8]:
    """{label k: logits (H, W)} -> uint8 labels; a pixel takes the highest positive logit."""
    labels = np.zeros(shape, np.uint8)
    if not per_object:
        return labels
    keys = sorted(per_object)
    stack = np.stack([per_object[k] for k in keys])
    best = stack.argmax(0)
    positive = stack.max(0) > 0
    labels[positive] = np.asarray(keys, np.uint8)[best[positive]]
    return labels


Mask = npt.NDArray[np.bool_]


def duplicate_map(
    masks: dict[int, dict[int, Mask]],
    *,
    min_iou: float = 0.5,
    separate_iou: float = 0.2,
    min_frames: int = 3,
) -> dict[int, int]:
    """Objects that segment the same person: {object: representative} (smallest id wins).

    A track breaks when a person is lost for a while; the detector then starts a new track while
    SAM 2 may still follow the person with the old object, so both segment the same body.
    Two objects are duplicates when, over the frames both are visible (>= `min_frames`), their
    median IoU is >= `min_iou`. Pairs whose median IoU is < `separate_iou` are different people
    seen together and are never put in one group, even through a chain of other merges (an
    earlier "intersection / smaller area" rule chained two friends into one track on the walking
    clip, DECISIONS D-3b.11).
    """
    ids = sorted(masks)
    pair_iou: dict[tuple[int, int], float] = {}
    for x, a in enumerate(ids):
        for b in ids[x + 1 :]:
            shared = [f for f in masks[a] if f in masks[b]]
            if len(shared) < min_frames:
                continue
            ious = []
            for f in shared:
                ma, mb = masks[a][f], masks[b][f]
                union = int((ma | mb).sum())
                ious.append(int((ma & mb).sum()) / union if union else 0.0)
            pair_iou[(a, b)] = float(np.median(ious))
    group = {i: {i} for i in ids}
    for (a, b), iou in sorted(pair_iou.items(), key=lambda kv: -kv[1]):
        if iou < min_iou or group[a] is group[b]:
            continue
        ga, gb = group[a], group[b]
        conflict = any(
            pair_iou.get((min(x, y), max(x, y)), 1.0) < separate_iou for x in ga for y in gb
        )
        if conflict:
            continue
        merged = ga | gb
        for i in merged:
            group[i] = merged
    return {i: min(group[i]) for i in ids}


OBJECTS_NPZ = "objects_small.npz"


def save_objects(path: Path, small: dict[int, dict[int, Mask]]) -> None:
    """Per-object quarter-resolution masks before merging, so merging can be re-tuned offline."""
    arrays: dict[str, npt.NDArray[np.generic]] = {}
    for o, per_frame in small.items():
        frames = sorted(per_frame)
        arrays[f"o{o}_frames"] = np.asarray(frames, np.int32)
        if frames:
            arrays[f"o{o}_masks"] = np.stack([per_frame[f] for f in frames])
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


def load_objects(path: Path) -> dict[int, dict[int, Mask]]:
    out: dict[int, dict[int, Mask]] = {}
    with np.load(path) as npz:
        for key in npz.files:
            if key.endswith("_frames"):
                o = int(key[1:-7])
                frames = npz[key].tolist()
                masks = npz[f"o{o}_masks"] if frames else []
                out[o] = {f: masks[i] for i, f in enumerate(frames)}
    return out


def relabel(labels: npt.NDArray[np.uint8], mapping: dict[int, int]) -> npt.NDArray[np.uint8]:
    """Apply {old: representative} and renumber representatives 1..n in order."""
    reps = sorted(set(mapping.values()))
    lut = np.zeros(256, np.uint8)
    for old, rep in mapping.items():
        lut[old] = reps.index(rep) + 1
    return lut[labels]


def _load_detr(device: str) -> tuple[Any, Any]:
    from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

    repo, rev = "facebook/detr-resnet-50", "no_timm"
    proc = DetrImageProcessor.from_pretrained(repo, revision=rev)
    # The no_timm config stores "dilation": null, which strict config validation in recent
    # transformers (5.16.1 on Kaggle) rejects; False is the plain ResNet-50 backbone.
    cfg_dict, _ = DetrConfig.get_config_dict(repo, revision=rev)
    if cfg_dict.get("dilation") is None:
        cfg_dict["dilation"] = False
    model = DetrForObjectDetection.from_pretrained(
        repo, revision=rev, config=DetrConfig.from_dict(cfg_dict)
    )
    return proc, model.to(device).eval()


def detect(
    frames: list[npt.NDArray[np.uint8]],
    keyframes: list[int],
    labels: list[str],
    score: float,
    device: str,
) -> dict[int, list[Box]]:
    import torch

    proc, model = _load_detr(device)
    names = model.config.id2label
    out: dict[int, list[Box]] = {}
    for f in keyframes:
        img = frames[f]
        inputs = proc(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            pred = model(**inputs)
        res = proc.post_process_object_detection(
            pred, target_sizes=[img.shape[:2]], threshold=score
        )[0]
        out[f] = [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for lab, b in zip(res["labels"].tolist(), res["boxes"].tolist(), strict=True)
            if names[int(lab)] in labels
        ]
    return out


def segment(
    frames: list[npt.NDArray[np.uint8]],
    tracks: list[BoxTrack],
    checkpoint: Path,
    config_name: str,
    device: str,
    *,
    min_iou: float = 0.5,
    objects_out: Path | None = None,
) -> tuple[npt.NDArray[np.uint8], dict[int, int]]:
    """Returns (labels after merging duplicates, {SAM object id: merged label})."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    h, w = frames[0].shape[:2]
    labels = np.zeros((len(frames), h, w), np.uint8)
    if not tracks:
        return labels, {}
    small: dict[int, dict[int, Mask]] = {k: {} for k in range(1, len(tracks) + 1)}
    predictor = build_sam2_video_predictor(config_name, str(checkpoint), device=device)
    jpeg = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    try:
        # The SAM 2 video predictor reads a folder of JPEG frames named by index.
        for i, img in enumerate(frames):
            cv2.imwrite(str(jpeg / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Same context as the official SAM 2 README video example; without bf16 autocast the
        # predictor mixed bfloat16 and float32 tensors on Kaggle (mat1/mat2 dtype error).
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(jpeg), offload_video_to_cpu=True)
            for k, track in enumerate(tracks, start=1):
                f0, box = track.first
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=f0, obj_id=k, box=np.array(box, np.float32)
                )
            for fidx, obj_ids, logits in predictor.propagate_in_video(state):
                per_obj = {
                    int(o): logits[j, 0].float().cpu().numpy() for j, o in enumerate(obj_ids)
                }
                labels[fidx] = merge_logits(per_obj, (h, w))
                for o, lg in per_obj.items():
                    m = lg[::4, ::4] > 0  # quarter resolution is enough to compare objects
                    if m.any():
                        small[o][fidx] = m
    finally:
        shutil.rmtree(jpeg, ignore_errors=True)
    if objects_out is not None:
        save_objects(objects_out, small)
    mapping = duplicate_map(small, min_iou=min_iou)
    merged = relabel(labels, mapping)
    return merged, {o: int(relabel(np.array([o], np.uint8), mapping)[0]) for o in mapping}


class Sam2ActorsStage:
    name: ClassVar[str] = "actors"
    version: ClassVar[str] = "0.3.0"
    impl: ClassVar[Impl] = "real"
    depends_on: ClassVar[tuple[str, ...]] = ("frames",)

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        checkpoint = Path(str(cfg.get("sam2_checkpoint", "")))
        if not checkpoint.is_file():
            raise StageError("real actors need config 'sam2_checkpoint' (SAM 2.1 .pt file)")
        device = str(cfg.get("device", "cuda"))
        frames = load_frames(ctx.inputs["frames"] / FRAMES_DIR)
        every = max(1, int(cfg.get("detect_every", 5)))
        keyframes = list(range(0, len(frames), every))
        dets = detect(
            frames,
            keyframes,
            list(cfg.get("labels", ["person"])),
            float(cfg.get("score", 0.8)),
            device,
        )
        tracks = link_detections(
            dets,
            link_iou=float(cfg.get("link_iou", 0.3)),
            max_gap=int(cfg.get("max_gap", 3)),
            min_hits=int(cfg.get("min_hits", 2)),
        )[: int(cfg.get("max_people", 32))]
        labels, merged = segment(
            frames,
            tracks,
            checkpoint,
            str(cfg.get("sam2_config", "configs/sam2.1/sam2.1_hiera_s.yaml")),
            device,
            min_iou=float(cfg.get("merge_iou", 0.5)),
            objects_out=ctx.out / OBJECTS_NPZ,
        )
        people = len(set(merged.values()))
        ctx.log.info("actors: %d box tracks -> %d people after merging", len(tracks), people)
        write_tracks(ctx.out, labels, "detr-resnet-50+sam2.1-hiera-small", ["person"] * people)
