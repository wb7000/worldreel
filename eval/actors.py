"""Actor (moving people) metrics against synthetic ground truth.

Mask quality: per frame, predicted tracks are matched to true actors by Hungarian assignment on
mask IoU (a pair counts only if IoU >= `min_iou`). Reported: mean IoU of matched true actors
(misses count as 0), ID switches (CLEAR-MOT definition: a true actor's matched predicted track
changes between frames where it is matched), false-positive tracks per frame.

3D quality: the lifted track is compared with the truth in each frame's own camera frame, scaled
to metres by the trajectory Sim(3). This removes the global alignment residual, which is a
camera error, not an actor error (DECISIONS D-3b.9).

Usage: python -m eval.actors <scene_dir> <item_dir>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from eval.geometry import quat_to_matrix, trajectory_errors
from formats.wreel import read_wreel
from stages.actors.common import read_tracks

U8 = npt.NDArray[np.uint8]


@dataclass
class MaskMetrics:
    mean_iou: float
    id_switches: int
    false_positives_per_frame: float
    frames: int


def mask_metrics(pred: U8, truth: U8, *, min_iou: float = 0.5) -> MaskMetrics:
    """`pred` and `truth` are label stacks (N, H, W): 0 = nobody, k = track / actor k."""
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {truth.shape}")
    true_ids = [int(t) for t in np.unique(truth) if t]
    last_match: dict[int, int] = {}
    ious: list[float] = []
    switches = 0
    false_pos = 0
    for p_frame, t_frame in zip(pred, truth, strict=True):
        p_ids = [int(p) for p in np.unique(p_frame) if p]
        t_ids = [t for t in true_ids if (t_frame == t).any()]
        iou = np.zeros((len(t_ids), len(p_ids)))
        for a, t in enumerate(t_ids):
            tm = t_frame == t
            for b, p in enumerate(p_ids):
                pm = p_frame == p
                union = int(np.count_nonzero(tm | pm))
                iou[a, b] = int(np.count_nonzero(tm & pm)) / max(union, 1)
        matched: dict[int, tuple[int, float]] = {}
        if iou.size:
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                if iou[r, c] >= min_iou:
                    matched[t_ids[r]] = (p_ids[c], float(iou[r, c]))
        for t in t_ids:
            if t in matched:
                p, value = matched[t]
                ious.append(value)
                if t in last_match and last_match[t] != p:
                    switches += 1
                last_match[t] = p
            else:
                ious.append(0.0)
        false_pos += len(p_ids) - len(matched)
    return MaskMetrics(
        mean_iou=float(np.mean(ious)) if ious else 1.0,
        id_switches=switches,
        false_positives_per_frame=false_pos / max(len(pred), 1),
        frames=len(pred),
    )


def track_metrics(scene: Path, item: Path) -> dict[str, Any]:
    """Local 3D position error and height per packaged actor, matched to the nearest true actor."""
    gt = json.loads((scene / "gt.json").read_text())
    world = read_wreel(item / "package" / "world.wreel")
    cam = json.loads((item / "camera" / "camera.json").read_text())
    est_pos = np.array([p["position"] for p in cam["poses"]])
    est_rot = np.stack([quat_to_matrix(p["rotation"]) for p in cam["poses"]])
    gt_pos = np.array([f["camera"]["position"] for f in gt["frames"]])
    gt_rot = np.stack([quat_to_matrix(f["camera"]["rotation"]) for f in gt["frames"]])
    _, sim = trajectory_errors(est_pos, est_rot, gt_pos, gt_rot)
    truth_tracks = [np.asarray(a["track"]["positions"]) for a in gt["actors"]]
    out: dict[str, Any] = {}
    for actor in world.manifest["actors"]:
        tr = world.track(actor["track"]["path"])
        frames = np.rint(np.asarray(tr["times"]) * gt["fps"]).astype(int)
        local = np.stack(
            [
                est_rot[f].T @ (np.asarray(p) - est_pos[f]) * sim.scale
                for f, p in zip(frames, tr["positions"], strict=True)
            ]
        )
        best: tuple[float, int, npt.NDArray[np.float64]] | None = None
        for k, truth in enumerate(truth_tracks):
            gt_local = np.einsum("nji,nj->ni", gt_rot[frames], truth[frames] - gt_pos[frames])
            errs = np.linalg.norm(local - gt_local, axis=1)
            if best is None or np.median(errs) < best[0]:
                best = (float(np.median(errs)), k, errs)
        assert best is not None
        out[actor["id"]] = {
            "matched_actor": gt["actors"][best[1]].get("id", best[1]),
            "samples": len(frames),
            "local_error_median_m": round(best[0], 3),
            "local_error_p90_m": round(float(np.percentile(best[2], 90)), 3),
            "height_m": round(actor["eye_height"] / 0.925 * sim.scale, 3),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene", type=Path, help="synthetic scene folder (gt.json)")
    ap.add_argument("item", type=Path, help="pipeline output folder of one item")
    ap.add_argument("--truth-masks", type=Path, help="label stack (.npz key 'labels') to score")
    args = ap.parse_args(argv)
    report: dict[str, Any] = {"tracks": track_metrics(args.scene, args.item)}
    if args.truth_masks:
        _, pred = read_tracks(args.item / "actors")
        truth = np.load(args.truth_masks)["labels"]
        if pred is not None:
            report["masks"] = asdict(mask_metrics(pred, truth))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
