"""Compare follow-camera paths (algorithm A-2) with the behind-the-actor baseline on a splat world.

Per actor: seen share (mean confidence of the front-most Gaussian per pixel, the planner's own
objective) and the view-direction gap to the nearest training camera ray (independent proxy for
novel-view quality), for the baseline offset [0, 1.15h, 1.0h] in the actor frame and the plan.

Usage: python -m eval.follow <world.wreel with A-1 confidence> <camera.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.geometry import quat_to_matrix
from formats.wreel import GaussianCloud, read_wreel
from stages.package.follow import FollowConfig, plan_follow, seen_score, view_gap_deg


def scene_gaussians(world: Any, max_points: int = 60000, seed: int = 0) -> GaussianCloud:
    layer = next(lyr for lyr in world.manifest["scene"]["layers"] if lyr["kind"] == "gaussians")
    g = GaussianCloud.unpack(world.entries[layer["lods"][-1]["path"]])
    keep = np.nonzero(g.opacity >= 0.2)[0]
    if len(keep) > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(keep, max_points, replace=False))
    return GaussianCloud(*(getattr(g, f)[keep] for f in g.__dataclass_fields__))


def run(world_path: Path, camera: Path) -> dict[str, Any]:
    world = read_wreel(world_path)
    g = scene_gaussians(world)
    pts = g.positions.astype(np.float64)
    cam = json.loads(camera.read_text(encoding="utf-8"))
    centres = np.array([p["position"] for p in cam["poses"] if p.get("registered", True)])
    cfg = FollowConfig()
    out: dict[str, Any] = {"gaussians_scored": len(pts)}
    for actor in world.manifest["actors"]:
        tr = world.track(actor["track"]["path"])
        pos = np.asarray(tr["positions"], np.float64)
        rots = [quat_to_matrix(q) for q in tr["rotations"]]
        ups = np.stack([r @ np.array([0.0, 1.0, 0.0]) for r in rots])
        h = actor["eye_height"] / 0.925
        base_eyes = np.stack(
            [p + r @ np.array([0.0, 1.15 * h, 1.0 * h]) for p, r in zip(pos, rots, strict=True)]
        )
        base_targets = pos + 0.55 * h * ups
        base = [
            seen_score(e, t, u, pts, g.confidence, cfg)
            for e, t, u in zip(base_eyes, base_targets, ups, strict=True)
        ]
        eyes, targets, scores = plan_follow(pos, ups, h, pts, g.confidence, cfg)
        gap_b = view_gap_deg(base_eyes, base_targets, centres)
        gap_p = view_gap_deg(eyes, targets, centres)
        out[actor["id"]] = {
            "samples": len(pos),
            "seen_baseline": round(float(np.mean([s for s, _ in base])), 3),
            "clear_baseline": round(float(np.mean([c for _, c in base])), 3),
            "seen_planned": round(float(scores.mean()), 3),
            "gap_deg_baseline_mean_p90": [
                round(float(gap_b.mean()), 1),
                round(float(np.percentile(gap_b, 90)), 1),
            ],
            "gap_deg_planned_mean_p90": [
                round(float(gap_p.mean()), 1),
                round(float(np.percentile(gap_p, 90)), 1),
            ],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("world", type=Path)
    ap.add_argument("camera", type=Path)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.world, a.camera), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
