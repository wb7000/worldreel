"""Synthetic scenes with exact ground truth, rendered by Blender (brief §9).

    python -m tools.synth render --seed 1 [--out data/synthetic/seed0001] [--engine EEVEE]
    python -m tools.synth check data/synthetic/seed0001
    python -m tools.synth to-inbox data/synthetic/seed0001     # encode MP4 + sidecar into inbox

A scene folder holds rgb/*.png, depth/*.npy and gt.json (see tools/blender/gen_scene.py).
`check` back-projects depth through the ground-truth cameras and verifies the result is
geometrically consistent, which catches any mistake in intrinsics, poses or depth semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from stages.reconstruct.mock import quat_to_matrix
from tools.toolpaths import REPO_ROOT, require_tool

GEN_SCRIPT = Path(__file__).resolve().parent / "blender" / "gen_scene.py"
DEFAULT_ROOT = REPO_ROOT / "data" / "synthetic"


def render(
    out: Path,
    *,
    seed: int,
    frames: int,
    fps: int,
    width: int,
    height: int,
    engine: str,
    samples: int,
    actors: int,
    boxes: int,
    eval_views: int = 0,
    texture: float = 2.0,
    timeout: float = 3600,
) -> Path:
    blender = require_tool("blender")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    cmd = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(GEN_SCRIPT),
        "--",
        "--out",
        str(out),
        "--seed",
        str(seed),
        "--frames",
        str(frames),
        "--fps",
        str(fps),
        "--width",
        str(width),
        "--height",
        str(height),
        "--engine",
        engine,
        "--samples",
        str(samples),
        "--actors",
        str(actors),
        "--boxes",
        str(boxes),
        "--eval-views",
        str(eval_views),
        "--texture",
        str(texture),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if "WORLDREEL_GEN_OK" not in proc.stdout:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-25:]
        raise RuntimeError("Blender generator failed:\n" + "\n".join(tail))
    return out


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def backproject(
    depth: npt.NDArray[np.float32],
    k: dict[str, float],
    position: list[float],
    rotation: list[float],
) -> npt.NDArray[np.float64]:
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    valid = depth > 0
    z = depth[valid].astype(np.float64)
    cam = np.stack(
        [(u[valid] - k["cx"]) / k["fx"] * z, -(v[valid] - k["cy"]) / k["fy"] * z, -z], axis=1
    )
    return cam @ quat_to_matrix(rotation).T + np.asarray(position)


def project(
    points: npt.NDArray[np.float64],
    k: dict[str, float],
    position: list[float],
    rotation: list[float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """World points → (pixel coords (N, 2), planar depth (N,))."""
    cam = (points - np.asarray(position)) @ quat_to_matrix(rotation)
    z = -cam[:, 2]
    u = cam[:, 0] / z * k["fx"] + k["cx"]
    v = -cam[:, 1] / z * k["fy"] + k["cy"]
    return np.stack([u, v], axis=1), z


def check(scene_dir: Path, *, ground_tol: float = 0.05, reproj_tol: float = 0.05) -> CheckReport:
    """Geometric self-consistency of a rendered scene's ground truth.

    1. Nothing lies below the ground plane (y < -ground_tol) after back-projection.
    2. Cross-frame reprojection: static points from the first frame, projected into the last
       frame's camera, land on that frame's depth (median relative error < reproj_tol).
    3. Every actor track has one sample per frame and unit quaternions, and each actor faces
       the way it moves.
    """
    gt: dict[str, Any] = json.loads((scene_dir / "gt.json").read_text(encoding="utf-8"))
    report = CheckReport()
    k = gt["intrinsics"]
    frames = gt["frames"]
    if not frames:
        report.errors.append("no frames")
        return report
    for f in frames:
        for key in ("rgb", "depth"):
            if not (scene_dir / f[key]).exists():
                report.errors.append(f"missing {f[key]}")
    if report.errors:
        return report

    first, last = frames[0], frames[-1]
    d0 = np.load(scene_dir / first["depth"])
    pts0 = backproject(d0, k, first["camera"]["position"], first["camera"]["rotation"])
    below = float((pts0[:, 1] < -ground_tol).mean()) if len(pts0) else 0.0
    report.stats["valid_depth_fraction"] = float((d0 > 0).mean())
    report.stats["below_ground_fraction"] = below
    report.stats["ground_fraction"] = (
        float((np.abs(pts0[:, 1]) < ground_tol).mean()) if len(pts0) else 0.0
    )
    if below > 0.001:
        report.errors.append(f"{below:.1%} of back-projected points lie below the ground plane")

    # Ground points only: actors move between frames, the ground does not.
    ground = pts0[np.abs(pts0[:, 1]) < ground_tol]
    targets = [] if gt.get("probe") or len(frames) < 2 else [("last frame", last)]
    targets += [(f"eval view {v['id']}", v) for v in gt.get("eval_views", [])]
    for label, target in targets:
        d1 = np.load(scene_dir / target["depth"])
        uv, z = project(ground, k, target["camera"]["position"], target["camera"]["rotation"])
        h, w = d1.shape
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inside = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        observed = d1[vi[inside], ui[inside]]
        visible = observed > 0
        rel = np.abs(observed[visible] - z[inside][visible]) / z[inside][visible]
        # Occlusion by boxes/actors makes some points disagree; the median must still be tight.
        median = float(np.median(rel)) if rel.size else math.inf
        key = "reprojection" if label == "last frame" else f"reprojection_{target['id']}"
        report.stats[f"{key}_median_rel_error"] = median
        report.stats[f"{key}_points"] = float(rel.size)
        if rel.size < 100:
            report.errors.append(f"only {rel.size} ground points reproject into the {label}")
        elif median > reproj_tol:
            report.errors.append(
                f"{label}: reprojection median error {median:.1%} > {reproj_tol:.0%}"
            )

    for actor in gt.get("actors", []):
        tr = actor["track"]
        if len(tr["times"]) != len(frames):
            report.errors.append(
                f"actor {actor['id']}: {len(tr['times'])} samples for {len(frames)} frames"
            )
        for q in tr["rotations"]:
            if abs(math.sqrt(sum(c * c for c in q)) - 1) > 1e-4:
                report.errors.append(f"actor {actor['id']}: non-unit quaternion")
                break
        # Actors face their local -Z (spec §7); that must agree with the direction of motion.
        pos = np.asarray(tr["positions"], dtype=np.float64)
        if len(pos) > 1:
            vel = np.diff(pos, axis=0)
            fwd = np.stack(
                [quat_to_matrix(q) @ np.array([0.0, 0.0, -1.0]) for q in tr["rotations"][:-1]]
            )
            speed = np.linalg.norm(vel, axis=1)
            moving = speed > 1e-6
            cos = np.sum(fwd[moving] * vel[moving], axis=1) / speed[moving]
            facing = float(cos.mean()) if cos.size else 1.0
            report.stats[f"actor_{actor['id']}_facing_cos"] = facing
            if facing < 0.95:
                report.errors.append(
                    f"actor {actor['id']}: faces away from its motion (mean cos {facing:.2f})"
                )
    return report


def to_inbox(scene_dir: Path, inbox: Path) -> Path:
    """Encode the RGB frames as an MP4 and drop it, with its info sidecar, into an inbox."""
    ffmpeg = require_tool("ffmpeg")
    gt = json.loads((scene_dir / "gt.json").read_text(encoding="utf-8"))
    inbox.mkdir(parents=True, exist_ok=True)
    name = f"synthetic_seed{gt['seed']:04d}.mp4"
    target = inbox / name
    if target.exists():
        raise FileExistsError(target)
    tmp = scene_dir / name
    subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-framerate",
            str(gt["fps"]),
            "-i",
            str(scene_dir / "rgb" / "%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(tmp),
        ],
        check=True,
    )
    shutil.move(str(tmp), str(target))
    target.with_name(name + ".info.json").write_text(
        json.dumps(
            {
                "source": (
                    f"WorldReel Blender generator {gt['generator_version']}, seed {gt['seed']}"
                ),
                "license": "CC0-1.0",
                "upload_rights": "synthetic",
                "notes": f"ground truth in {scene_dir}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.synth")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--frames", type=int, default=24)
    r.add_argument("--fps", type=int, default=12)
    r.add_argument("--width", type=int, default=320)
    r.add_argument("--height", type=int, default=240)
    r.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    r.add_argument("--samples", type=int, default=16)
    r.add_argument("--actors", type=int, default=2)
    r.add_argument("--boxes", type=int, default=5)
    r.add_argument("--eval-views", type=int, default=0)
    r.add_argument("--texture", type=float, default=2.0)
    c = sub.add_parser("check")
    c.add_argument("scene", type=Path)
    i = sub.add_parser("to-inbox")
    i.add_argument("scene", type=Path)
    i.add_argument("--inbox", type=Path, default=REPO_ROOT / "data" / "inbox")
    args = parser.parse_args(argv)

    if args.cmd == "render":
        out = args.out or DEFAULT_ROOT / f"seed{args.seed:04d}"
        render(
            out,
            seed=args.seed,
            frames=args.frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
            engine=args.engine,
            samples=args.samples,
            actors=args.actors,
            boxes=args.boxes,
            eval_views=args.eval_views,
            texture=args.texture,
        )
        report = check(out)
        print(f"rendered {out}")
        args.scene = out
    elif args.cmd == "to-inbox":
        print(f"wrote {to_inbox(args.scene, args.inbox)}")
        return 0
    else:
        report = check(args.scene)
    for key, value in report.stats.items():
        print(f"  {key}: {value:.4f}")
    for e in report.errors:
        print(f"  ERROR: {e}")
    print("ground truth consistent" if report.ok else "ground truth INCONSISTENT")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
