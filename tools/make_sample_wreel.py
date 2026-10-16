"""Build a small hand-made synthetic `.wreel` for tests and the player.

Usage: ``python -m tools.make_sample_wreel [--out player/public/samples/courtyard.wreel]``

The scene: a 20 m courtyard floor, a wall with an unseen ("guessed") area behind it, a few
pillars (one side "inferred"), and two walking people (synthetic identities). Everything is
generated from a seed, so the same seed, key and timestamp give a byte-identical file.
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from formats.common import dump_json, load_or_create_signing_key
from formats.wreel import FORMAT_VERSION, PROVENANCE_CLASSES, PointCloud, write_wreel

GENERATOR_VERSION = "0.1.0"
DEFAULT_OUT = Path("player/public/samples/courtyard.wreel")
DURATION = 12.0

OBSERVED, INFERRED, GUESSED = (
    PROVENANCE_CLASSES.index(c) for c in ("observed", "inferred", "guessed")
)
Vec = tuple[float, float, float]
Quat = list[float]


# --------------------------------------------------------------------------- math helpers


def quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def yaw_quat(yaw: float) -> Quat:
    return [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]


def look_quat(eye: Vec, target: Vec) -> Quat:
    """Rotation that points the -Z forward axis from `eye` at `target` (Y up, no roll)."""
    fx, fy, fz = (t - e for t, e in zip(target, eye, strict=True))
    norm = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / norm, fy / norm, fz / norm
    yaw = math.atan2(-fx, -fz)
    pitch = math.asin(max(-1.0, min(1.0, fy)))
    q = quat_mul(yaw_quat(yaw), [math.sin(pitch / 2), 0.0, 0.0, math.cos(pitch / 2)])
    return [round(c, 6) for c in q]


# --------------------------------------------------------------------------- geometry


class CloudBuilder:
    def __init__(self) -> None:
        self.parts: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8], int, float]] = []

    def add(
        self, pts: npt.NDArray[np.float64], rgb: tuple[int, int, int], cls: int, conf: float
    ) -> None:
        colors = np.tile(np.array(rgb, dtype=np.uint8), (len(pts), 1))
        self.parts.append((pts.astype(np.float32), colors, cls, conf))

    def add_colored(
        self, pts: npt.NDArray[np.float64], colors: npt.NDArray[np.uint8], cls: int, conf: float
    ) -> None:
        self.parts.append((pts.astype(np.float32), colors, cls, conf))

    def build(self) -> PointCloud:
        positions = np.concatenate([p for p, _, _, _ in self.parts])
        colors = np.concatenate([c for _, c, _, _ in self.parts])
        cls = np.concatenate([np.full(len(p), k, dtype=np.uint8) for p, _, k, _ in self.parts])
        conf = np.concatenate([np.full(len(p), v, dtype=np.float32) for p, _, _, v in self.parts])
        return PointCloud(positions, colors, cls, conf)


def subsample(cloud: PointCloud, step: int) -> PointCloud:
    return PointCloud(
        cloud.positions[::step].copy(),
        cloud.colors[::step].copy(),
        cloud.cls[::step].copy(),
        cloud.confidence[::step].copy(),
    )


def box_surface(rng: np.random.Generator, lo: Vec, hi: Vec, n: int) -> npt.NDArray[np.float64]:
    """Points on the surface of an axis-aligned box."""
    lo_a, hi_a = np.array(lo), np.array(hi)
    pts = rng.uniform(lo_a, hi_a, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    side = rng.integers(0, 2, size=n)
    pts[np.arange(n), axis] = np.where(side == 1, hi_a[axis], lo_a[axis])
    return pts


def static_scene(rng: np.random.Generator) -> PointCloud:
    b = CloudBuilder()
    # Courtyard floor (observed in front of the wall, guessed behind it at z < -4).
    gx, gz = np.meshgrid(np.linspace(-10, 10, 121), np.linspace(-10, 10, 121))
    floor = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    floor += rng.normal(0, 0.01, floor.shape) * [1, 0, 1]
    checker = ((np.floor(floor[:, 0]) + np.floor(floor[:, 2])) % 2).astype(bool)
    floor_rgb = np.where(checker[:, None], [150, 145, 135], [120, 116, 108]).astype(np.uint8)
    hidden = floor[:, 2] < -4.2
    b.add_colored(floor[~hidden], floor_rgb[~hidden], OBSERVED, 1.0)
    # Behind the wall nothing was seen: fill with a guessed, low-confidence grey floor.
    b.add(floor[hidden], (128, 128, 128), GUESSED, 0.15)

    # Wall along z = -4 (front face observed).
    wall = box_surface(rng, (-6.0, 0.0, -4.2), (6.0, 3.0, -4.0), 5000)
    b.add(wall, (176, 92, 70), OBSERVED, 0.95)

    # Pillars: front faces observed, the far pillar's back side only inferred.
    for x, z, rgb in (
        (-5.0, 2.0, (70, 110, 160)),
        (5.0, 2.0, (70, 150, 110)),
        (0.0, -1.5, (200, 170, 80)),
    ):
        pts = box_surface(rng, (x - 0.4, 0.0, z - 0.4), (x + 0.4, 2.5, z + 0.4), 1500)
        back = pts[:, 2] < z - 0.2
        b.add(pts[~back], rgb, OBSERVED, 0.9)
        b.add(pts[back], rgb, INFERRED, 0.55)
    return b.build()


def person_cloud(rng: np.random.Generator, shirt: tuple[int, int, int]) -> PointCloud:
    """A stylised person in the actor's local frame (feet at origin, facing -Z)."""
    b = CloudBuilder()
    n = 900
    theta = rng.uniform(0, 2 * math.pi, n)
    h = rng.uniform(0.0, 1.45, n)
    radius = np.where(h < 0.85, 0.13, 0.2)
    body = np.stack([radius * np.cos(theta), h, radius * np.sin(theta)], axis=1)
    legs = h < 0.85
    b.add(body[legs], (40, 45, 60), OBSERVED, 0.9)
    b.add(body[~legs], shirt, OBSERVED, 0.9)
    head = rng.normal(0, 1, (300, 3))
    head = head / np.linalg.norm(head, axis=1, keepdims=True) * 0.12 + [0, 1.62, 0]
    b.add(head, (200, 170, 150), OBSERVED, 0.85)
    nose = rng.normal(0, 0.015, (40, 3)) + np.array([0, 1.62, -0.12])
    b.add(nose, (230, 60, 60), OBSERVED, 0.85)  # makes the facing direction visible
    return b.build()


# --------------------------------------------------------------------------- motion


def circle_track(radius: float, center: Vec, period: float, steps: int) -> dict[str, Any]:
    times, positions, rotations = [], [], []
    for i in range(steps + 1):
        t = DURATION * i / steps
        a = 2 * math.pi * t / period
        x, z = center[0] + radius * math.cos(a), center[2] + radius * math.sin(a)
        vx, vz = -math.sin(a), math.cos(a)  # counter-clockwise tangent
        times.append(round(t, 4))
        positions.append([round(x, 4), 0.0, round(z, 4)])
        rotations.append([round(c, 6) for c in yaw_quat(math.atan2(-vx, -vz))])
    return {"times": times, "positions": positions, "rotations": rotations}


def line_track(start: Vec, end: Vec, steps: int) -> dict[str, Any]:
    """Walk start → end for the first half, back for the second half."""
    times, positions, rotations = [], [], []
    for i in range(steps + 1):
        t = DURATION * i / steps
        phase = t / (DURATION / 2)
        u = phase if phase <= 1 else 2 - phase
        direction = 1 if phase <= 1 else -1
        x = start[0] + (end[0] - start[0]) * u
        z = start[2] + (end[2] - start[2]) * u
        vx, vz = (end[0] - start[0]) * direction, (end[2] - start[2]) * direction
        times.append(round(t, 4))
        positions.append([round(x, 4), 0.0, round(z, 4)])
        rotations.append([round(c, 6) for c in yaw_quat(math.atan2(-vx, -vz))])
    return {"times": times, "positions": positions, "rotations": rotations}


def director_camera() -> list[dict[str, Any]]:
    keys = []
    for i in range(7):
        t = DURATION * i / 6
        a = math.radians(-60 + 120 * i / 6)
        eye = (round(12 * math.sin(a), 4), round(4.5 - 1.5 * i / 6, 4), round(12 * math.cos(a), 4))
        keys.append(
            {
                "t": t,
                "position": list(eye),
                "rotation": look_quat(eye, (0.0, 1.0, 0.0)),
                "fov_deg": 50 - i,
            }
        )
    return keys


# --------------------------------------------------------------------------- assembly


def layer(
    layer_id: str, cloud: PointCloud, prefix: str, point_size: float
) -> tuple[dict[str, Any], dict[str, bytes]]:
    coarse = subsample(cloud, 8)
    lods: list[dict[str, Any]] = [
        {"level": 0, "path": f"{prefix}/lod0.pts", "count": len(coarse)},
        {"level": 1, "path": f"{prefix}/lod1.pts", "count": len(cloud)},
    ]
    entries = {lods[0]["path"]: coarse.pack(), lods[1]["path"]: cloud.pack()}
    return {"id": layer_id, "kind": "points", "point_size": point_size, "lods": lods}, entries


def build_sample(
    seed: int = 7, created_at: str | None = None
) -> tuple[dict[str, Any], dict[str, bytes]]:
    rng = np.random.default_rng(seed)
    entries: dict[str, bytes] = {}

    static_layer, data = layer("static", static_scene(rng), "scene/static", 0.08)
    entries.update(data)

    actors: list[dict[str, Any]] = []
    walks = {
        "a1": ("Walker A", (60, 120, 200), circle_track(3.0, (0.0, 0.0, 3.0), 12.0, 96), 101),
        "a2": ("Walker B", (210, 90, 60), line_track((-6.0, 0.0, 6.0), (6.0, 0.0, 6.0), 96), 202),
    }
    for actor_id, (label, shirt, track, identity_seed) in walks.items():
        geo, data = layer("body", person_cloud(rng, shirt), f"actors/{actor_id}", 0.035)
        entries.update(data)
        track_path = f"actors/{actor_id}/track.json"
        entries[track_path] = dump_json(track)
        actors.append(
            {
                "id": actor_id,
                "type": "person",
                "label": label,
                "identity": {"policy": "synthetic", "identity_seed": identity_seed},
                "geometry": geo,
                "eye_height": 1.62,
                "track": {"path": track_path},
            }
        )

    footsteps = [
        {
            "t": round(t, 2),
            "type": "footstep",
            "actor_id": "a1",
            "attributes": {"surface": "stone", "wet": False, "footwear": "sneaker", "speed": 1.6},
        }
        for t in np.arange(0.25, DURATION, 0.5)
    ]

    manifest: dict[str, Any] = {
        "format": "wreel",
        "format_version": FORMAT_VERSION,
        "title": "Courtyard (synthetic sample)",
        "duration": DURATION,
        "provenance": {
            "ai_label": "AI-generated",
            "notice": "Synthetic test scene generated by WorldReel. Not a recording.",
            "tool": "worldreel",
            "tool_version": GENERATOR_VERSION,
            "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "source": {"kind": "synthetic"},
            "identity_policy": "synthetic-only",
            "rights": {
                "upload_rights": "synthetic",
                "license": "CC0-1.0",
                "takedown": {"status": "none"},
            },
        },
        "generation": {
            "seeds": {"global": seed},
            "stages": [{"name": "tools.make_sample_wreel", "version": GENERATOR_VERSION}],
            "models": [],
        },
        "scene": {
            "bounds": {"min": [-10.0, 0.0, -10.0], "max": [10.0, 3.0, 10.0]},
            "background": [0.62, 0.72, 0.85],
            "lighting": {
                "ambient": {"color": [1.0, 1.0, 1.0], "intensity": 0.35},
                "lights": [
                    {
                        "id": "sun",
                        "type": "directional",
                        "color": [1.0, 0.96, 0.9],
                        "intensity": 1.0,
                        "direction": [-0.4, -1.0, -0.3],
                    }
                ],
            },
            "layers": [static_layer],
        },
        "actors": actors,
        "cameras": [
            {
                "id": "director",
                "name": "Director's cut",
                "kind": "director",
                "keyframes": director_camera(),
            }
        ],
        "pov_presets": [
            {
                "id": "directors_cut",
                "name": "Director's cut",
                "mode": "directors-cut",
                "camera_id": "director",
            },
            {"id": "free", "name": "Free camera", "mode": "free"},
            {
                "id": "follow_a1",
                "name": "Follow Walker A",
                "mode": "third-person",
                "actor_id": "a1",
                "offset": [0.0, 2.2, 4.5],
            },
            {
                "id": "follow_a2",
                "name": "Follow Walker B",
                "mode": "third-person",
                "actor_id": "a2",
                "offset": [0.0, 2.2, 4.5],
            },
            {
                "id": "eyes_a1",
                "name": "Walker A's eyes",
                "mode": "first-person",
                "actor_id": "a1",
                "comfort": True,
                "fov_deg": 60,
            },
            {"id": "overhead", "name": "Overhead", "mode": "overhead", "height": 22.0},
        ],
        "regions": [
            {
                "id": "behind_wall",
                "bounds": {"min": [-10.0, 0.0, -10.0], "max": [10.0, 3.0, -4.2]},
                "class": "guessed",
                "confidence": 0.15,
                "sources": [
                    {"kind": "model", "model": "flat-fill-baseline", "version": GENERATOR_VERSION}
                ],
            },
            {
                "id": "pillar_backs",
                "bounds": {"min": [-5.4, 0.0, -1.9], "max": [5.4, 2.5, 1.8]},
                "class": "inferred",
                "confidence": 0.55,
                "sources": [
                    {"kind": "model", "model": "box-symmetry", "version": GENERATOR_VERSION}
                ],
            },
        ],
        "audio": {
            "tracks": [],
            "events": footsteps,
            "mood": [{"t": 0.0, "tension": 2}, {"t": 6.0, "tension": 6}],
            "mix": {"ducking": True, "loudness_target_lufs": -16},
        },
        "interaction": {
            "rules": [
                {
                    "id": "tension_rise",
                    "trigger": {"type": "time", "t": 6.0},
                    "action": {"type": "set_tension", "value": 6},
                }
            ],
            "branches": [],
        },
        "files": {},
    }
    return manifest, entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the synthetic sample .wreel")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--created-at", default=None, help="fixed RFC 3339 timestamp for reproducible output"
    )
    args = parser.parse_args(argv)
    manifest, entries = build_sample(args.seed, args.created_at)
    write_wreel(args.out, manifest, entries, load_or_create_signing_key())
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
