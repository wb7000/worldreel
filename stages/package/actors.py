"""Package tracked people as `.wreel` actors (spec §7) with SYNTHETIC avatars.

Brief §5/§11: people reconstructed from footage are never shown with their real appearance by
default. Each person becomes a generic stylised figure (no face, seeded neutral colours, scaled
to the estimated height) that moves along the person's lifted 3D track. Only the motion comes
from the video. The identity seed is stored so the file regenerates identically.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from formats.common import dump_json
from formats.wreel import PointCloud
from pipeline.io import read_json
from stages.actors.common import read_tracks
from stages.actors.lift import Plane, fit_ground_plane, lift_track
from tools.make_sample_wreel import person_cloud

F64 = npt.NDArray[np.float64]
REFERENCE_HEIGHT = 1.75  # height of tools.make_sample_wreel.person_cloud
MIN_MASK_FRACTION = 0.0005  # ignore masks smaller than this fraction of the frame
# Neutral, non-identifying clothing colours for avatars.
PALETTE = [
    (70, 110, 160),
    (160, 90, 70),
    (80, 140, 90),
    (150, 130, 60),
    (110, 90, 150),
    (90, 90, 90),
]


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _align_up(normal: F64) -> list[float]:
    """Quaternion rotating +Y onto the floor normal (so avatars stand on tilted floors)."""
    up = np.array([0.0, 1.0, 0.0])
    axis = np.cross(up, normal)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    angle = math.atan2(s, float(up @ normal))
    axis = axis / s * math.sin(angle / 2)
    return [float(axis[0]), float(axis[1]), float(axis[2]), math.cos(angle / 2)]


def avatar(seed: int, height: float) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = person_cloud(rng, PALETTE[seed % len(PALETTE)])
    scale = np.float32(height / REFERENCE_HEIGHT)
    return PointCloud(cloud.positions * scale, cloud.colors, cloud.cls, cloud.confidence)


def package_actors(
    actors_dir: Path, cam: dict[str, Any], static_points: F64, seed: int
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]], dict[str, Any]]:
    """Returns (actor entries, file entries, POV presets, report)."""
    meta, labels = read_tracks(actors_dir)
    report: dict[str, Any] = {"tracks": len(meta.get("tracks", [])), "packaged": 0, "skipped": []}
    if labels is None or not meta.get("tracks"):
        return [], {}, [], report
    poses = cam["poses"]
    if len(labels) != len(poses):
        raise ValueError(f"{len(labels)} mask frames for {len(poses)} camera poses")
    plane: Plane = fit_ground_plane(static_points)
    report["ground_normal"] = [round(float(x), 4) for x in plane.normal]
    k = cam["intrinsics"]
    times = [float(p["t"]) for p in poses]
    min_pixels = MIN_MASK_FRACTION * labels.shape[1] * labels.shape[2]
    up = _align_up(plane.normal)
    actors: list[dict[str, Any]] = []
    entries: dict[str, bytes] = {}
    presets: list[dict[str, Any]] = []
    for idx, track in enumerate(meta["tracks"], start=1):
        masks = {
            f: labels[f] == idx
            for f, px in zip(track["frames"], track["pixels"], strict=True)
            if poses[f].get("registered", True) and px >= min_pixels
        }
        lifted = lift_track(masks, poses, times, k, plane)
        if lifted is None:
            report["skipped"].append(track["id"])
            continue
        identity_seed = int(seed * 1000 + idx)
        # Heights are in reconstruction units (monocular scale is unknown), so no metric clip;
        # an impossible value falls back to the median of the other people.
        height = float(lifted.height)
        if not np.isfinite(height) or height <= 0:
            report["skipped"].append(track["id"])
            continue
        body = avatar(identity_seed, height)
        aid = track["id"]
        prefix = f"actors/{aid}"
        entries[f"{prefix}/lod0.pts"] = body.pack()
        entries[f"{prefix}/track.json"] = dump_json(
            {
                "times": lifted.times,
                "positions": lifted.positions,
                "rotations": [_quat_mul(up, q) for q in lifted.rotations],
            }
        )
        actors.append(
            {
                "id": aid,
                "type": "person",
                "label": f"Person {idx} (synthetic avatar)",
                "identity": {"policy": "synthetic-replacement", "identity_seed": identity_seed},
                "geometry": {
                    "id": "body",
                    "kind": "points",
                    "point_size": round(0.035 * height / REFERENCE_HEIGHT, 4),
                    "lods": [{"level": 0, "path": f"{prefix}/lod0.pts", "count": len(body)}],
                },
                "eye_height": round(0.925 * height, 3),
                "track": {"path": f"{prefix}/track.json"},
            }
        )
        presets += [
            {
                "id": f"follow_{aid}",
                "name": f"Follow person {idx}",
                "mode": "third-person",
                "actor_id": aid,
                "offset": [0.0, 0.6 * height + 0.6, 2.2 * height],
            },
            {
                "id": f"eyes_{aid}",
                "name": f"Person {idx}'s eyes",
                "mode": "first-person",
                "actor_id": aid,
                "comfort": True,
                "fov_deg": 60,
            },
        ]
        report["packaged"] += 1
        report.setdefault("heights", {})[aid] = round(height, 3)
    return actors, entries, presets, report


def camera_json(camera_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = read_json(camera_dir / "camera.json")
    return data
