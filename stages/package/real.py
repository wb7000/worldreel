"""`package`: assemble upstream stage outputs into a signed `.wreel` (spec/wreel.md).

There is no separate mock: packaging is cheap, deterministic and needs no GPU, so the real
implementation also serves the mock pipeline.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from formats.common import load_or_create_signing_key
from formats.wreel import FORMAT_VERSION, PointCloud, write_wreel
from pipeline.io import (
    ACTORS_JSON,
    AUDIO_JSON,
    CAMERA_JSON,
    FRAMES_JSON,
    POINTS_JSON,
    POINTS_PTS,
    REGIONS_JSON,
    WORLD_WREEL,
    read_json,
)
from pipeline.stage import Impl, StageContext, StageError

TOOL_VERSION = "0.2.0"
LOD0_STEP = 4


def _concat(clouds: list[PointCloud]) -> PointCloud:
    return PointCloud(
        np.concatenate([c.positions for c in clouds]),
        np.concatenate([c.colors for c in clouds]),
        np.concatenate([c.cls for c in clouds]),
        np.concatenate([c.confidence for c in clouds]),
    )


def _subsample(c: PointCloud, step: int) -> PointCloud:
    return PointCloud(
        c.positions[::step].copy(),
        c.colors[::step].copy(),
        c.cls[::step].copy(),
        c.confidence[::step].copy(),
    )


class PackageStage:
    name: ClassVar[str] = "package"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "real"
    depends_on: ClassVar[tuple[str, ...]] = (
        "frames",
        "camera",
        "reconstruct",
        "actors",
        "fill",
        "audio",
    )

    def run(self, ctx: StageContext) -> None:
        inp = ctx.inputs
        frames = read_json(inp["frames"] / FRAMES_JSON)
        cam = read_json(inp["camera"] / CAMERA_JSON)
        actors = read_json(inp["actors"] / ACTORS_JSON)["actors"]
        if actors:
            raise StageError("packaging tracked actors is implemented in Phase 3")
        audio = read_json(inp["audio"] / AUDIO_JSON)
        regions = read_json(inp["fill"] / REGIONS_JSON)["regions"]
        point_size = float(read_json(inp["reconstruct"] / POINTS_JSON).get("point_size", 0.05))

        cloud = _concat(
            [
                PointCloud.unpack((inp["reconstruct"] / POINTS_PTS).read_bytes()),
                PointCloud.unpack((inp["fill"] / POINTS_PTS).read_bytes()),
            ]
        )
        if len(cloud) == 0:
            raise StageError("no points to package")
        coarse = _subsample(cloud, LOD0_STEP)
        entries = {"scene/static/lod0.pts": coarse.pack(), "scene/static/lod1.pts": cloud.pack()}
        lo = cloud.positions.min(axis=0) - 0.5
        hi = cloud.positions.max(axis=0) + 0.5

        k = cam["intrinsics"]
        fov_y = math.degrees(2 * math.atan((k["height"] / 2) / k["fy"]))
        duration = max(float(frames["duration"]), max((p["t"] for p in cam["poses"]), default=0.0))
        item = ctx.item
        manifest: dict[str, Any] = {
            "format": "wreel",
            "format_version": FORMAT_VERSION,
            "title": str(ctx.config.get("title", f"Reconstruction of {item.id}")),
            "duration": duration,
            "provenance": {
                "ai_label": "AI-reconstructed",
                "notice": (
                    "AI reconstruction of a video. Not a recording; unseen areas are guessed."
                ),
                "tool": "worldreel",
                "tool_version": TOOL_VERSION,
                "created_at": str(
                    ctx.config.get("created_at") or datetime.now(UTC).isoformat(timespec="seconds")
                ),
                "source": {"kind": "video", "sha256": item.sha256},
                # No people are tracked yet; once actors exist, people get synthetic replacement.
                "identity_policy": "none",
                "rights": {
                    "upload_rights": item.upload_rights
                    if item.upload_rights in _RIGHTS
                    else "unknown",
                    **({"license": item.license} if item.license else {}),
                    "takedown": {"status": "none"},
                },
            },
            "generation": {
                "seeds": {"global": ctx.seed},
                "stages": [
                    {"name": name, "version": str(rec.get("version", "")), **_config_sha(rec)}
                    for name, rec in sorted(ctx.upstream.items())
                ]
                + [{"name": self.name, "version": self.version}],
                "models": [],
                "inputs": [{"role": "video", "sha256": item.sha256}],
            },
            "scene": {
                "bounds": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
                "background": [0.62, 0.72, 0.85],
                "lighting": {"ambient": {"color": [1.0, 1.0, 1.0], "intensity": 1.0}, "lights": []},
                "layers": [
                    {
                        "id": "static",
                        "kind": "points",
                        "point_size": point_size,
                        "lods": [
                            {"level": 0, "path": "scene/static/lod0.pts", "count": len(coarse)},
                            {"level": 1, "path": "scene/static/lod1.pts", "count": len(cloud)},
                        ],
                    }
                ],
            },
            "actors": [],
            "cameras": [
                {
                    "id": "director",
                    "name": "Original camera",
                    "kind": "director",
                    "keyframes": [
                        {
                            "t": p["t"],
                            "position": p["position"],
                            "rotation": p["rotation"],
                            "fov_deg": round(fov_y, 3),
                        }
                        for p in cam["poses"]
                    ],
                }
            ],
            "pov_presets": [
                {
                    "id": "directors_cut",
                    "name": "Original camera",
                    "mode": "directors-cut",
                    "camera_id": "director",
                },
                {"id": "free", "name": "Free camera", "mode": "free"},
                {
                    "id": "overhead",
                    "name": "Overhead",
                    "mode": "overhead",
                    "height": float(max(10.0, (hi - lo).max() * 1.5)),
                },
            ],
            "regions": regions,
            "audio": {"tracks": audio["tracks"], "events": audio["events"], "mood": audio["mood"]},
            "interaction": {"rules": [], "branches": []},
            "files": {},
        }
        key_path = ctx.config.get("signing_key")
        key = load_or_create_signing_key(Path(key_path) if key_path else None)
        write_wreel(ctx.out / WORLD_WREEL, manifest, entries, key)


_RIGHTS = {"unknown", "owner", "licensed", "public-domain", "synthetic"}


def _config_sha(record: dict[str, Any] | Any) -> dict[str, str]:
    sha = record.get("config_sha256") if isinstance(record, dict) else None
    return {"config_sha256": sha} if sha else {}
