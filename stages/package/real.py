"""`package`: assemble upstream stage outputs into a signed `.wreel` (spec/wreel.md).

There is no separate mock: packaging is cheap, deterministic and needs no GPU, so the real
implementation also serves the mock pipeline.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from formats.common import load_or_create_signing_key
from formats.wreel import FORMAT_VERSION, PointCloud, write_wreel
from pipeline.io import (
    AUDIO_JSON,
    CAMERA_JSON,
    FRAMES_JSON,
    POINTS_JSON,
    POINTS_PTS,
    REGIONS_JSON,
    WORLD_WREEL,
    read_json,
    write_json,
)
from pipeline.stage import Impl, StageContext, StageError
from stages.audio.events import footstep_audio
from stages.audio.stems import render_stems, stem_tracks
from stages.package.actors import package_actors

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
    version: ClassVar[str] = "0.7.0"
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

        actors, actor_entries, actor_presets, actor_report = package_actors(
            inp["actors"], cam, cloud.positions[cloud.cls == 0].astype(np.float64), ctx.seed
        )
        entries.update(actor_entries)
        if actor_report["tracks"]:
            ctx.log.info("actors: %s", actor_report)
            write_json(ctx.out / "actors_report.json", actor_report)

        # Event-driven footsteps (algorithm A-6, D-5.2): generated clips, scene-wide surface and
        # weather from config `audio` (e.g. {"surface": "gravel", "weather": "rain"}).
        tracks = {a["id"]: json.loads(entries[a["track"]["path"]]) for a in actors}
        audio_defaults = {k: v for k, v in (ctx.config.get("audio") or {}).items() if k != "mood"}
        steps, step_entries, step_report = footstep_audio(actors, tracks, audio_defaults)
        entries.update(step_entries)
        audio["events"] = sorted(audio["events"] + steps, key=lambda e: e["t"])
        if steps:
            ctx.log.info("footsteps: %s", step_report)

        k = cam["intrinsics"]
        fov_y = math.degrees(2 * math.atan((k["height"] / 2) / k["fy"]))
        duration = max(float(frames["duration"]), max((p["t"] for p in cam["poses"]), default=0.0))
        # Layered stems (D-5.3): ambience, mood music, foley; separate tracks, generated audio.
        audio_cfg = dict(ctx.config.get("audio") or {})
        if audio_cfg.get("mood"):
            audio["mood"] = audio_cfg["mood"]
        stems = render_stems(
            duration,
            audio["events"],
            audio["mood"],
            step_entries,
            weather=str(audio_cfg.get("weather", "dry")),
            seed=ctx.seed,
        )
        stem_list, stem_entries = stem_tracks(stems)
        entries.update(stem_entries)
        audio["tracks"] = audio["tracks"] + stem_list
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
                # People from footage are shown as synthetic avatars (brief §5/§11).
                "identity_policy": "synthetic-replacement" if actors else "none",
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
            "actors": actors,
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
                        # Frames the camera stage could not track carry a copied pose; leave
                        # them out so the path only contains real estimates.
                        if p.get("registered", True)
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
                *actor_presets,
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
