"""Stage `generate` (real): build a `.wreel` world from a `.wscript` scene.

Pipeline: places (A-9) -> avatars (character bible + chosen portrait) -> blocking -> camera ->
audio -> package. Everything carries `generated` provenance: a generated world is made, not
recorded. The heavy generation (portraits, voices) happens upstream on Kaggle and is consumed
here as files, so packaging needs no GPU.

Usage as a CLI: ``python -m stages.generate.real --config generate.toml`` writes one `.wreel`.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast

import cv2
import numpy as np

from formats.common import dump_json, load_or_create_signing_key, sha256_hex
from formats.wreel import FORMAT_VERSION, PROVENANCE_CLASSES, PointCloud, write_wreel
from formats.wscript import read_wscript
from pipeline.io import WORLD_WREEL
from pipeline.stage import SOURCE, Impl, StageContext, StageError
from stages.audio.events import footstep_audio
from stages.audio.stems import DEFAULT_TARGET_LUFS, render_stems, stem_tracks
from stages.generate.avatars import POINT_SIZE, Avatar, build_avatar
from stages.generate.bible import expand
from stages.generate.blocking import _statement_times, plan_blocking
from stages.generate.camera import plan_camera
from stages.generate.places import build_place

GENERATED = PROVENANCE_CLASSES.index("generated")
TOOL_VERSION = "0.1.0"
LOD0_STEP = 4

# Default portrait choice (D-8.10b: chosen per character after SFace distinctness).
DEFAULT_CHOICE = {"C01": "C01_5.png", "C02": "C02_3.png", "C03": "C03_0.png", "C04": "C04_0.png"}


def _portrait_face_box(path: Path, size: tuple[int, int]) -> tuple[float, float, float, float]:
    """YuNet face box (x, y, w, h) in portrait pixels; raises if no face."""
    det = cv2.FaceDetectorYN.create(
        str(Path("data/models/face_detection_yunet_2023mar.onnx")), "", size, 0.6
    )
    det.setInputSize(size)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        raise ValueError(f"no face in portrait {path}")
    f = faces[int(np.argmax(faces[:, -1]))]
    return (float(f[0]), float(f[1]), float(f[2]), float(f[3]))


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if w.getnchannels() > 1:
        data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return rate, data


def _write_wav_bytes(x: np.ndarray, rate: int) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767).astype("<i2")
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@dataclass
class WorldParts:
    manifest: dict[str, Any]
    entries: dict[str, bytes]


def build_world(
    script_dir: Path,
    scene_id: str,
    duration: float,
    timing: list[dict[str, Any]] | dict[str, Any] | None,
    portraits_dir: Path,
    choice: dict[str, str],
    seed: int,
    surface: str = "stone",
    weather: str = "rain",
    scene_audio: Path | None = None,
) -> WorldParts:
    script = read_wscript(script_dir)
    script_sha = sha256_hex((script_dir / "wscript.json").read_bytes())
    if scene_id not in script.scenes:
        raise StageError(f"scene {scene_id} not in script")
    statements = script.scenes[scene_id]
    scene_entry = script.scene_entry(scene_id)
    location = str(scene_entry["location"])
    if timing is None:
        timing = []
    elif isinstance(timing, dict):
        timing = cast(list[dict[str, Any]], timing.get(scene_id, []))
    # Keep only voiced lines that fall inside the world window, clipped to its end.
    timing = [
        dict(e, start=float(e["start"]), end=min(float(e["end"]), duration))
        for e in timing
        if float(e["start"]) < duration and float(e["end"]) > 0
    ]
    continuity = dict(scene_entry.get("continuity", {}))
    continuity["__location__"] = location
    continuity["characters"] = dict(continuity.get("characters", {}))

    # --- place (A-9) ---
    loc_entry = script.header["dictionary"]["locations"][location]
    place = build_place(loc_entry, seed)

    # --- avatars (bible + chosen portrait + face box) ---
    chars = script.header["dictionary"]["characters"]
    scene_location = continuity.get("__location__", "")
    # A character is present if they start at the scene location or move there during the window.
    times = _statement_times(statements, timing, duration)
    arriving: dict[str, float] = {}
    for s in statements:
        from formats.wscript import Change

        if not isinstance(s, Change):
            continue
        cid = s.target
        if cid not in chars:
            continue
        if any(k == "location" and op == "=" and v == scene_location for k, op, v in s.ops):
            t0 = times.get(statements.index(s), duration)
            if t0 < duration:
                arriving[cid] = t0
    start_here = [
        cid
        for cid, st in continuity.get("characters", {}).items()
        if st.get("location") == scene_location and cid in chars
    ]
    present = sorted(set(start_here) | set(arriving))
    actors: list[dict[str, Any]] = []
    actor_entries: dict[str, bytes] = {}
    for cid in present:
        ident = expand(
            int(chars[cid]["identity"]["seed"]), str(chars[cid]["identity"].get("requested", ""))
        )
        portrait_file = portraits_dir / choice.get(cid, f"{cid}_0.png")
        raw = cv2.imread(str(portrait_file))
        if raw is None:
            raise FileNotFoundError(portrait_file)
        rgb = np.asarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        h, w = rgb.shape[:2]
        box = _portrait_face_box(portrait_file, (w, h))
        avatar: Avatar = build_avatar(ident, rgb, box)
        cloud = avatar.cloud
        coarse = PointCloud(
            cloud.positions[::LOD0_STEP].copy(),
            cloud.colors[::LOD0_STEP].copy(),
            cloud.cls[::LOD0_STEP].copy(),
            cloud.confidence[::LOD0_STEP].copy(),
        )
        geo_id = f"avatar_{cid}"
        actor_entries[f"{geo_id}/lod0.pts"] = coarse.pack()
        actor_entries[f"{geo_id}/lod1.pts"] = cloud.pack()
        actors.append(
            {
                "id": cid,
                "type": "person",
                "label": str(chars[cid]["name"]),
                "character_id": cid,
                "identity": {"policy": "synthetic", "identity_seed": ident.seed},
                "geometry": {
                    "id": geo_id,
                    "kind": "points",
                    "point_size": POINT_SIZE,
                    "lods": [
                        {"level": 0, "path": f"{geo_id}/lod0.pts", "count": len(coarse)},
                        {"level": 1, "path": f"{geo_id}/lod1.pts", "count": len(cloud)},
                    ],
                },
                "eye_height": round(float(avatar.eye_height), 4),
                "track": {"path": f"actors/{cid}/track.json"},
            }
        )

    # --- blocking -> tracks ---
    blocking = plan_blocking(statements, continuity, place, timing, duration, arriving=arriving)
    track_entries: dict[str, bytes] = {}
    for cid in blocking.present:
        track_entries[f"actors/{cid}/track.json"] = dump_json(blocking.tracks[cid])

    # --- camera (director's cut) ---
    camera_keyframes = plan_camera(statements, blocking, place, timing, duration)

    # --- audio ---
    audio_defaults = {"surface": surface, "weather": weather}
    steps, step_entries, _step_report = footstep_audio(
        actors, {a["id"]: blocking.tracks[a["id"]] for a in actors}, audio_defaults
    )
    entries: dict[str, bytes] = {}
    entries.update(actor_entries)
    entries.update(track_entries)
    entries.update(step_entries)

    # static scene layer (the place)
    static_coarse = PointCloud(
        place.cloud.positions[::LOD0_STEP].copy(),
        place.cloud.colors[::LOD0_STEP].copy(),
        place.cloud.cls[::LOD0_STEP].copy(),
        place.cloud.confidence[::LOD0_STEP].copy(),
    )
    entries["scene/static/lod0.pts"] = static_coarse.pack()
    entries["scene/static/lod1.pts"] = place.cloud.pack()

    # dialogue tracks: slice the voiced scene WAV per line, one track per character line
    dlg_tracks: list[dict[str, Any]] = []
    if scene_audio is not None and scene_audio.exists() and timing:
        rate, data = _read_wav(scene_audio)
        for k, entry in enumerate(timing):
            speaker = str(entry["speaker"])
            start, end = float(entry["start"]), float(entry["end"])
            if end <= start:
                continue
            i0, i1 = int(start * rate), int(end * rate)
            if i1 > len(data):
                i1 = len(data)
            if i0 >= i1:
                continue
            clip = data[i0:i1]
            # normalise to avoid clipping when summed
            peak = float(np.abs(clip).max()) or 1.0
            clip = clip / peak * 0.9
            path = f"audio/dlg_{speaker}_{k}.wav"
            entries[path] = _write_wav_bytes(clip, rate)
            dlg_tracks.append(
                {
                    "id": f"dlg_{speaker}_{k}",
                    "layer": "dialogue",
                    "character_id": speaker,
                    "path": path,
                    "media_type": "audio/wav",
                    "start": start,
                    "license": "synthetic",
                    "positional": True,
                    "actor_id": speaker,
                }
            )

    # layered stems: ambience, music, foley (footstep clips placed by events)
    stems = render_stems(
        duration,
        [e for e in steps],
        [{"t": 0.0, "tension": int(scene_entry.get("tension", 3))}],
        step_entries,
        weather=weather,
        seed=seed,
    )
    stem_list, stem_entries = stem_tracks(stems)
    entries.update(stem_entries)

    audio = {
        "tracks": dlg_tracks + stem_list,
        "events": steps,
        "mood": [{"t": 0.0, "tension": int(scene_entry.get("tension", 3))}],
        "mix": {"ducking": True, "loudness_target_lufs": DEFAULT_TARGET_LUFS},
    }

    # --- manifest ---
    lo = place.bounds["min"]
    hi = place.bounds["max"]
    manifest: dict[str, Any] = {
        "format": "wreel",
        "format_version": FORMAT_VERSION,
        "title": f"{script.header['title']} — {scene_entry['title']} ({scene_id})",
        "duration": round(float(duration), 4),
        "provenance": {
            "ai_label": "AI-generated",
            "notice": (
                "AI-generated world from a script. Not a recording; every region is generated."
            ),
            "tool": "worldreel",
            "tool_version": TOOL_VERSION,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": {"kind": "script", "sha256": script_sha},
            "identity_policy": "synthetic-only",
            "rights": {
                "upload_rights": "synthetic",
                "license": "CC0-1.0",
                "takedown": {"status": "none"},
            },
        },
        "generation": {
            "seeds": {"global": seed},
            "stages": [{"name": "generate", "version": TOOL_VERSION}],
            "models": [],
            "inputs": [{"role": "script", "sha256": script_sha}],
        },
        "scene": {
            "bounds": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
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
            "layers": [
                {
                    "id": "static",
                    "kind": "points",
                    "point_size": 0.08,
                    "lods": [
                        {"level": 0, "path": "scene/static/lod0.pts", "count": len(static_coarse)},
                        {"level": 1, "path": "scene/static/lod1.pts", "count": len(place.cloud)},
                    ],
                }
            ],
        },
        "actors": actors,
        "cameras": [
            {
                "id": "director",
                "name": "Director's cut",
                "kind": "director",
                "keyframes": camera_keyframes,
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
                "id": "overhead",
                "name": "Overhead",
                "mode": "overhead",
                "height": float(max(10.0, (np.array(hi) - np.array(lo)).max() * 1.5)),
            },
            *[
                {
                    "id": f"follow_{cid}",
                    "name": f"Follow {chars[cid]['name']}",
                    "mode": "third-person",
                    "actor_id": cid,
                    "offset": [0.0, 1.6, 3.0],
                }
                for cid in blocking.present
            ],
            *[
                {
                    "id": f"eyes_{cid}",
                    "name": f"{chars[cid]['name']}'s eyes",
                    "mode": "first-person",
                    "actor_id": cid,
                    "comfort": True,
                    "fov_deg": 60,
                }
                for cid in blocking.present
            ],
        ],
        "regions": [
            {
                "id": f"place_{location}",
                "bounds": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
                "class": "generated",
                "confidence": 1.0,
                "sources": [
                    {"kind": "model", "model": "procedural-places", "version": TOOL_VERSION}
                ],
            }
        ],
        "audio": audio,
        "interaction": {"rules": [], "branches": []},
        "files": {},
    }
    return WorldParts(manifest, entries)


class GenerateStage:
    name: ClassVar[str] = "generate"
    version: ClassVar[str] = TOOL_VERSION
    impl: ClassVar[Impl] = "real"
    depends_on: ClassVar[tuple[str, ...]] = (SOURCE,)

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.config
        script_dir = Path(str(cfg.get("script", ""))) if cfg.get("script") else None
        if script_dir is None or not script_dir.exists():
            p = ctx.item.path
            script_dir = p if p.is_dir() else p.parent
        scene_id = str(cfg.get("scene_id", ""))
        duration = float(cfg.get("duration", 0.0) or 0.0)
        portraits_dir = Path(str(cfg.get("portraits", "")))
        choice = {k: str(v) for k, v in (cfg.get("choice") or DEFAULT_CHOICE).items()}
        seed = int(ctx.config.get("seed", ctx.seed))
        surface = str(cfg.get("surface", "stone"))
        weather = str(cfg.get("weather", "rain"))
        scene_audio = Path(cfg["scene_audio"]) if cfg.get("scene_audio") else None
        timing_path = Path(cfg["timing"]) if cfg.get("timing") else None
        timing: list[dict[str, Any]] = []
        if timing_path and timing_path.exists():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))

        if duration <= 0 and timing:
            duration = max((float(e["end"]) for e in timing), default=10.0)
        if duration <= 0:
            duration = 10.0

        parts = build_world(
            script_dir,
            scene_id,
            duration,
            timing,
            portraits_dir,
            choice,
            seed,
            surface=surface,
            weather=weather,
            scene_audio=scene_audio,
        )
        key_path = cfg.get("signing_key")
        key = load_or_create_signing_key(Path(key_path) if key_path else None)
        write_wreel(ctx.out / WORLD_WREEL, parts.manifest, parts.entries, key)
