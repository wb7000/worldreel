"""`.wedit` export (spec/wedit.md 1.1): a bundle of standard files a video editor can open.

name.wedit/
  wedit.json, signature.json, PROVENANCE.txt
  video/main.mp4                 rendered camera path, H.264 + AAC mix, AI label burned in
  stems/<layer>.wav              one per audio layer of the world
  layers/depth/%06d.png          16-bit depth (scale in wedit.json `encoding`), 0 = nothing
  layers/mask_<actor>/%06d.png   8-bit masks, 255 where the actor is visible
  timeline/main.otio             OpenTimelineIO (reference timeline)
  timeline/main.mlt              MLT XML (opens in Shotcut; layer tracks present but hidden)
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import opentimelineio as otio

from formats.common import load_or_create_signing_key
from formats.wedit import FORMAT_VERSION, write_wedit
from formats.wreel import Wreel
from stages.audio.synth import SAMPLE_RATE, wav_samples, write_wav
from stages.export.flat import export_mp4
from stages.export.render import Frame

FRAME_PATTERN = "%06d.png"


def _frame_name(i: int) -> str:
    return f"{i:06d}.png"


class _LayerWriter:
    """Writes depth and per-actor masks while the video renders (one render pass)."""

    def __init__(self, root: Path, actor_ids: list[str]) -> None:
        self.root = root
        self.actor_ids = actor_ids
        self.max_depth = 0.0
        self.depths: list[np.ndarray] = []
        (root / "layers" / "depth").mkdir(parents=True)
        for aid in actor_ids:
            (root / "layers" / f"mask_{aid}").mkdir(parents=True)

    def __call__(self, i: int, fr: Frame) -> None:
        self.depths.append(fr.depth)  # quantised at the end, once the range is known
        self.max_depth = max(self.max_depth, float(fr.depth.max()))
        for k, aid in enumerate(self.actor_ids, start=1):
            mask = np.where(fr.actor_ids == k, 255, 0).astype(np.uint8)
            cv2.imwrite(str(self.root / "layers" / f"mask_{aid}" / _frame_name(i)), mask)

    def finish(self) -> float:
        """Write the depth PNGs; returns scene units per count."""
        scale = self.max_depth / 65535.0 if self.max_depth > 0 else 1.0
        for i, d in enumerate(self.depths):
            q = np.clip(np.round(d / scale), 0, 65535).astype(np.uint16)
            cv2.imwrite(str(self.root / "layers" / "depth" / _frame_name(i)), q)
        return scale


def _otio_timeline(
    fps: float, frames: int, stems: list[dict[str, Any]], layers: list[dict[str, Any]]
) -> otio.schema.Timeline:
    rate = fps
    dur = otio.opentime.RationalTime(frames, rate)
    rng = otio.opentime.TimeRange(otio.opentime.RationalTime(0, rate), dur)
    tl = otio.schema.Timeline(name="WorldReel export")
    video = otio.schema.Track(name="V1 main", kind=otio.schema.TrackKind.Video)
    video.append(
        otio.schema.Clip(
            name="main",
            media_reference=otio.schema.ExternalReference(
                target_url="../video/main.mp4", available_range=rng
            ),
            source_range=rng,
        )
    )
    tl.tracks.append(video)
    for layer in layers:
        tr = otio.schema.Track(name=f"layer {layer['id']}", kind=otio.schema.TrackKind.Video)
        ref = otio.schema.ImageSequenceReference(
            target_url_base=f"../{layer['path']}",
            name_prefix="",
            name_suffix=".png",
            start_frame=0,
            frame_step=1,
            rate=rate,
            frame_zero_padding=6,
            available_range=rng,
        )
        tr.append(otio.schema.Clip(name=layer["id"], media_reference=ref, source_range=rng))
        tr.enabled = False  # layers are present for compositing, off by default
        tl.tracks.append(tr)
    for s in stems:
        tr = otio.schema.Track(name=f"A {s['layer']}", kind=otio.schema.TrackKind.Audio)
        tr.append(
            otio.schema.Clip(
                name=s["layer"],
                media_reference=otio.schema.ExternalReference(
                    target_url=f"../{s['path']}", available_range=rng
                ),
                source_range=rng,
            )
        )
        tl.tracks.append(tr)
    return tl


def _mlt_xml(
    fps: float,
    width: int,
    height: int,
    frames: int,
    stems: list[dict[str, Any]],
    layers: list[dict[str, Any]],
) -> str:
    """MLT XML: one producer and playlist per file; video main track first, hidden layer tracks,
    audio tracks mixed into the output."""
    out = frames - 1
    fr_num, fr_den = (round(fps * 1000), 1000) if abs(fps - round(fps)) > 1e-9 else (round(fps), 1)
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<mlt LC_NUMERIC="C" version="7.0.0" title="WorldReel export">',
        f'  <profile description="WorldReel {width}x{height} {fps:g} fps" '
        f'width="{width}" height="{height}" progressive="1" sample_aspect_num="1" '
        f'sample_aspect_den="1" display_aspect_num="{width}" display_aspect_den="{height}" '
        f'frame_rate_num="{fr_num}" frame_rate_den="{fr_den}" colorspace="709"/>',
    ]
    tracks: list[tuple[str, str]] = []  # (playlist id, hide attribute)

    def producer(pid: str, resource: str, extra: str = "") -> None:
        lines.append(f'  <producer id="{pid}" in="0" out="{out}">')
        lines.append(f'    <property name="resource">{resource}</property>{extra}')
        lines.append("  </producer>")

    # Paths are relative to the timeline file (timeline/), as MLT resolves them (D-6.5).
    producer("p_main", "../video/main.mp4")
    for layer in layers:
        producer(
            f"p_{layer['id']}",
            f"../{layer['path']}{FRAME_PATTERN}?begin=0",
            '\n    <property name="ttl">1</property>',
        )
    for s in stems:
        producer(f"p_{s['id']}", f"../{s['path']}")
    # The main clip's own (mixed) sound is muted: the stems are the timeline's audio, otherwise
    # the mix and the stems would play together (D-6.5).
    playlists = [("pl_main", "p_main", "audio")]
    playlists += [(f"pl_{layer['id']}", f"p_{layer['id']}", "video") for layer in layers]  # hidden
    playlists += [(f"pl_{s['id']}", f"p_{s['id']}", "video") for s in stems]  # audio only
    for plid, pid, hide in playlists:
        lines.append(f'  <playlist id="{plid}">')
        lines.append(f'    <entry producer="{pid}" in="0" out="{out}"/>')
        lines.append("  </playlist>")
        tracks.append((plid, hide))
    lines.append(f'  <tractor id="tractor0" in="0" out="{out}">')
    lines.append("    <multitrack>")
    for plid, hide in tracks:
        attr = f' hide="{hide}"' if hide else ""
        lines.append(f'      <track producer="{plid}"{attr}/>')
    lines.append("    </multitrack>")
    for k in range(1, len(tracks)):
        lines.append(f'    <transition id="mix{k}">')
        lines.append('      <property name="a_track">0</property>')
        lines.append(f'      <property name="b_track">{k}</property>')
        lines.append('      <property name="mlt_service">mix</property>')
        lines.append('      <property name="always_active">1</property>')
        lines.append('      <property name="sum">1</property>')
        lines.append("    </transition>")
    lines.append("  </tractor>")
    lines.append("</mlt>")
    return "\n".join(lines) + "\n"


def export_wedit(
    world: Wreel,
    world_bytes_sha256: str,
    out: Path,
    camera_id: str,
    *,
    fps: float = 24.0,
    width: int = 854,
    height: int = 480,
    as_zip: bool = False,
) -> dict[str, Any]:
    m = world.manifest
    actor_ids = [a["id"] for a in m.get("actors", [])]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "video").mkdir()
        layers_writer = _LayerWriter(root, actor_ids)
        video_info = export_mp4(
            world,
            root / "video" / "main.mp4",
            camera_id,
            fps=fps,
            width=width,
            height=height,
            on_frame=layers_writer,
        )
        depth_scale = layers_writer.finish()
        frames = int(video_info["frames"])

        stems: list[dict[str, Any]] = []
        (root / "stems").mkdir()
        for track in m["audio"]["tracks"]:
            samples = wav_samples(world.entries[track["path"]])
            path = f"stems/{track['layer']}.wav"
            write_wav(root / path, samples)
            stems.append(
                {
                    "id": f"stem_{track['layer']}",
                    "layer": track["layer"],
                    "path": path,
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                    "license": track["license"],
                }
            )

        layers: list[dict[str, Any]] = [
            {
                "id": "depth",
                "kind": "depth",
                "path": "layers/depth/",
                "pattern": FRAME_PATTERN,
                "encoding": (
                    f"png16; depth = value x {depth_scale:.6g} scene units along the view axis; "
                    "0 = no surface"
                ),
            }
        ]
        layers += [
            {
                "id": f"mask_{aid}",
                "kind": "mask",
                "path": f"layers/mask_{aid}/",
                "pattern": FRAME_PATTERN,
                "encoding": "png8; 255 = actor visible",
            }
            for aid in actor_ids
        ]
        (root / "timeline").mkdir()
        otio.adapters.write_to_file(
            _otio_timeline(fps, frames, stems, layers), str(root / "timeline" / "main.otio")
        )
        (root / "timeline" / "main.mlt").write_text(
            _mlt_xml(fps, width, height, frames, stems, layers), encoding="utf-8"
        )

        entries = {
            str(p.relative_to(root)).replace("\\", "/"): p.read_bytes()
            for p in root.rglob("*")
            if p.is_file()
        }
        manifest: dict[str, Any] = {
            "format": "wedit",
            "format_version": FORMAT_VERSION,
            "level": "flat",
            "provenance": m["provenance"],
            "source_wreel": {"sha256": world_bytes_sha256, "title": m.get("title", "")},
            "camera_path": camera_id,
            "video": {
                "path": "video/main.mp4",
                "fps": fps,
                "width": width,
                "height": height,
                "duration": frames / fps,
                "codec": "h264",
            },
            "stems": stems,
            "render_layers": layers,
            "timeline": [
                {"format": "otio", "path": "timeline/main.otio"},
                {"format": "mlt", "path": "timeline/main.mlt"},
            ],
            "changes": [],
            "files": {},
        }
        if out.exists():
            shutil.rmtree(out) if out.is_dir() else out.unlink()
        write_wedit(out, manifest, entries, load_or_create_signing_key(), as_zip=as_zip)
    return {
        "frames": frames,
        "stems": len(stems),
        "layers": len(layers),
        "files": len(entries),
        "depth_scale": depth_scale,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["export_wedit", "sha256_file"]
