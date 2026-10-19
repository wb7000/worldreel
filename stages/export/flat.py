"""Flat MP4 export of one camera path (brief: "a normal MP4 of one chosen camera path").

Frames come from the offline renderer (`stages.export.render`), audio is the loudness-normalised
mix of the world's stems. The AI label (spec §4) is kept with the pixels: written to the MP4
metadata and burned in as a small badge in the top-left corner of every frame, so a shared clip
cannot lose it.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from formats.wreel import Wreel
from stages.audio.stems import DEFAULT_TARGET_LUFS, mixdown
from stages.audio.synth import wav_samples, write_wav
from stages.export.render import Frame, camera_at, prepare, render
from tools.toolpaths import find_tool


def frames(
    world: Wreel, camera_id: str, fps: float, width: int, height: int, fog_strength: float = 1.0
) -> Iterator[tuple[float, Frame]]:
    scene = prepare(world)
    n = max(1, round(float(world.manifest["duration"]) * fps))
    for i in range(n):
        t = i / fps
        yield (
            t,
            render(scene, camera_at(world.manifest, camera_id, t), t, width, height, fog_strength),
        )


def burn_label(rgb: np.ndarray, text: str) -> np.ndarray:
    """Small dark badge with the label, top-left (scales with the frame height)."""
    out = rgb.copy()
    scale = max(0.35, out.shape[0] / 1000)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = int(6 * scale) + 2
    x0, y0 = pad, pad
    x1, y1 = x0 + tw + 2 * pad, y0 + th + base + 2 * pad
    roi = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (roi * 0.35).astype(np.uint8)
    cv2.putText(
        out,
        text,
        (x0 + pad, y1 - pad - base),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 200, 120),
        1,
        cv2.LINE_AA,
    )
    return out


def mix_wav(world: Wreel, path: Path) -> bool:
    """Write the world's mix to `path`; False when the world has no audio tracks."""
    stems = {
        t["layer"]: wav_samples(world.entries[t["path"]]) for t in world.manifest["audio"]["tracks"]
    }
    if not stems:
        return False
    mix_cfg = world.manifest["audio"].get("mix", {})
    target = float(mix_cfg.get("loudness_target_lufs", DEFAULT_TARGET_LUFS))
    write_wav(path, mixdown(stems, target, bool(mix_cfg.get("ducking", True))))
    return True


def export_mp4(
    world: Wreel,
    out: Path,
    camera_id: str,
    *,
    fps: float = 24.0,
    width: int = 854,
    height: int = 480,
    crf: int = 20,
    on_frame: Callable[[int, Frame], None] | None = None,
) -> dict[str, Any]:
    """Render `camera_id` and encode H.264 + AAC. `on_frame` sees every rendered frame (the
    `.wedit` exporter uses it to write depth and mask layers in the same pass)."""
    ffmpeg = find_tool("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found (worldreel.local.toml [tools] ffmpeg)")
    prov = world.manifest["provenance"]
    label = f"{prov['ai_label'].upper()} - not a recording"
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "mix.wav"
        has_audio = mix_wav(world, audio)
        cmd = [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        if has_audio:
            cmd += ["-i", str(audio)]
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-metadata",
            f"comment={prov['notice']}",
            "-metadata",
            f"description={prov['ai_label']}: {prov['notice']}",
            "-movflags",
            "+faststart",
        ]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd.append(str(out))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        count = 0
        for i, (_, fr) in enumerate(frames(world, camera_id, fps, width, height)):
            if on_frame is not None:
                on_frame(i, fr)
            proc.stdin.write(burn_label(fr.rgb, label).tobytes())
            count += 1
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
    return {
        "frames": count,
        "fps": fps,
        "width": width,
        "height": height,
        "audio": has_audio,
        "label": label,
    }
