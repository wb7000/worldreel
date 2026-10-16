"""Real `frames` stage: ffprobe + ffmpeg decode, scene-cut detection, audio extraction."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from pipeline.io import AUDIO_WAV, FRAMES_DIR, FRAMES_JSON, PROXY_NPY, save_npy, write_json
from pipeline.stage import SOURCE, Impl, StageContext, StageError
from tools.toolpaths import find_tool

DEFAULTS: dict[str, Any] = {
    "fps": 5.0,  # frames extracted per second of video
    "max_width": 640,  # PNG frames
    "proxy_width": 64,  # proxy.npy frames for cheap stages and mocks
    "scene_threshold": 0.3,  # ffmpeg scene-change score for a cut
    "max_seconds": 600.0,  # refuse longer inputs in Phase 2
}


def _run(cmd: list[str], *, binary: bool = False) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(cmd, capture_output=True, check=True, text=not binary)
    except subprocess.CalledProcessError as exc:
        err = (
            exc.stderr
            if isinstance(exc.stderr, str)
            else (exc.stderr or b"").decode(errors="replace")
        )
        raise StageError(f"{Path(cmd[0]).name} failed: {err.strip()[-500:]}") from exc


def probe(ffprobe: Path, source: Path) -> dict[str, Any]:
    out = _run(
        [str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)]
    )
    info: dict[str, Any] = json.loads(out.stdout)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise StageError("no video stream found")
    num, _, den = str(video.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0.0)
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "source_fps": round(fps, 4),
        "duration": round(duration, 4),
        "codec": video.get("codec_name", ""),
        "has_audio": any(s.get("codec_type") == "audio" for s in info.get("streams", [])),
    }


def _even(x: float) -> int:
    return max(2, round(x / 2) * 2)


class FramesStage:
    name: ClassVar[str] = "frames"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "real"
    depends_on: ClassVar[tuple[str, ...]] = (SOURCE,)

    def run(self, ctx: StageContext) -> None:
        cfg = {**DEFAULTS, **ctx.config}
        ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
        if ffmpeg is None or ffprobe is None:
            raise StageError(
                "ffmpeg/ffprobe not found (see worldreel.example.toml), or use impl=mock"
            )
        src = ctx.item.path
        meta = probe(ffprobe, src)
        if meta["duration"] > cfg["max_seconds"]:
            raise StageError(
                f"video is {meta['duration']:.0f} s; limit is {cfg['max_seconds']:.0f} s"
            )
        fps = float(cfg["fps"])
        w, h = meta["width"], meta["height"]
        out_w = min(int(cfg["max_width"]), _even(w))
        out_h = _even(h * out_w / w)
        proxy_w = int(cfg["proxy_width"])
        proxy_h = _even(h * proxy_w / w)

        frames_dir = ctx.out / FRAMES_DIR
        frames_dir.mkdir()
        base = [str(ffmpeg), "-v", "error", "-nostdin", "-i", str(src)]
        _run([*base, "-vf", f"fps={fps},scale={out_w}:{out_h}", str(frames_dir / "%06d.png")])

        raw = _run(
            [
                *base,
                "-vf",
                f"fps={fps},scale={proxy_w}:{proxy_h}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            binary=True,
        ).stdout
        frame_bytes = proxy_w * proxy_h * 3
        n = len(raw) // frame_bytes
        proxy = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape(
            n, proxy_h, proxy_w, 3
        )
        save_npy(ctx.out / PROXY_NPY, proxy)

        cuts = self._scene_cuts(ffmpeg, src, float(cfg["scene_threshold"]))
        if meta["has_audio"]:
            _run([*base, "-vn", "-ac", "1", "-ar", "48000", str(ctx.out / AUDIO_WAV)])

        count = len(list(frames_dir.glob("*.png")))
        write_json(
            ctx.out / FRAMES_JSON,
            {
                **meta,
                "fps": fps,
                "count": count,
                "frame_width": out_w,
                "frame_height": out_h,
                "proxy_count": int(n),
                "proxy_width": proxy_w,
                "proxy_height": proxy_h,
                "cuts": cuts,
                "audio": AUDIO_WAV if meta["has_audio"] else None,
            },
        )
        ctx.log.info("extracted %d frames (%dx%d), %d cuts", count, out_w, out_h, len(cuts))

    @staticmethod
    def _scene_cuts(ffmpeg: Path, src: Path, threshold: float) -> list[float]:
        vf = f"select='gt(scene,{threshold})',showinfo"
        proc = subprocess.run(
            [str(ffmpeg), "-nostdin", "-i", str(src), "-vf", vf, "-f", "null", "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        return [round(float(t), 3) for t in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]
