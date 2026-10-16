"""Test media helpers: tiny real videos when ffmpeg is installed."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.toolpaths import find_tool

HAS_FFMPEG = find_tool("ffmpeg") is not None and find_tool("ffprobe") is not None
HAS_BLENDER = find_tool("blender") is not None


def make_clip(
    path: Path, *, seconds: float = 2.0, pattern: str = "testsrc", audio: bool = True
) -> Path:
    ffmpeg = find_tool("ffmpeg")
    assert ffmpeg is not None
    cmd = [
        str(ffmpeg),
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"{pattern}=duration={seconds}:size=96x72:rate=10",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path
