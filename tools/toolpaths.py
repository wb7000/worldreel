"""Locate external programs (ffmpeg, ffprobe, Blender) without assuming they are on PATH.

Lookup order for a tool `name`:
1. environment variable `WORLDREEL_<NAME>` (e.g. `WORLDREEL_BLENDER`)
2. `[tools]` table in `worldreel.local.toml` at the repository root (git-ignored;
   copy `worldreel.example.toml`)
3. `PATH`
4. well-known install locations for this platform
"""

from __future__ import annotations

import os
import shutil
import sys
import tomllib
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG = REPO_ROOT / "worldreel.local.toml"

_KNOWN_WINDOWS: dict[str, list[str]] = {
    "ffmpeg": [r"D:\ffmpeg\bin\ffmpeg.exe", r"C:\ffmpeg\bin\ffmpeg.exe"],
    "ffprobe": [r"D:\ffmpeg\bin\ffprobe.exe", r"C:\ffmpeg\bin\ffprobe.exe"],
    "blender": [
        rf"C:\Program Files\Blender Foundation\Blender {v}\blender.exe"
        for v in ("4.2", "4.1", "4.0", "3.6")
    ],
}
_KNOWN_OTHER: dict[str, list[str]] = {
    "blender": ["/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender"],
}


def _local_config() -> dict[str, str]:
    if not LOCAL_CONFIG.exists():
        return {}
    with LOCAL_CONFIG.open("rb") as fh:
        tools = tomllib.load(fh).get("tools", {})
    return {str(k): str(v) for k, v in tools.items()}


@cache
def find_tool(name: str) -> Path | None:
    candidates: list[str] = []
    env = os.environ.get(f"WORLDREEL_{name.upper()}")
    if env:
        candidates.append(env)
    configured = _local_config().get(name)
    if configured:
        candidates.append(configured)
    on_path = shutil.which(name)
    if on_path:
        candidates.append(on_path)
    known = _KNOWN_WINDOWS if sys.platform == "win32" else _KNOWN_OTHER
    candidates.extend(known.get(name, []))
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def require_tool(name: str) -> Path:
    path = find_tool(name)
    if path is None:
        raise FileNotFoundError(
            f"{name} not found. Set WORLDREEL_{name.upper()} or add it to the [tools] table in "
            f"{LOCAL_CONFIG.name} (see worldreel.example.toml)."
        )
    return path
