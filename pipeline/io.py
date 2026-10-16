"""Small file helpers shared by stages. Stage outputs are the only interface between stages, so
their file names live here as constants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# frames
FRAMES_JSON = "frames.json"
FRAMES_DIR = "frames"
PROXY_NPY = "proxy.npy"  # uint8 (N, H, W, 3) low-res frames for cheap stages and mocks
AUDIO_WAV = "audio.wav"
# camera
CAMERA_JSON = "camera.json"
DEPTH_PROXY_NPY = "depth_proxy.npy"  # float32 (N, H, W) metres, proxy resolution
# reconstruct / fill
POINTS_PTS = "points.pts"
POINTS_JSON = "points.json"
REGIONS_JSON = "regions.json"
# actors
ACTORS_JSON = "actors.json"
# audio
AUDIO_JSON = "audio.json"
# package
WORLD_WREEL = "world.wreel"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_npy(path: Path, array: npt.NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)


def load_npy(path: Path) -> npt.NDArray[Any]:
    array: npt.NDArray[Any] = np.load(path, allow_pickle=False)
    return array
