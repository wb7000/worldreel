"""Files the `actors` stage writes and other stages read (2D person tracks).

    actors.json   {"backend": ..., "frame_count": N,
                   "tracks": [{"id", "label", "frames", "pixels"}]}
    masks.npz     key "labels": uint8 (N, H, W) at frame resolution; 0 = no person,
                  k = tracks[k-1]

Stages that build the *static* scene (camera, reconstruct, splat training) must ignore labelled
pixels: people move, so their pixels violate the static-scene assumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from pipeline.io import ACTORS_JSON, read_json, write_json

MASKS_NPZ = "masks.npz"


def write_tracks(
    out: Path, labels: npt.NDArray[np.uint8], backend: str, track_labels: list[str]
) -> None:
    tracks = []
    for k, label in enumerate(track_labels, start=1):
        frames = [int(f) for f in np.nonzero((labels == k).any(axis=(1, 2)))[0]]
        tracks.append(
            {
                "id": f"p{k}",
                "label": label,
                "frames": frames,
                "pixels": [int((labels[f] == k).sum()) for f in frames],
            }
        )
    np.savez_compressed(out / MASKS_NPZ, labels=labels.astype(np.uint8))
    write_json(
        out / ACTORS_JSON, {"backend": backend, "frame_count": len(labels), "tracks": tracks}
    )


def read_tracks(actors_dir: Path) -> tuple[dict[str, Any], npt.NDArray[np.uint8] | None]:
    meta: dict[str, Any] = read_json(actors_dir / ACTORS_JSON)
    path = actors_dir / MASKS_NPZ
    if not path.exists():
        return meta, None
    with np.load(path) as npz:
        labels: npt.NDArray[np.uint8] = npz["labels"]
    return meta, labels


def static_masks(
    actors_dir: Path | None, count: int, shape: tuple[int, int], dilate_px: int = 4
) -> list[npt.NDArray[np.bool_]] | None:
    """Per-frame boolean masks of pixels to ignore (people, dilated a little), resized to
    `shape` (H, W); None when there are no people."""
    if actors_dir is None or not (actors_dir / ACTORS_JSON).exists():
        return None
    _, labels = read_tracks(actors_dir)
    if labels is None or not labels.any():
        return None
    if len(labels) != count:
        raise ValueError(f"actor masks cover {len(labels)} frames, expected {count}")
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    out = []
    for lab in labels:
        m = (lab > 0).astype(np.uint8)
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out.append(cv2.dilate(m, kernel) > 0)
    return out
