"""Turn trained splats (gsplat parameters, OpenCV axes) into a `.wreel` Gaussian layer
(spec §6.3, WorldReel axes), and swap such a layer into an existing reconstruction `.wreel`.

numpy only: runs on the laptop. Training happens on the GPU (stages/reconstruct/splat_train.py);
only the trained parameters come back, and the file is signed locally with the owner's key.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import numpy.typing as npt

from formats.wreel import (
    GAUSSIAN_FORMAT_VERSION,
    PROVENANCE_CLASSES,
    GaussianCloud,
    Wreel,
)

F64 = npt.NDArray[np.float64]
_FLIP = np.diag([1.0, -1.0, -1.0])
OBSERVED = PROVENANCE_CLASSES.index("observed")
FOREGROUND_CONFIDENCE = 0.9
# The background sphere's colour is seen in the video; only its distance is assumed. It stays
# `observed` but with lower confidence (DECISIONS D-3b.7).
BACKGROUND_CONFIDENCE = 0.6
MIN_OPACITY = 1.0 / 255.0


def quat_wxyz_to_matrix(q: F64) -> F64:
    """(N, 4) quaternions [w, x, y, z] (any norm) → (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def matrix_to_quat_xyzw(r: F64) -> F64:
    """(N, 3, 3) rotation matrices → (N, 4) unit quaternions [x, y, z, w] with w >= 0."""
    m = r
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.empty((len(m), 4))
    c0 = tr > 0
    c1 = ~c0 & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    c2 = ~c0 & ~c1 & (m[:, 1, 1] > m[:, 2, 2])
    c3 = ~c0 & ~c1 & ~c2
    s = np.sqrt(np.maximum(tr[c0] + 1.0, 1e-12)) * 2
    q[c0] = np.stack(
        [
            (m[c0, 2, 1] - m[c0, 1, 2]) / s,
            (m[c0, 0, 2] - m[c0, 2, 0]) / s,
            (m[c0, 1, 0] - m[c0, 0, 1]) / s,
            0.25 * s,
        ],
        -1,
    )
    s = np.sqrt(np.maximum(1.0 + m[c1, 0, 0] - m[c1, 1, 1] - m[c1, 2, 2], 1e-12)) * 2
    q[c1] = np.stack(
        [
            0.25 * s,
            (m[c1, 0, 1] + m[c1, 1, 0]) / s,
            (m[c1, 0, 2] + m[c1, 2, 0]) / s,
            (m[c1, 2, 1] - m[c1, 1, 2]) / s,
        ],
        -1,
    )
    s = np.sqrt(np.maximum(1.0 + m[c2, 1, 1] - m[c2, 0, 0] - m[c2, 2, 2], 1e-12)) * 2
    q[c2] = np.stack(
        [
            (m[c2, 0, 1] + m[c2, 1, 0]) / s,
            0.25 * s,
            (m[c2, 1, 2] + m[c2, 2, 1]) / s,
            (m[c2, 0, 2] - m[c2, 2, 0]) / s,
        ],
        -1,
    )
    s = np.sqrt(np.maximum(1.0 + m[c3, 2, 2] - m[c3, 0, 0] - m[c3, 1, 1], 1e-12)) * 2
    q[c3] = np.stack(
        [
            (m[c3, 0, 2] + m[c3, 2, 0]) / s,
            (m[c3, 1, 2] + m[c3, 2, 1]) / s,
            0.25 * s,
            (m[c3, 1, 0] - m[c3, 0, 1]) / s,
        ],
        -1,
    )
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1
    return q


def splats_to_gaussians(
    params: dict[str, npt.NDArray[Any]], *, centre_cv: F64, background_beyond: float
) -> GaussianCloud:
    """Trainer parameters (log-scales, logit opacities/colours, wxyz quats, OpenCV world) →
    WorldReel Gaussians. Gaussians farther than `background_beyond` from `centre_cv` are the
    background sphere (identified by distance, because densification reorders Gaussians)."""
    means = params["means"].astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-params["opacities"].astype(np.float64)))
    keep = opacity >= MIN_OPACITY
    means, opacity = means[keep], opacity[keep]
    scales = np.exp(params["scales"][keep].astype(np.float64))
    colors = 1.0 / (1.0 + np.exp(-params["colors"][keep].astype(np.float64)))
    rot_cv = quat_wxyz_to_matrix(params["quats"][keep].astype(np.float64))
    rot_wr = _FLIP @ rot_cv @ _FLIP  # covariance F R S² Rᵀ F = (F R F) S² (F R F)ᵀ
    background = np.linalg.norm(means - centre_cv, axis=1) > background_beyond
    confidence = np.where(background, BACKGROUND_CONFIDENCE, FOREGROUND_CONFIDENCE)
    return GaussianCloud(
        positions=(means @ _FLIP.T).astype(np.float32),
        scales=scales.astype(np.float32),
        rotations=matrix_to_quat_xyzw(rot_wr).astype(np.float32),
        colors=np.clip(np.round(colors * 255), 0, 255).astype(np.uint8),
        opacity=opacity.astype(np.float32),
        cls=np.full(len(means), OBSERVED, np.uint8),
        confidence=confidence.astype(np.float32),
    )


def subset(g: GaussianCloud, idx: npt.NDArray[np.int64]) -> GaussianCloud:
    return GaussianCloud(
        g.positions[idx],
        g.scales[idx],
        g.rotations[idx],
        g.colors[idx],
        g.opacity[idx],
        g.cls[idx],
        g.confidence[idx],
    )


def coarse_lod(g: GaussianCloud, fraction: float = 0.2) -> GaussianCloud:
    """LOD 0: the Gaussians that contribute most (opacity x largest cross-section)."""
    s = np.sort(g.scales, axis=1)
    weight = g.opacity * s[:, 2] * s[:, 1]
    n = max(1, int(len(g) * fraction))
    return subset(g, np.sort(np.argsort(-weight)[:n]))


def wreel_with_gaussians(
    template: Wreel, g: GaussianCloud, *, layer_id: str = "static", stage_record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Copy of a reconstruction `.wreel` whose static point layer is replaced by Gaussians."""
    manifest = copy.deepcopy(template.manifest)
    entries = {k: v for k, v in template.entries.items() if not k.startswith(f"scene/{layer_id}/")}
    coarse = coarse_lod(g)
    lod0, lod1 = f"scene/{layer_id}/lod0.gsp", f"scene/{layer_id}/lod1.gsp"
    entries[lod0], entries[lod1] = coarse.pack(), g.pack()
    layers = [layer for layer in manifest["scene"]["layers"] if layer["id"] != layer_id]
    layers.insert(
        0,
        {
            "id": layer_id,
            "kind": "gaussians",
            "lods": [
                {"level": 0, "path": lod0, "count": len(coarse)},
                {"level": 1, "path": lod1, "count": len(g)},
            ],
        },
    )
    manifest["scene"]["layers"] = layers
    manifest["format_version"] = GAUSSIAN_FORMAT_VERSION
    manifest["generation"]["stages"].append(stage_record)
    manifest["generation"]["models"].append(
        {
            "name": "gsplat (3D Gaussian Splatting, per-scene training)",
            "version": "1.5.3",
            "license": "Apache-2.0",
        }
    )
    fg = g.positions[g.confidence >= FOREGROUND_CONFIDENCE]
    if len(fg):
        lo, hi = np.percentile(fg, 1, axis=0) - 0.5, np.percentile(fg, 99, axis=0) + 0.5
        manifest["scene"]["bounds"] = {"min": [float(v) for v in lo], "max": [float(v) for v in hi]}
    manifest["files"] = {}
    return manifest, entries
