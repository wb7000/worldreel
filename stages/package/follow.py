"""Seen-space follow planner (algorithm A-2, docs/ALGORITHMS.md).

Plans a "follow this person" camera path offline that keeps the view on parts of the scene the
video actually observed (per-Gaussian confidence from A-1) and keeps the person in sight. The
result is an ordinary camera path (spec §8) played by a `directors-cut` preset, so the format
does not change. numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from stages.package.gaussians import matrix_to_quat_xyzw

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FollowConfig:
    yaw_step_deg: float = 20.0
    distances: tuple[float, ...] = (0.9, 1.3)  # x height
    eye_height: float = 1.15  # x height above the floor
    target_height: float = 0.55  # x height (chest)
    fov_deg: float = 60.0  # vertical
    buffer: tuple[int, int] = (48, 27)  # width, height of the scoring buffer
    clear_ratio: float = 0.9  # line of sight: nothing nearer than this x target distance
    yaw_cost: float = 0.5  # per 180 deg of change between samples
    distance_cost: float = 0.2  # per height of change between samples


def look_basis(eye: F64, target: F64, up: F64) -> tuple[F64, F64, F64]:
    """(right, true up, forward) for a camera at `eye` looking at `target`."""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
    r = r / np.linalg.norm(r)
    return r, np.cross(r, f), f


def look_quaternion(eye: F64, target: F64, up: F64) -> list[float]:
    """WorldReel camera rotation (looks along -Z, +Y up) as [x, y, z, w]."""
    r, u, f = look_basis(eye, target, up)
    m = np.stack([r, u, -f], axis=1)[None]
    return [float(x) for x in matrix_to_quat_xyzw(m)[0]]


def seen_score(
    eye: F64,
    target: F64,
    up: F64,
    points: F64,
    confidence: npt.NDArray[np.floating],
    cfg: FollowConfig,
) -> tuple[float, bool]:
    """(mean confidence of the front-most Gaussian per pixel — empty pixels count 0,
    whether the target is in clear line of sight)."""
    w, h = cfg.buffer
    r, u, f = look_basis(eye, target, up)
    d = points - eye
    z = d @ f
    front = z > 1e-3
    fy = (h / 2) / math.tan(math.radians(cfg.fov_deg) / 2)
    px = (w / 2 + fy * (d[front] @ r) / z[front]).astype(np.int64)
    py = (h / 2 - fy * (d[front] @ u) / z[front]).astype(np.int64)
    zf = z[front]
    cf = np.asarray(confidence, np.float64)[front]
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    pix = py[ok] * w + px[ok]
    zf, cf = zf[ok], cf[ok]
    depth = np.full(w * h, np.inf)
    np.minimum.at(depth, pix, zf)
    conf = np.zeros(w * h)
    nearest = zf <= depth[pix]
    np.maximum.at(conf, pix[nearest], cf[nearest])
    dist = float(np.linalg.norm(target - eye))
    cx, cy = w // 2, h // 2
    window = depth.reshape(h, w)[cy - 1 : cy + 2, cx - 1 : cx + 2]
    clear = bool(window.min() >= cfg.clear_ratio * dist)
    return float(conf.mean()), clear


def plan_follow(
    positions: F64,
    ups: F64,
    height: float,
    points: F64,
    confidence: npt.NDArray[np.floating],
    cfg: FollowConfig | None = None,
) -> tuple[F64, F64, F64]:
    """Per track sample: (eyes N x 3, targets N x 3, scores N). Viterbi over a ring of
    candidates (yaw x distance) trading seen share against smoothness."""
    cfg = cfg or FollowConfig()
    yaws = np.radians(np.arange(0.0, 360.0, cfg.yaw_step_deg))
    cands = [(y, d) for y in yaws for d in cfg.distances]
    t_count, c_count = len(positions), len(cands)
    score = np.zeros((t_count, c_count))
    eyes = np.zeros((t_count, c_count, 3))
    targets = np.zeros((t_count, 3))
    for t in range(t_count):
        up = ups[t] / np.linalg.norm(ups[t])
        # Horizontal frame on the floor plane.
        a = np.cross(up, [0.0, 0.0, 1.0])
        a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-9 else np.array([1.0, 0.0, 0.0])
        b = np.cross(a, up)
        targets[t] = positions[t] + cfg.target_height * height * up
        for c, (yaw, dist) in enumerate(cands):
            eye = (
                positions[t]
                + dist * height * (math.cos(yaw) * a + math.sin(yaw) * b)
                + cfg.eye_height * height * up
            )
            eyes[t, c] = eye
            s, clear = seen_score(eye, targets[t], up, points, confidence, cfg)
            score[t, c] = s if clear else 0.0
    # Transition cost between candidates (yaw change and distance change).
    cy = np.array([y for y, _ in cands])
    cd = np.array([d for _, d in cands])
    dyaw = np.abs((cy[:, None] - cy[None, :] + math.pi) % (2 * math.pi) - math.pi) / math.pi
    trans = cfg.yaw_cost * dyaw + cfg.distance_cost * np.abs(cd[:, None] - cd[None, :])
    best = score[0].copy()
    back = np.zeros((t_count, c_count), np.int64)
    for t in range(1, t_count):
        total = best[:, None] - trans
        back[t] = total.argmax(0)
        best = total.max(0) + score[t]
    path = np.zeros(t_count, np.int64)
    path[-1] = int(best.argmax())
    for t in range(t_count - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    idx = np.arange(t_count)
    return eyes[idx, path], targets, score[idx, path]


def view_gap_deg(eyes: F64, targets: F64, train_centres: F64) -> F64:
    """Per sample: angle between the follow ray to the target and the closest training-camera
    ray to the same target (a proxy for how far the view is from what the video saw)."""
    out = np.zeros(len(eyes))
    for i, (e, tg) in enumerate(zip(eyes, targets, strict=True)):
        f = (tg - e) / np.linalg.norm(tg - e)
        rays = tg - train_centres
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        out[i] = float(np.degrees(np.arccos(np.clip(rays @ f, -1.0, 1.0)).min()))
    return out


def follow_cameras(
    manifest: dict[str, Any],
    tracks: dict[str, dict[str, Any]],
    points: F64,
    confidence: npt.NDArray[np.floating],
    cfg: FollowConfig | None = None,
) -> dict[str, Any]:
    """Replace each third-person "Follow person N" preset in `manifest` with a planned camera
    path (A-2). `tracks` maps actor id -> its track (times, positions, rotations)."""
    from eval.geometry import quat_to_matrix

    cfg = cfg or FollowConfig()
    report: dict[str, Any] = {}
    cameras = manifest.setdefault("cameras", [])
    presets = manifest.get("pov_presets", [])
    by_id = {a["id"]: a for a in manifest.get("actors", [])}
    for i, preset in enumerate(presets):
        aid = preset.get("actor_id")
        if preset.get("mode") != "third-person" or aid not in tracks or aid not in by_id:
            continue
        tr = tracks[aid]
        pos = np.asarray(tr["positions"], np.float64)
        ups = np.stack([quat_to_matrix(q) @ np.array([0.0, 1.0, 0.0]) for q in tr["rotations"]])
        height = by_id[aid]["eye_height"] / 0.925
        eyes, targets, scores = plan_follow(pos, ups, height, points, confidence, cfg)
        cam_id = f"follow_cam_{aid}"
        cameras.append(
            {
                "id": cam_id,
                "name": f"{preset['name']} (planned)",
                "kind": "alternative",
                "keyframes": [
                    {
                        "t": float(t),
                        "position": [float(x) for x in e],
                        "rotation": look_quaternion(e, tg, u),
                        "fov_deg": cfg.fov_deg,
                    }
                    for t, e, tg, u in zip(tr["times"], eyes, targets, ups, strict=True)
                ],
            }
        )
        presets[i] = {
            "id": preset["id"],
            "name": preset["name"],
            "mode": "directors-cut",
            "camera_id": cam_id,
        }
        report[aid] = round(float(scores.mean()), 3)
    return report
