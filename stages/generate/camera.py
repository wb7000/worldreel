"""Director's-cut camera path from POV hints and speaker changes.

For each statement the planner picks a camera placement: a `>` POV hint wins; otherwise a
dialogue line puts the camera on its speaker (an over-the-shoulder shot behind the previous
speaker), and everything else holds the previous placement. Keyframes sit at statement times,
the first at t=0 and the last at the scene end.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from formats.wscript import Dialogue, Pov
from stages.generate.blocking import Blocking, _statement_times
from stages.generate.places import Place

DEFAULT_FOV = 50.0


def _yaw_quat(yaw: float) -> list[float]:
    return [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]


def plan_camera(
    statements: list[Any],
    blocking: Blocking,
    place: Place,
    timing: list[dict[str, Any]] | None,
    duration: float,
) -> list[dict[str, Any]]:
    """Return camera keyframes sorted by time."""
    times = _statement_times(statements, timing, duration)
    centre: list[float] = [
        (place.bounds["min"][0] + place.bounds["max"][0]) / 2,
        1.6,
        (place.bounds["min"][2] + place.bounds["max"][2]) / 2,
    ]

    keyframes: list[dict[str, Any]] = []
    last_camera: tuple[list[float], list[float], float] | None = None

    def emit(
        p: int,
        kind: str,
        target_cid: str | None = None,
        pov: Pov | None = None,
        fov: float = DEFAULT_FOV,
    ) -> None:
        nonlocal last_camera
        t = float(times.get(p, 0.0))
        pos, rot = _placement(kind, blocking, place, centre, target_cid, pov, last_camera)
        if last_camera is not None:
            lp, lr, _lf = last_camera
            # identical consecutive placements collapse to one keyframe
            if _same(pos, lp) and _same(rot, lr):
                return
        keyframes.append(
            {
                "t": round(t, 4),
                "position": [round(v, 4) for v in pos],
                "rotation": [round(v, 6) for v in rot],
                "fov_deg": round(fov, 3),
            }
        )
        last_camera = (pos, rot, fov)

    for p, s in enumerate(statements):
        if isinstance(s, Pov):
            emit(p, "pov", pov=s, target_cid=s.target)
        elif isinstance(s, Dialogue):
            emit(p, "speaker", target_cid=s.speaker)

    # bookends: establishing wide at t=0, hold the last placement to the end.
    if not keyframes:
        emit(-1, "wide")
    elif keyframes[0]["t"] > 0:
        pos, rot = _placement("wide", blocking, place, centre, None, None, None)
        keyframes.insert(0, {"t": 0.0, "position": pos, "rotation": rot, "fov_deg": DEFAULT_FOV})
    if keyframes[-1]["t"] < duration:
        last = keyframes[-1]
        keyframes.append(
            {
                "t": round(float(duration), 4),
                "position": last["position"],
                "rotation": last["rotation"],
                "fov_deg": last["fov_deg"],
            }
        )
    # dedupe identical consecutive times
    out: list[dict[str, Any]] = []
    for kf in keyframes:
        if out and abs(out[-1]["t"] - kf["t"]) < 1e-6:
            continue
        out.append(kf)
    return out


def _same(a: Sequence[float], b: Sequence[float], tol: float = 1e-6) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=False))


def _placement(
    kind: str,
    blocking: Blocking,
    place: Place,
    centre: list[float],
    target_cid: str | None,
    pov: Pov | None,
    last_camera: tuple[list[float], list[float], float] | None,
) -> tuple[list[float], list[float]]:
    """Camera (position, rotation quaternion) for one placement."""
    if pov is not None:
        return _pov_placement(pov, blocking, place, centre)

    spots = blocking.spots
    if kind == "wide":
        return _wide(place, centre)
    if kind == "speaker" and target_cid in spots:
        eye = blocking.eye_target.get(target_cid, centre)
        pos = np_eye_behind(spots[target_cid], eye, 2.6)
        return pos, _look_vec(pos, eye)
    return _wide(place, centre)


def _wide(place: Place, centre: list[float]) -> tuple[list[float], list[float]]:
    lo, hi = place.bounds["min"], place.bounds["max"]
    span = max(hi[0] - lo[0], hi[2] - lo[2]) or 4.0
    dist = max(5.0, 1.6 * span)
    pos: list[float] = [
        centre[0] - dist * 0.6,
        centre[1] + dist * 0.45,
        centre[2] - dist * 0.6,
    ]
    return pos, _look_vec(pos, centre)


def _pov_placement(
    pov: Pov, blocking: Blocking, place: Place, centre: list[float]
) -> tuple[list[float], list[float]]:
    mode = pov.mode
    target = pov.target
    spots = blocking.spots
    if mode == "first" and target in spots:
        eye = blocking.eye_target.get(target, centre)
        return list(spots[target]), _look_vec(list(spots[target]), eye)
    if mode == "third" and target in spots:
        eye = blocking.eye_target.get(target, centre)
        pos = np_eye_behind(spots[target], eye, 3.2)
        pos[1] += 0.9
        return pos, _look_vec(pos, eye)
    if mode == "over" and target in spots:
        eye = blocking.eye_target.get(target, centre)
        return np_eye_behind(spots[target], eye, 2.6), _look_vec(list(spots[target]), eye)
    if mode == "wide":
        return _wide(place, centre)
    if mode == "free":
        return _wide(place, centre)
    return _wide(place, centre)


def np_eye_behind(origin: list[float], look_at: list[float], back: float) -> list[float]:
    """`back` metres behind `origin`, looking at `look_at` (camera over the shoulder)."""
    import numpy as np

    o = np.array(origin, dtype=np.float64)
    look = np.array(look_at, dtype=np.float64)
    fwd = look - o
    fwd[1] = 0.0
    n = float(np.linalg.norm(fwd)) or 1.0
    fwd = fwd / n
    pos = o - fwd * back
    pos[1] = o[1] + 0.5
    return [float(v) for v in pos]


def _look_vec(eye: list[float], target: list[float]) -> list[float]:
    fx, fy, fz = (t - e for t, e in zip(target, eye, strict=False))
    n = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / n, fy / n, fz / n
    yaw = math.atan2(-fx, -fz)
    pitch = math.asin(max(-1.0, min(1.0, fy)))
    q = _yaw_quat(yaw)
    if abs(pitch) > 1e-6:
        q = _mul(q, [math.sin(pitch / 2), 0.0, 0.0, math.cos(pitch / 2)])
    return q


def _mul(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
