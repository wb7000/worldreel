"""Blocking: who stands where and how they move through a scene.

From the script's continuity state (who is at the scene's location), the `=` state changes
(moves become walks between spots) and the voiced timing (when each line is spoken). Pure Python;
no model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from formats.wscript import Change, Dialogue
from stages.generate.places import Place

F64 = npt.NDArray[np.float64]


@dataclass
class Blocking:
    spots: dict[str, list[float]]  # character id -> home position (place coords)
    tracks: dict[str, dict[str, list[Any]]]  # character id -> {times, positions, rotations}
    eye_target: dict[str, list[float]]  # character id -> position they face
    present: list[str]  # character ids at the scene's location, in spot order


def _yaw_quat(yaw: float) -> list[float]:
    return [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]


def _statement_times(
    statements: list[Any], timing: list[dict[str, Any]] | None, duration: float
) -> dict[int, float]:
    """Map every `.wsl` statement's position to a time in seconds.

    Dialogue lines carry their statement `index` into `timing.json` (start/end). Dialogue lines are
    matched to timing entries in statement order vs start-time order; every other statement is
    interpolated between neighbouring dialogue times (clamped at the ends). Without timing, all
    statements spread evenly over the duration.
    """
    n = len(statements)
    dlg = [(i, s) for i, s in enumerate(statements) if isinstance(s, Dialogue)]
    dlg_times: dict[int, float] = {}
    if timing and dlg:
        timing_sorted = sorted(timing, key=lambda e: float(e["start"]))
        for k, (pos, _s) in enumerate(dlg):
            if k < len(timing_sorted):
                dlg_times[pos] = float(timing_sorted[k]["start"])
    elif dlg:
        for k, (pos, _s) in enumerate(dlg):
            dlg_times[pos] = duration * (k + 1) / (len(dlg) + 1)
    times: dict[int, float] = {}
    positions = sorted(dlg_times)
    for p in range(n):
        if p in dlg_times:
            times[p] = dlg_times[p]
            continue
        before = [q for q in positions if q < p]
        after = [q for q in positions if q > p]
        if before and after:
            b, a = before[-1], after[0]
            times[p] = dlg_times[b] + (dlg_times[a] - dlg_times[b]) * (p - b) / (a - b)
        elif before:
            times[p] = dlg_times[before[-1]]
        elif after:
            times[p] = dlg_times[after[0]]
        elif n:
            times[p] = duration * (p + 1) / (n + 1)
    return times


def plan_blocking(
    statements: list[Any],
    continuity: dict[str, Any],
    place: Place,
    timing: list[dict[str, Any]] | None,
    duration: float,
    arriving: dict[str, float] | None = None,
) -> Blocking:
    """Plan one scene's blocking.

    `continuity` is the scene's `continuity.characters` (cid -> {location, wearing, holding,
    state}) plus a `__location__` key naming the scene's location. `timing` is the voiced per-line
    start/end (from `timing.json`). `duration` is the world length. `arriving` maps character ids
    to the time they move into the scene location during the window. Only `=` changes whose time
    falls inside [0, duration] become walks.
    """
    scene_location = continuity.get("__location__", "")
    chars = continuity.get("characters", {})
    start_here = sorted(cid for cid, st in chars.items() if st.get("location") == scene_location)
    arriving = dict(arriving or {})
    present = sorted(set(start_here) | set(arriving))

    spot_names = list(place.spots)
    spots: dict[str, list[float]] = {}
    for k, cid in enumerate(present):
        spots[cid] = list(place.spots[spot_names[k % len(spot_names)]])

    centre = [
        (place.bounds["min"][0] + place.bounds["max"][0]) / 2,
        1.4,
        (place.bounds["min"][2] + place.bounds["max"][2]) / 2,
    ]
    eye_target: dict[str, list[float]] = {}
    for cid in present:
        others = [spots[o] for o in present if o != cid]
        target = np.mean(others, axis=0) if others else np.array(centre)
        eye_target[cid] = [round(float(v), 4) for v in target]

    times = _statement_times(statements, timing, duration)
    scene_end = float(duration)
    tracks: dict[str, dict[str, list[Any]]] = {}

    # Arrivals: walk in from off-screen to their spot, arriving at their arrival time.
    offscreen = np.array([8.0, 0.0, 8.0])
    for cid, t0 in arriving.items():
        t0 = min(max(t0, 0.0), scene_end)
        spot = np.array(spots[cid], dtype=np.float64)
        start = spot + offscreen
        n_steps = max(4, int(t0 * 8)) if t0 > 0 else 1
        t_l, p_l, r_l = [], [], []
        for i in range(n_steps + 1):
            u = i / n_steps
            tt = min(scene_end, t0 * u)
            pos = start + (spot - start) * u
            yaw = math.atan2(-(eye_target[cid][0] - pos[0]), -(eye_target[cid][2] - pos[2]))
            t_l.append(round(float(tt), 4))
            p_l.append([round(float(v), 4) for v in pos])
            r_l.append(_yaw_quat(yaw))
        tracks[cid] = {"times": t_l, "positions": p_l, "rotations": r_l}

    # Characters present at the start who move away within the window: walk to a far spot.
    for s in statements:
        if not isinstance(s, Change) or s.target == "world":
            continue
        cid = s.target
        if cid not in spots or cid in arriving:
            continue
        if not any(k == "location" and op == "=" for k, op, _v in s.ops):
            continue
        t0 = times.get(statements.index(s), scene_end)
        if t0 >= scene_end:
            continue
        home = np.array(spots[cid], dtype=np.float64)
        dest_pts = max(
            place.spots.values(), key=lambda sp: float(np.linalg.norm(np.array(sp) - home))
        )
        dest = np.array(dest_pts, dtype=np.float64)
        n_steps = max(8, int((scene_end - t0) * 8))
        t_l, p_l, r_l = [], [], []
        for i in range(n_steps + 1):
            u = i / n_steps
            tt = min(scene_end, t0 + (scene_end - t0) * u)
            pos = home + (dest - home) * u
            yaw = math.atan2(-(eye_target[cid][0] - pos[0]), -(eye_target[cid][2] - pos[2]))
            t_l.append(round(float(tt), 4))
            p_l.append([round(float(v), 4) for v in pos])
            r_l.append(_yaw_quat(yaw))
        tracks[cid] = {"times": t_l, "positions": p_l, "rotations": r_l}

    # Hold tracks for everyone without one yet.
    for cid in present:
        if cid in tracks:
            continue
        pos = spots[cid]
        yaw = math.atan2(-(eye_target[cid][0] - pos[0]), -(eye_target[cid][2] - pos[2]))
        tracks[cid] = {
            "times": [0.0, scene_end],
            "positions": [pos, pos],
            "rotations": [_yaw_quat(yaw), _yaw_quat(yaw)],
        }

    return Blocking(spots=spots, tracks=tracks, eye_target=eye_target, present=present)
