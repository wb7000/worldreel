"""First-person vs third-person audio on a real `.wreel` (brief Phase 5 acceptance).

Mirrors the player (player/src/audio.ts + cameras.ts): the listener is the view's eye — the
actor's head (eye height, 0.15 m forward) in first person, the follow camera (offset
[0, 1.15h, 1.0h] in the actor's heading frame, looking at the chest) in third person. For every
positional footstep, level (Web Audio inverse distance model, ref 1 m) and azimuth are computed
for both listeners at the step's time.

Usage: python -m eval.pov_audio world.wreel --actor p1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval.geometry import quat_to_matrix
from formats.wreel import read_wreel


def _sample(track: dict[str, Any], t: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(track["times"])
    i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    a = float(np.clip((t - times[i - 1]) / max(times[i] - times[i - 1], 1e-9), 0, 1))
    pos = (1 - a) * np.asarray(track["positions"][i - 1]) + a * np.asarray(track["positions"][i])
    rot = quat_to_matrix(track["rotations"][i if a > 0.5 else i - 1])
    return pos, rot


def _heading(rot: np.ndarray) -> np.ndarray:
    f = rot @ np.array([0.0, 0.0, -1.0])
    f[1] = 0.0
    return np.asarray(f / (np.linalg.norm(f) or 1.0), np.float64)


def heard(eye: np.ndarray, forward: np.ndarray, source: np.ndarray) -> tuple[float, float]:
    v = source - eye
    d = float(np.linalg.norm(v))
    gain_db = 20 * math.log10(1.0 / (1.0 + max(d, 1.0) - 1.0))
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right) or 1.0
    return gain_db, math.degrees(math.atan2(float(v @ right), float(v @ forward)))


def run(world: Path, actor_id: str) -> dict[str, Any]:
    w = read_wreel(world)
    actor = next(a for a in w.manifest["actors"] if a["id"] == actor_id)
    track = w.track(actor["track"]["path"])
    h = actor["eye_height"] / 0.925
    rows = []
    for e in w.manifest["audio"]["events"]:
        if e.get("type") != "footstep" or "position" not in e:
            continue
        pos, rot = _sample(track, float(e["t"]))
        fwd = _heading(rot)
        fp_eye = pos + np.array([0.0, actor["eye_height"], 0.0]) + 0.15 * fwd
        tp_eye = pos + fwd * (-1.0 * h) + np.array([0.0, 1.15 * h, 0.0])
        tp_fwd = (pos + np.array([0.0, 0.8 * actor["eye_height"], 0.0])) - tp_eye
        tp_fwd /= np.linalg.norm(tp_fwd)
        src = np.asarray(e["position"], np.float64)
        g1, a1 = heard(fp_eye, fwd, src)
        g3, a3 = heard(tp_eye, tp_fwd, src)
        rows.append(
            {"own": e.get("actor_id") == actor_id, "dgain": g1 - g3, "dazimuth": abs(a1 - a3)}
        )
    own = [r for r in rows if r["own"]]
    others = [r for r in rows if not r["own"]]

    def summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
        if not rs:
            return {"events": 0}
        return {
            "events": len(rs),
            "median_level_diff_db": round(float(np.median([r["dgain"] for r in rs])), 2),
            "median_direction_diff_deg": round(float(np.median([r["dazimuth"] for r in rs])), 1),
        }

    return {"actor": actor_id, "own_steps": summary(own), "other_people_steps": summary(others)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("world", type=Path)
    ap.add_argument("--actor", default="p1")
    a = ap.parse_args(argv)
    print(json.dumps(run(a.world, a.actor), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
