"""Footstep events from actor tracks, with a bank of generated clips (algorithm A-6).

Steps: an actor's track is walked in time; a step is emitted each time the horizontal distance
since the last step reaches one stride. Cadence follows speed (about 1.8 steps/s at 1.4 m/s,
up to 3 steps/s when running) — an approximation of typical human gait, not a measurement.
Scene units are converted to metres with the actor's height (monocular scale is unknown;
1.75 m is assumed for the stored height, as in packaging).

Each event carries the spec's attribute vocabulary (`stages.audio.synth`) and `clip`, the path
of the generated sound inside the `.wreel`. Identical attribute sets share clips (3 variants
each, used in turn), so the bank stays small.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from stages.audio.synth import footstep, wav_bytes

ASSUMED_HEIGHT_M = 1.75
VARIANTS = 3


def cadence_hz(speed_mps: float) -> float:
    """Steps per second for a walking / running speed (approximate human gait)."""
    return float(np.clip(1.8 + 0.6 * (speed_mps - 1.4), 1.5, 3.0))


def step_times(
    times: list[float], positions: list[list[float]], metres_per_unit: float
) -> list[tuple[float, list[float], float]]:
    """(time, position, speed m/s) of each step along a track."""
    t = np.asarray(times, np.float64)
    p = np.asarray(positions, np.float64)
    out: list[tuple[float, list[float], float]] = []
    if len(t) < 2:
        return out
    travelled = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            continue
        seg = np.linalg.norm((p[i] - p[i - 1])[[0, 2]]) * metres_per_unit
        speed = seg / dt
        if speed < 0.2:  # standing still: no steps
            continue
        stride = speed / cadence_hz(speed)
        travelled += seg
        while travelled >= stride:
            travelled -= stride
            # Clamp: when speed drops, the carried distance can exceed the new stride and the
            # step would land before this segment (negative times seen in packaging, D-5.2).
            frac = float(np.clip(1.0 - travelled / seg, 0.0, 1.0)) if seg > 0 else 1.0
            out.append(
                (
                    float(t[i - 1] + frac * dt),
                    list(p[i - 1] + frac * (p[i] - p[i - 1])),
                    float(speed),
                )
            )
    return out


def _bucket(attrs: dict[str, Any]) -> dict[str, Any]:
    """Quantise continuous attributes so similar steps share clips."""
    b = dict(attrs)
    b["speed"] = max(0.5, round(float(attrs.get("speed", 1.4)) * 2) / 2)  # 0.5 m/s steps
    b["weight"] = round(float(attrs.get("weight", 70.0)) / 10) * 10
    return b


def footstep_audio(
    actors: list[dict[str, Any]],
    tracks: dict[str, dict[str, Any]],
    defaults: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    """(events, clip entries, report). `defaults`: scene-wide attributes, e.g. {"surface":
    "gravel", "weather": "rain", "footwear": "boot"}; per-actor overrides in
    actor["audio"] are not part of the format, so only scene defaults apply."""
    base = {"surface": "concrete", "weather": "dry", "footwear": "sneaker", "weight": 70.0}
    base.update(defaults or {})
    events: list[dict[str, Any]] = []
    entries: dict[str, bytes] = {}
    used: dict[str, int] = {}
    for actor in actors:
        tr = tracks.get(actor["id"])
        if tr is None:
            continue
        height_units = float(actor.get("eye_height", 1.62)) / 0.925
        mpu = ASSUMED_HEIGHT_M / height_units if height_units > 0 else 1.0
        for t, pos, speed in step_times(tr["times"], tr["positions"], mpu):
            attrs = _bucket({**base, "speed": speed})
            key = hashlib.sha256(json.dumps(attrs, sort_keys=True).encode()).hexdigest()[:12]
            variant = used.get(key, 0) % VARIANTS
            used[key] = used.get(key, 0) + 1
            path = f"audio/fx/step_{key}_{variant}.wav"
            if path not in entries:
                seed = int(key[:8], 16) + variant
                entries[path] = wav_bytes(footstep(attrs, seed=seed))
            events.append(
                {
                    "t": round(t, 3),
                    "type": "footstep",
                    "actor_id": actor["id"],
                    "position": [float(v) for v in pos],
                    "attributes": {
                        **{k: v for k, v in attrs.items()},
                        "wet": attrs.get("weather") == "rain",
                        "clip": path,
                    },
                }
            )
    events.sort(key=lambda e: e["t"])
    report = {"footsteps": len(events), "clips": len(entries), "defaults": base}
    return events, entries, report
