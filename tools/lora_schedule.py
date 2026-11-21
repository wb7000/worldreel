"""When the character LoRA acts during the denoising trajectory.

The v4 adapter is the only artifact that ever produced identity (0.9256 mean
SFace cosine), but at full strength it also froze the pose: the delivery scored
a mean frame difference of 0.484 with 180/215 frames changing by less than 0.5
grey levels -- "an image that shakes" (D-9.25/D-9.27).

A LoRA delta is not the only knob: the *transformer* is called once per CFG
branch with the current timestep, so the adapter weight can depend on where the
trajectory is. The early, high-noise steps decide the layout, the pose and the
trajectory (the motion); the late steps refine what is already there (the face
texture and identity). Applying the identity adapter only late keeps the base
model's motion and still buys identity.

This module holds the schedule so it can be unit-tested; the kernel job
(`training/kaggle/jobs/character_render_lora_probe.py`) imports it.
"""

from __future__ import annotations

import math

# Schedule modes.
STATIC = "static"
LATE = "late"
EARLY = "early"


def progress_from_timestep(timesteps: list[float], t_value: float) -> float:
    """Where the current timestep sits in the trajectory, 0 (first) to 1 (last).

    The scheduler's timestep list is recovered by matching the value, so the
    progress axis is the STEP INDEX and not the noise level: "late" then means
    "the last N% of the steps" whatever the sigma schedule (UniPC with
    flow_shift=3 does not space the sigmas linearly).
    """
    if not timesteps:
        return 0.0
    index = min(range(len(timesteps)), key=lambda i: abs(timesteps[i] - t_value))
    return index / max(1, len(timesteps) - 1)


def adapter_weight(mode: str, strength: float, start: float, progress: float) -> float:
    """The LoRA weight for one step.

    * static -- `strength` for every step.
    * late   -- 0 until `progress` passes `start`, then a linear ramp up to
      `strength` at the end of the trajectory.
    * early  -- the mirror image: `strength` at the start, 0 after `start`.
    """
    if mode == STATIC:
        return strength
    start = min(1.0, max(0.0, start))
    if mode == LATE:
        span = max(1e-6, 1.0 - start)
        fraction = (progress - start) / span
    elif mode == EARLY:
        span = max(1e-6, start)
        fraction = (start - progress) / span
    else:
        raise ValueError(f"unknown schedule mode: {mode!r}")
    return strength * min(1.0, max(0.0, fraction))


def schedule_at(
    mode: str, strength: float, start: float, timesteps: list[float], t_value: float
) -> float:
    """The LoRA weight for the step whose timestep is `t_value`."""
    if mode == STATIC:
        return strength
    progress = progress_from_timestep(timesteps, t_value)
    return adapter_weight(mode, strength, start, progress)


def describe(mode: str, start: float, steps: int) -> str:
    """A one-line description of the schedule, for the job log."""
    if mode == STATIC:
        return "static"
    active = math.ceil((1.0 - start) * steps) if mode == LATE else math.ceil(start * steps)
    return f"{mode} start={start:.2f} ({active}/{steps} steps affected)"
