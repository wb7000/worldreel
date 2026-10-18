from __future__ import annotations

import numpy as np

from stages.audio.events import cadence_hz, footstep_audio, step_times


def _walk(
    speed: float, seconds: float = 10.0, fps: float = 10.0
) -> tuple[list[float], list[list[float]]]:
    t = np.arange(0, seconds + 1e-9, 1 / fps)
    return t.tolist(), [[speed * ti, 0.0, 0.0] for ti in t]


def test_walking_gives_about_cadence_times_duration_steps() -> None:
    times, pos = _walk(1.4)
    steps = step_times(times, pos, metres_per_unit=1.0)
    assert abs(len(steps) - cadence_hz(1.4) * 10.0) <= 1
    assert all(abs(s[2] - 1.4) < 1e-6 for s in steps)
    gaps = np.diff([s[0] for s in steps])
    assert np.allclose(gaps, 1 / cadence_hz(1.4), atol=0.02)


def test_standing_still_gives_no_steps_and_running_steps_faster() -> None:
    times, pos = _walk(0.0)
    assert step_times(times, pos, 1.0) == []
    run = step_times(*_walk(4.0), metres_per_unit=1.0)
    walk = step_times(*_walk(1.4), metres_per_unit=1.0)
    assert len(run) > len(walk)


def test_units_are_scaled_by_actor_height() -> None:
    # Track in units where the person is 0.875 units tall (half of 1.75 m): 0.7 units/s = 1.4 m/s.
    times, pos = _walk(0.7)
    actor = {"id": "p1", "eye_height": 0.875 * 0.925}
    events, _, _ = footstep_audio([actor], {"p1": {"times": times, "positions": pos}})
    assert abs(events[0]["attributes"]["speed"] - 1.5) < 1e-9  # 1.4 m/s bucketed to 0.5 m/s
    assert abs(len(events) - cadence_hz(1.4) * 10.0) <= 1


def test_clip_bank_is_shared_and_events_carry_attributes() -> None:
    times, pos = _walk(1.4)
    actors = [{"id": "p1", "eye_height": 1.62}, {"id": "p2", "eye_height": 1.62}]
    tracks = {"p1": {"times": times, "positions": pos}, "p2": {"times": times, "positions": pos}}
    events, entries, report = footstep_audio(
        actors, tracks, {"surface": "gravel", "weather": "rain"}
    )
    assert report["footsteps"] == len(events) and len(entries) <= 3  # one attribute set, 3 variants
    a = events[0]["attributes"]
    assert a["surface"] == "gravel" and a["weather"] == "rain" and a["wet"] is True
    assert a["clip"] in entries and entries[a["clip"]][:4] == b"RIFF"
    assert all(
        isinstance(v, (str, int, float, bool)) for e in events for v in e["attributes"].values()
    )


def test_slowing_down_never_puts_a_step_outside_its_segment() -> None:
    # Run at 4 m/s, then walk at 1 m/s: the carried distance exceeds the shorter stride.
    times = [i * 0.1 for i in range(41)]
    x, pos = 0.0, []
    for i in range(41):
        pos.append([x, 0.0, 0.0])
        x += (4.0 if i < 20 else 1.0) * 0.1
    steps = step_times(times, pos, 1.0)
    ts = [s[0] for s in steps]
    assert ts == sorted(ts) and min(ts) >= 0.0 and max(ts) <= times[-1]
