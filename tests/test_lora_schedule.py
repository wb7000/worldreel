"""The adapter schedule decides whether the identity LoRA freezes the video.

D-9.27: the v4 adapter buys identity but freezes the pose (180/215 still
frames). Applying it only to the END of the denoising trajectory should keep
the base model's motion. These tests pin the schedule arithmetic.
"""

from __future__ import annotations

import pytest

from tools.lora_schedule import (
    EARLY,
    LATE,
    STATIC,
    adapter_weight,
    describe,
    progress_from_timestep,
    schedule_at,
)


def test_static_keeps_the_strength_for_every_step() -> None:
    for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert adapter_weight(STATIC, 1.0, 0.0, progress) == 1.0
    assert adapter_weight(STATIC, 0.5, 0.0, 0.5) == 0.5


def test_late_is_off_until_the_start_then_ramps_to_full() -> None:
    assert adapter_weight(LATE, 1.0, 0.75, 0.0) == 0.0
    assert adapter_weight(LATE, 1.0, 0.75, 0.75) == 0.0
    assert adapter_weight(LATE, 1.0, 0.75, 0.875) == pytest.approx(0.5)
    assert adapter_weight(LATE, 1.0, 0.75, 1.0) == pytest.approx(1.0)


def test_early_is_the_mirror_image_of_late() -> None:
    assert adapter_weight(EARLY, 1.0, 0.25, 0.0) == pytest.approx(1.0)
    assert adapter_weight(EARLY, 1.0, 0.25, 0.25) == pytest.approx(0.0)
    assert adapter_weight(EARLY, 1.0, 0.25, 0.125) == pytest.approx(0.5)


def test_the_weight_never_leaves_the_unit_range() -> None:
    for mode in (LATE, EARLY):
        for start in (0.0, 0.25, 0.5, 0.75, 1.0):
            for progress in (0.0, 0.33, 0.66, 1.0):
                weight = adapter_weight(mode, 1.0, start, progress)
                assert 0.0 <= weight <= 1.0


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        adapter_weight("sideways", 1.0, 0.0, 0.5)


def test_progress_follows_the_step_index_not_the_value() -> None:
    # A descending, unevenly spaced timestep list (the scheduler's own shape).
    timesteps = [1000.0, 900.0, 700.0, 400.0, 0.0]
    assert progress_from_timestep(timesteps, 1000.0) == pytest.approx(0.0)
    assert progress_from_timestep(timesteps, 400.0) == pytest.approx(0.75)
    assert progress_from_timestep(timesteps, 0.0) == pytest.approx(1.0)


def test_progress_of_an_empty_schedule_is_zero() -> None:
    assert progress_from_timestep([], 500.0) == 0.0


def test_schedule_at_uses_the_timestep_of_the_step() -> None:
    timesteps = [1000.0, 750.0, 500.0, 250.0, 0.0]
    # The late schedule (start 0.5) is off for the first three steps and ramps
    # over the last two.
    assert schedule_at(LATE, 1.0, 0.5, timesteps, 1000.0) == pytest.approx(0.0)
    assert schedule_at(LATE, 1.0, 0.5, timesteps, 500.0) == pytest.approx(0.0)
    assert schedule_at(LATE, 1.0, 0.5, timesteps, 250.0) == pytest.approx(0.5)
    assert schedule_at(LATE, 1.0, 0.5, timesteps, 0.0) == pytest.approx(1.0)


def test_describe_counts_the_affected_steps() -> None:
    assert "25/50" in describe(LATE, 0.5, 50)
    assert describe(STATIC, 0.5, 50) == "static"
