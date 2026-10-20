from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyworld")

from stages.generate.psola import f0_track, psola, tilt
from stages.generate.voice_quality import PRESETS, Shape, hnr_db, shape_voice_psola

FS = 16000


def _vowel(f0: float = 140.0, seconds: float = 1.2) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    out: np.ndarray = (
        np.sum([np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12)], axis=0) * 0.2
    )
    return out


def test_ratio_one_returns_the_waveform_exactly() -> None:
    x = _vowel()
    assert np.array_equal(psola(x, FS, np.ones(400)), x)
    assert np.array_equal(shape_voice_psola(x, FS, Shape()), np.clip(x, -1, 1))


def test_two_semitones_up_keeps_length_and_cleanness() -> None:
    x = _vowel()
    f0 = f0_track(x, FS)
    y = psola(x, FS, np.full(len(f0), 2 ** (2 / 12)), f0)
    assert len(y) == len(x)
    f_in, f_out = np.median(f0[f0 > 0]), np.median(f0_track(y, FS)[f0_track(y, FS) > 0])
    assert abs(12 * np.log2(f_out / f_in) - 2.0) < 0.4
    mid = slice(FS // 5, -FS // 5)
    assert hnr_db(y[mid], FS) > hnr_db(x[mid], FS) - 3


def test_joy_preset_raises_pitch_without_resynthesis_noise() -> None:
    x = _vowel()
    y = shape_voice_psola(x, FS, PRESETS["joyful"].amplified(0.8))
    f_in = np.median(f0_track(x, FS)[f0_track(x, FS) > 0])
    f_out = np.median(f0_track(y, FS)[f0_track(y, FS) > 0])
    assert f_out > f_in * 2 ** (1.0 / 12) and len(y) == len(x)


def test_tilt_is_a_filter_that_brightens() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, FS)
    bright = tilt(noise, FS, 4.0)
    hi = slice(FS // 2 * 3 // 4, None)
    assert np.abs(np.fft.rfft(bright))[hi].mean() > 2 * np.abs(np.fft.rfft(noise))[hi].mean()
