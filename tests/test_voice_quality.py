from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyworld")

from stages.generate.voice_quality import PRESETS, Shape, shape_voice
from stages.generate.voices import pitch_spread_semitones, pitch_track

FS = 16000


def _voice(seconds: float = 1.5, f0: float = 140.0) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    glide = f0 * 2 ** (np.sin(2 * np.pi * 1.2 * t) * 2 / 12)
    phase = 2 * np.pi * np.cumsum(glide) / FS
    out: np.ndarray = np.sum([np.sin(k * phase) / k for k in range(1, 12)], axis=0) * 0.2
    return out  # harmonic "vowel"


def test_neutral_shape_keeps_pitch_and_length() -> None:
    x = _voice()
    y = shape_voice(x, FS, Shape())
    assert abs(len(y) - len(x)) < FS * 0.02
    assert abs(np.median(pitch_track(y, FS)) - np.median(pitch_track(x, FS))) < 6


def test_joy_raises_and_widens_pitch_and_sad_lowers_and_slows() -> None:
    x = _voice()
    base_f0, base_spread = np.median(pitch_track(x, FS)), pitch_spread_semitones(x, FS)
    joy = shape_voice(x, FS, PRESETS["joyful"])
    sad = shape_voice(x, FS, PRESETS["despair"])
    assert np.median(pitch_track(joy, FS)) > base_f0 * 2 ** (1.5 / 12)
    assert pitch_spread_semitones(joy, FS) > base_spread * 1.3
    assert np.median(pitch_track(sad, FS)) < base_f0 and len(sad) > len(x) * 1.08


def test_intensity_interpolates_and_whisper_removes_voicing() -> None:
    half = PRESETS["joyful"].scaled(0.5)
    assert half.shift_st == pytest.approx(1.0) and half.range_x == pytest.approx(1.25)
    x = _voice()
    whisper = shape_voice(x, FS, PRESETS["whisper_tense"].scaled(1.0))
    assert len(pitch_track(whisper, FS)) < 0.5 * len(pitch_track(x, FS))
