from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from stages.audio.synth import SAMPLE_RATE, footstep, profile, write_wav


def test_footstep_is_deterministic_and_bounded() -> None:
    a = footstep({"surface": "gravel"}, seed=3)
    b = footstep({"surface": "gravel"}, seed=3)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32 and np.abs(a).max() <= 0.9 + 1e-6
    assert not np.array_equal(a, footstep({"surface": "gravel"}, seed=4))


def test_rain_makes_the_tail_longer_and_changes_the_spectrum() -> None:
    dry = profile(footstep({"surface": "concrete"}, seed=0))
    wet = profile(footstep({"surface": "concrete", "weather": "rain"}, seed=0))
    assert wet[-1] > dry[-1]  # decay to -20 dB
    assert np.abs(wet[:9] - dry[:9]).max() > 3.0  # some octave band moves by > 3 dB


def test_surfaces_differ_in_spectral_profile() -> None:
    wood = profile(footstep({"surface": "wood"}, seed=0))[:9]
    gravel = profile(footstep({"surface": "gravel"}, seed=0))[:9]
    # wood is low and resonant, gravel is bright and crunchy: 8 kHz band far apart
    assert gravel[7] - wood[7] > 20.0


def test_unknown_attributes_fall_back_to_defaults() -> None:
    assert np.array_equal(footstep({"surface": "lava", "footwear": "skates"}), footstep({}))


def test_write_wav_mono_and_stereo(tmp_path: Path) -> None:
    x = np.sin(np.linspace(0, 100, SAMPLE_RATE // 10)).astype(np.float32) * 0.5
    write_wav(tmp_path / "m.wav", x)
    write_wav(tmp_path / "s.wav", np.stack([x, -x], axis=1))
    with wave.open(str(tmp_path / "m.wav")) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SAMPLE_RATE)
        assert w.getnframes() == len(x)
    with wave.open(str(tmp_path / "s.wav")) as w:
        assert w.getnchannels() == 2
