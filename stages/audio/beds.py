"""Generated ambience and mood music driven by the tension curve (brief §7 mood layer).

Everything is synthesised (numpy + scipy), seeded and deterministic: nothing to license.
The tension curve is the spec's `audio.mood` list ({t, tension 0-10}), read as a step function
like the player does, then smoothed so changes glide instead of jumping.

Music (own simple procedural scheme, not a model):
  * a pad chord: major (A major-ish) when calm, crossfading to minor plus a low drone when tense;
  * a pulse whose tempo rises from 60 to 140 bpm with tension and whose level and brightness
    grow with it (silent below tension 3).
Ambience: wind (low-passed noise with slow gusts, deeper with tension) and, if raining, rain
(dense high droplets + broadband hiss).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfilt

from stages.audio.synth import SAMPLE_RATE

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]


def tension_curve(mood: list[dict[str, Any]], duration: float, rate: int = SAMPLE_RATE) -> F64:
    """Per-sample tension (0-10): step function of the mood points, smoothed over ~1.5 s."""
    n = int(duration * rate)
    t = np.arange(n) / rate
    curve = np.full(n, 5.0)
    for m in sorted(mood, key=lambda m: float(m["t"])):
        curve[t >= float(m["t"])] = float(np.clip(m["tension"], 0, 10))
    k = int(1.5 * rate)
    if n > k:
        # Moving average via a running sum: O(n). (A direct convolution with a 1.5 s kernel was
        # O(n k) and took minutes for 20 s of audio.)
        padded = np.concatenate(
            [np.full(k // 2, curve[0]), curve, np.full(k - k // 2 - 1, curve[-1])]
        )
        csum = np.concatenate([[0.0], np.cumsum(padded)])
        curve = (csum[k:] - csum[:-k]) / k
    return curve


def _lowpass(x: F64, freq: float, order: int = 2) -> F64:
    sos = butter(order, freq, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    return np.asarray(sosfilt(sos, x), np.float64)


def _bandpass(x: F64, lo: float, hi: float, order: int = 2) -> F64:
    sos = butter(order, (lo, hi), btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return np.asarray(sosfilt(sos, x), np.float64)


def ambience(
    duration: float, mood: list[dict[str, Any]], weather: str = "dry", seed: int = 0
) -> F32:
    rng = np.random.default_rng(seed)
    n = int(duration * SAMPLE_RATE)
    tension = tension_curve(mood, duration)
    t = np.arange(n) / SAMPLE_RATE
    gust = 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t + rng.uniform(0, 6.28)) * np.sin(
        2 * np.pi * 0.023 * t
    )
    wind = _lowpass(rng.standard_normal(n), 400.0) * (0.15 + 0.25 * gust * (0.3 + tension / 10))
    out = wind
    if weather == "rain":
        hiss = _bandpass(rng.standard_normal(n), 800.0, 9000.0) * 0.06
        drops = np.zeros(n)
        count = int(duration * 400)  # ~400 audible droplets per second
        for i in rng.integers(0, max(n - 200, 1), count):
            drops[i : i + 64] += (
                rng.uniform(0.1, 0.5) * np.exp(-np.arange(64) / 8.0) * rng.choice([-1, 1])
            )
        out = out + hiss + _bandpass(drops, 2000.0, 12000.0) * 0.5
    peak = float(np.abs(out).max()) or 1.0
    return np.asarray(out / peak * 0.5, np.float32)


def music(duration: float, mood: list[dict[str, Any]], seed: int = 0) -> F32:
    """Fixed layer gains (no per-render normalising: that made the music quieter as tension added
    layers, D-5.3); peak <= 0.32 + 0.1 + 0.1 + 0.35 < 1 by construction."""
    rng = np.random.default_rng(seed)
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    tension = tension_curve(mood, duration)
    tense = np.clip((tension - 3.0) / 7.0, 0.0, 1.0)  # 0 calm .. 1 intense
    root = 110.0  # A2
    major = [1.0, 5 / 4, 3 / 2, 2.0]
    minor = [1.0, 6 / 5, 3 / 2, 2.0]
    pad_major = sum(np.sin(2 * np.pi * root * 2 * r * t + rng.uniform(0, 6.28)) for r in major)
    pad_minor = sum(np.sin(2 * np.pi * root * 2 * r * t + rng.uniform(0, 6.28)) for r in minor)
    breath = 0.8 + 0.2 * np.sin(2 * np.pi * 0.1 * t)
    # Equal-power crossfade (linear weights lost ~3 dB halfway between the chords, D-5.3).
    pad = (
        (np.cos(tense * np.pi / 2) * pad_major + np.sin(tense * np.pi / 2) * pad_minor)
        * 0.08
        * breath
    )
    drone = np.sin(2 * np.pi * root * t) * 0.1 * tense  # A2, grows with tension
    shimmer = sum(np.sin(2 * np.pi * root * 8 * r * t + rng.uniform(0, 6.28)) for r in minor)
    shimmer = shimmer * 0.025 * tense  # high partials: brighter when tense
    # Pulse: phase accumulates with a tension-dependent tempo (60 -> 140 bpm).
    bpm = 60.0 + 80.0 * tense
    phase = np.cumsum(bpm / 60.0 / SAMPLE_RATE)
    beat_pos = phase - np.floor(phase)
    since_beat = beat_pos / (bpm / 60.0)  # seconds since the last beat
    click = np.exp(-since_beat * 30.0) * np.sin(2 * np.pi * (300.0 + 900.0 * tense) * since_beat)
    pulse = click * np.clip((tension - 3.0) / 4.0, 0.0, 1.0) * 0.35
    out = pad + drone + shimmer + pulse
    return np.asarray(np.clip(out, -1.0, 1.0), np.float32)


def place_clips(duration: float, events: list[dict[str, Any]], clips: dict[str, F32]) -> F32:
    """Mono mixdown of event clips at their times (the non-positional foley stem)."""
    n = int(duration * SAMPLE_RATE)
    out = np.zeros(n, np.float64)
    for e in events:
        clip = clips.get(str(e.get("attributes", {}).get("clip", "")))
        if clip is None:
            continue
        i = int(float(e["t"]) * SAMPLE_RATE)
        if i >= n:
            continue
        seg = clip[: n - i]
        out[i : i + len(seg)] += seg
    return np.asarray(out, np.float32)
