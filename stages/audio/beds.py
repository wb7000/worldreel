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
        # Soft rain: a low-passed noise wash plus sparse, dull drops. (The first version used
        # ~400 bright 2-12 kHz clicks per second and hiss up to 9 kHz: heard as harsh buzzing,
        # 35 % of its energy above 4 kHz, D-5.5.)
        wash = _lowpass(_lowpass(rng.standard_normal(n), 1500.0), 3500.0) * 0.25
        drops = np.zeros(n)
        for i in rng.integers(0, max(n - 600, 1), int(duration * 60)):
            drops[i : i + 512] += (
                rng.uniform(0.2, 0.6) * np.exp(-np.arange(512) / 60.0) * rng.standard_normal(512)
            )
        out = out + wash + _lowpass(drops, 2500.0) * 0.4
    peak = float(np.abs(out).max()) or 1.0
    return np.asarray(out / peak * 0.5, np.float32)


def _warm_tone(freq: F64 | float, t: F64, rng: np.random.Generator, voices: int = 3) -> F64:
    """Soft, slightly detuned band-limited sawtooth (6 harmonics, 1/k), several voices: a warm
    chorus with slow (< 1 Hz) movement instead of the buzzing of pure-sine clusters (D-5.5)."""
    out = np.zeros(len(t))
    for v in range(voices):
        detune = 1.0 + (v - (voices - 1) / 2) * 0.0015
        phase = 2 * np.pi * np.cumsum(np.broadcast_to(freq, t.shape) * detune) / SAMPLE_RATE
        phase = phase + rng.uniform(0, 6.28)
        for k in range(1, 7):
            out += np.sin(k * phase) / k
    return out / voices


def music(duration: float, mood: list[dict[str, Any]], seed: int = 0) -> F32:
    """Warm pad + heartbeat pulse driven by tension (D-5.5 redesign; D-5.3 was pure sines that
    buzzed). Chord: root, fifth, octave and a third that *glides* from major (5/4) to minor (6/5)
    as tension rises — one voice, so no two thirds beat against each other. All low-passed at
    900 Hz. Pulse: a soft low double thump (heartbeat-like) from tension 4, 60 -> 120 bpm."""
    rng = np.random.default_rng(seed)
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    tension = tension_curve(mood, duration)
    tense = np.clip((tension - 3.0) / 7.0, 0.0, 1.0)  # 0 calm .. 1 intense
    root = 110.0  # A2
    third = root * (1.25 + (1.2 - 1.25) * tense)
    pad = (
        _warm_tone(root, t, rng)
        + _warm_tone(root * 1.5, t, rng) * 0.8
        + _warm_tone(root * 2.0, t, rng) * 0.6
        + _warm_tone(third, t, rng) * 0.7
    )
    pad = (
        _lowpass(_lowpass(pad, 900.0), 900.0) * 0.07 * (0.85 + 0.15 * np.sin(2 * np.pi * 0.08 * t))
    )
    drone = _lowpass(_warm_tone(root / 2, t, rng), 300.0) * 0.12 * tense
    bpm = 60.0 + 60.0 * tense
    phase = np.cumsum(bpm / 60.0 / SAMPLE_RATE)
    beat_pos = phase - np.floor(phase)
    since = beat_pos / (bpm / 60.0)
    second = np.clip(since - 0.22, 0.0, None)

    def thump(s: F64) -> F64:
        return (
            np.exp(-s * 18.0) * np.sin(2 * np.pi * (48.0 + 20.0 * np.exp(-s * 30.0)) * s) * (s > 0)
        )

    pulse = (thump(since) + 0.6 * thump(second)) * np.clip((tension - 3.5) / 4.0, 0.0, 1.0) * 0.45
    out = pad + drone + pulse
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
