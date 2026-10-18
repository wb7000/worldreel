"""Procedural footstep synthesis (algorithm A-6, docs/ALGORITHMS.md) and audio feature helpers.

Every sound is generated (numpy + scipy.signal), deterministic per seed: no recorded libraries,
so nothing to license (brief §7). Mono float32 in [-1, 1] at `SAMPLE_RATE`.

Attribute vocabulary (spec §11 leaves it to this engine):
  surface  : concrete | stone | wood | gravel | grass | sand | dirt     (default concrete)
  wet      : bool                                                       (default False)
  footwear : boot | sneaker | heels | barefoot                          (default sneaker)
  weight   : kg                                                         (default 70)
  speed    : m/s                                                        (default 1.4)
  weather  : dry | rain                                                 (rain implies wet)
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 44100
F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

SURFACES = ("concrete", "stone", "wood", "gravel", "grass", "sand", "dirt")
FOOTWEAR = ("boot", "sneaker", "heels", "barefoot")


FilterKind = Literal["lowpass", "highpass", "bandpass"]


def _filt(x: F64, kind: FilterKind, freq: float | tuple[float, float], order: int = 2) -> F64:
    sos = butter(order, freq, btype=kind, fs=SAMPLE_RATE, output="sos")
    return np.asarray(sosfilt(sos, x), np.float64)


def _env(n: int, decay_s: float, attack_s: float = 0.0015) -> F64:
    t = np.arange(n) / SAMPLE_RATE
    attack = np.clip(t / max(attack_s, 1e-6), 0.0, 1.0)
    return np.asarray(attack * np.exp(-t / max(decay_s, 1e-6)), np.float64)


def _burst(rng: np.random.Generator, length_s: float, decay_s: float) -> F64:
    n = int(length_s * SAMPLE_RATE)
    return rng.standard_normal(n) * _env(n, decay_s)


def _modes(freqs: list[float], decay_s: float, length_s: float, rng: np.random.Generator) -> F64:
    n = int(length_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    out = np.zeros(n)
    for f in freqs:
        out += rng.uniform(0.85, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return out * _env(n, decay_s)


def _impact(surface: str, footwear: str, weight: float, rng: np.random.Generator) -> F64:
    """One transient (heel or toe) on a surface."""
    heavy = float(np.clip(weight / 70.0, 0.4, 2.0))
    length = 0.25
    n = int(length * SAMPLE_RATE)

    def grains(
        count: tuple[int, int], spread_s: float, band: tuple[float, float], gain: float
    ) -> F64:
        g = np.zeros(n)
        k = int(rng.integers(*count))
        for start in rng.exponential(spread_s / max(k, 1), k).cumsum():
            i = int(start * SAMPLE_RATE)
            if i >= n - 400:
                break
            click = _filt(_burst(rng, 0.008, 0.0015), "bandpass", band)
            g[i : i + len(click)] += click * rng.uniform(0.6, 1.0)
        return g * gain

    thump = _modes([rng.uniform(86, 94)], 0.03, length, rng) * heavy
    if surface == "concrete":  # crisp slap (1-6 kHz) + body; dirt is the muffled one
        x = _filt(_burst(rng, length, 0.01), "bandpass", (1000.0, 6000.0)) + thump * 0.6
    elif surface == "stone":  # flagstones ring: bright crack + mid/high resonances
        crack = _filt(_burst(rng, length, 0.008), "highpass", 1500.0)
        ring = _modes(
            [rng.uniform(1160, 1240), rng.uniform(2000, 2100), rng.uniform(3000, 3100)],
            0.03,
            length,
            rng,
        )
        x = crack + ring * 0.5 + thump * 0.4
    elif surface == "wood":  # hollow boards: low/mid modes that ring
        body = _modes(
            [rng.uniform(205, 215), rng.uniform(410, 430), rng.uniform(745, 775)], 0.07, length, rng
        )
        x = body + _filt(_burst(rng, length, 0.006), "highpass", 2000.0) * 0.3 + thump * 0.4
    elif surface == "gravel":  # crunchy: many bright grains
        x = grains((36, 44), 0.12, (2500.0, 9000.0), 1.0) + thump * 0.3
    elif surface == "grass":  # soft swish, no click
        x = _filt(_burst(rng, length, 0.07), "bandpass", (300.0, 3000.0)) * 0.6
    elif surface == "sand":  # fine-grained soft "shh" (1-4 kHz grains), almost no body
        x = grains((58, 66), 0.15, (1000.0, 4000.0), 0.6) + thump * 0.1
    else:  # dirt: muffled thud + a few crumbly mid clicks
        x = (
            _filt(_burst(rng, length, 0.04), "lowpass", 1800.0)
            + grains((8, 10), 0.08, (800.0, 2500.0), 0.6)
            + thump * 0.5
        )
    if footwear == "sneaker":
        x = _filt(x, "lowpass", 5000.0)
    elif footwear == "barefoot":
        x = _filt(x, "lowpass", 2500.0) * 0.6
    elif footwear == "heels":
        x = x * 0.5 + _filt(_burst(rng, length, 0.004), "highpass", 3500.0) * 0.8
    elif footwear == "boot":
        x = x + _modes([rng.uniform(60, 80)], 0.04, length, rng) * 0.3 * heavy
    return x


def _splash(rng: np.random.Generator, length_s: float, soft: bool) -> F64:
    """Water. Hard surfaces splash: a long spray tail (150-250 ms) plus a low squelch. Soft ground
    (sand, dirt, grass) soaks the water up and squelches instead: mostly a longer low-mid squelch
    with little spray. A single spray for every surface made wet sand and wet dirt alike (D-5.1)."""
    spray = _filt(_burst(rng, length_s, rng.uniform(0.18, 0.22)), "bandpass", (2500.0, 7000.0))
    if soft:
        squelch = _filt(_burst(rng, length_s, rng.uniform(0.12, 0.14)), "bandpass", (200.0, 700.0))
        return squelch * 1.0 + spray * 0.15
    squelch = _filt(_burst(rng, length_s, 0.08), "bandpass", (300.0, 900.0))
    return spray * 0.6 + squelch * 0.8


def footstep(attributes: dict[str, Any], seed: int = 0) -> F32:
    """One footstep (heel + toe) for the given attributes; peak-normalised to the loudness that
    weight and speed imply (max 0.9)."""
    rng = np.random.default_rng(seed)
    surface = str(attributes.get("surface", "concrete"))
    if surface not in SURFACES:
        surface = "concrete"
    footwear = str(attributes.get("footwear", "sneaker"))
    if footwear not in FOOTWEAR:
        footwear = "sneaker"
    weight = float(attributes.get("weight", 70.0))
    speed = float(attributes.get("speed", 1.4))
    wet = bool(attributes.get("wet", False)) or attributes.get("weather") == "rain"
    gap = float(np.clip(0.11 - 0.02 * (speed - 1.4), 0.05, 0.13))  # heel -> toe
    total = 0.45
    out = np.zeros(int(total * SAMPLE_RATE))
    heel = _impact(surface, footwear, weight, rng)
    toe = _impact(surface, footwear, weight, rng) * 0.45
    out[: len(heel)] += heel
    i = int(gap * SAMPLE_RATE)
    out[i : i + len(toe)] += toe[: len(out) - i]
    if wet:
        # Splash scaled to the impact's own energy, so the surface stays recognisable under it.
        sp = _splash(rng, 0.4, soft=surface in ("sand", "dirt", "grass"))
        rms_heel = float(np.sqrt(np.mean(heel**2))) or 1.0
        rms_sp = float(np.sqrt(np.mean(sp**2))) or 1.0
        sp *= 0.7 * rms_heel / rms_sp
        start = int(0.01 * SAMPLE_RATE)
        out[start : start + len(sp)] += sp[: len(out) - start]
    level = float(np.clip(0.35 * (weight / 70.0) ** 0.5 * (0.7 + 0.3 * speed / 1.4), 0.1, 0.9))
    peak = float(np.abs(out).max()) or 1.0
    fade = np.ones(len(out))
    fade[-int(0.02 * SAMPLE_RATE) :] = np.linspace(1, 0, int(0.02 * SAMPLE_RATE))
    return np.asarray(out / peak * level * fade, np.float32)


# ---- features (for measurement) -------------------------------------------------------------


def features(x: npt.NDArray[np.floating[Any]]) -> dict[str, float]:
    """Spectral centroid (Hz), decay time to -20 dB of the peak RMS (ms), share of energy above
    3 kHz, and onset count (peaks of the 2 ms RMS envelope above -18 dB of its maximum)."""
    x = np.asarray(x, np.float64)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    total = float(spec.sum()) or 1.0
    centroid = float((freqs * spec).sum() / total)
    hf = float(spec[freqs >= 3000.0].sum() / total)
    win = int(0.002 * SAMPLE_RATE)
    frames = len(x) // win
    rms = np.sqrt((x[: frames * win].reshape(frames, win) ** 2).mean(1) + 1e-20)
    db = 20 * np.log10(rms / rms.max())
    above = np.nonzero(db > -20.0)[0]
    decay_ms = float((above.max() - db.argmax()) * 2.0) if len(above) else 0.0
    peaks = (db[1:-1] > db[:-2]) & (db[1:-1] >= db[2:]) & (db[1:-1] > -18.0)
    return {
        "centroid_hz": centroid,
        "decay_ms": decay_ms,
        "hf_share": hf,
        "onsets": float(peaks.sum()),
    }


OCTAVE_CENTRES = (63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def profile(x: npt.NDArray[np.floating[Any]]) -> F64:
    """Spectral-temporal profile: energy per octave band (dB relative to the total; 9 bands,
    63 Hz - 16 kHz) plus decay times to -10 and -20 dB (in units of 10 ms)."""
    x = np.asarray(x, np.float64)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    total = float(spec.sum()) or 1.0
    bands = []
    for c in OCTAVE_CENTRES:
        sel = (freqs >= c / np.sqrt(2)) & (freqs < c * np.sqrt(2))
        bands.append(10 * np.log10(max(float(spec[sel].sum()) / total, 1e-9)))
    win = int(0.002 * SAMPLE_RATE)
    frames = len(x) // win
    rms = np.sqrt((x[: frames * win].reshape(frames, win) ** 2).mean(1) + 1e-20)
    db = 20 * np.log10(rms / rms.max())
    peak = int(db.argmax())
    decays = []
    for level in (-10.0, -20.0):
        above = np.nonzero(db > level)[0]
        decays.append((above.max() - peak) * 0.2 if len(above) else 0.0)  # 10 ms units
    return np.array(bands + decays, np.float64)


def write_wav(path: Path, x: npt.NDArray[np.floating[Any]], rate: int = SAMPLE_RATE) -> None:
    """16-bit PCM mono or stereo (shape (n,) or (n, 2)) with the standard-library wave module."""
    a = np.asarray(x, np.float64)
    pcm = (np.clip(a, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1 if a.ndim == 1 else a.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def wav_bytes(x: npt.NDArray[np.floating[Any]], rate: int = SAMPLE_RATE) -> bytes:
    """The same 16-bit PCM WAV as `write_wav`, in memory (for `.wreel` entries)."""
    import io

    a = np.asarray(x, np.float64)
    pcm = (np.clip(a, -1.0, 1.0) * 32767.0).round().astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1 if a.ndim == 1 else a.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
