"""Per-emotion voice-quality shaping after synthesis (A-11, docs/ALGORITHMS.md).

The WORLD vocoder (pyworld) splits speech into pitch (f0), spectral envelope (timbre) and
aperiodicity (breathiness); each is edited per emotion and the line is resynthesised. Presets
were set before any listening; the line's intensity interpolates from "no change" to the preset.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

FRAME_MS = 5.0


@dataclass(frozen=True)
class Shape:
    shift_st: float = 0.0  # pitch shift, semitones
    range_x: float = 1.0  # scale of log-f0 around its mean
    tremor_hz: float = 0.0
    tremor_st: float = 0.0
    jitter_st: float = 0.0
    warp: float = 1.0  # spectral-envelope frequency warp (> 1 raises formants: "smiling")
    tilt_db: float = 0.0  # dB per octave above 1 kHz (+ bright / harsh, - dark)
    breath: float = 0.0  # aperiodicity pulled toward 1 (1 = whisper)
    tempo: float = 1.0  # > 1 slower

    def scaled(self, intensity: float) -> Shape:
        """Interpolate from no change (0) to this preset (1)."""
        k = float(np.clip(intensity, 0.0, 1.0))
        neutral = Shape()
        vals = {
            f.name: getattr(neutral, f.name)
            + k * (getattr(self, f.name) - getattr(neutral, f.name))
            for f in fields(self)
        }
        return replace(neutral, **vals)

    def amplified(self, k: float) -> Shape:
        """Like `scaled` but allows k > 1 (stronger than the preset) and keeps timing (tempo 1);
        used on whole-conversation renders whose timing must not move (D-8.6)."""
        neutral = Shape()
        vals = {
            f.name: getattr(neutral, f.name)
            + k * (getattr(self, f.name) - getattr(neutral, f.name))
            for f in fields(self)
        }
        vals["tempo"] = 1.0
        vals["breath"] = min(vals["breath"], 0.9)
        vals["range_x"] = max(vals["range_x"], 0.3)
        return replace(neutral, **vals)


PRESETS: dict[str, Shape] = {
    "excited": Shape(2.0, 1.5, warp=1.06, tilt_db=2.5, breath=0.10, tempo=0.93),
    "joyful": Shape(2.0, 1.5, warp=1.06, tilt_db=2.5, breath=0.10, tempo=0.93),
    "comic_sarcastic": Shape(0.5, 1.7, warp=1.03, tilt_db=1.0, tempo=1.05),
    "despair": Shape(-1.5, 0.7, warp=0.98, tilt_db=-3.0, breath=0.25, tempo=1.12),
    "crying": Shape(1.0, 1.2, 6.0, 0.8, 0.3, tilt_db=-1.0, breath=0.35, tempo=1.08),
    "angry": Shape(1.0, 1.4, tilt_db=4.0, tempo=0.95),
    "afraid": Shape(2.5, 1.3, 7.0, 0.4, 0.15, warp=1.02, tilt_db=1.0, breath=0.30, tempo=0.90),
    "whisper_tense": Shape(0.0, 0.8, tilt_db=1.0, breath=0.85),
    "shocked_grief": Shape(1.5, 1.4, 5.5, 0.6, 0.2, tilt_db=-1.0, breath=0.30, tempo=1.05),
    "tender": Shape(-0.5, 0.9, tilt_db=-1.5, breath=0.20, tempo=1.05),
    "dry_humour": Shape(0.0, 0.8),
    "neutral": Shape(),
}


def _warp_envelope(sp: np.ndarray, alpha: float) -> np.ndarray:
    if abs(alpha - 1.0) < 1e-6:
        return sp
    bins = np.arange(sp.shape[1], dtype=np.float64)
    src = np.clip(bins / alpha, 0, sp.shape[1] - 1)
    out = np.empty_like(sp)
    for t in range(sp.shape[0]):
        out[t] = np.interp(src, bins, sp[t])
    return out


def _tilt(sp: np.ndarray, fs: int, db_per_oct: float) -> np.ndarray:
    if db_per_oct == 0.0:
        return sp
    freqs = np.linspace(0, fs / 2, sp.shape[1])
    octaves = np.log2(np.maximum(freqs, 1000.0) / 1000.0)
    gain_power = 10 ** (db_per_oct * octaves / 10)  # sp is a power spectrum
    out: np.ndarray = sp * gain_power[None, :]
    return out


def _retime(arr: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return arr
    n = len(arr)
    m = max(2, round(n * factor))
    pos = np.linspace(0, n - 1, m)
    idx = np.clip(np.round(pos).astype(int), 0, n - 1)
    picked: np.ndarray = arr[idx]
    return picked


def shape_voice(x: np.ndarray, fs: int, shape: Shape, seed: int = 0) -> np.ndarray:
    """Return the line resynthesised with `shape` applied (float64, same loudness as `x`)."""
    import pyworld as pw

    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.abs(x).max()
    if peak > 1.5:  # int PCM
        x = x / 32768.0
    f0, sp, ap = pw.wav2world(x, fs, frame_period=FRAME_MS)
    voiced = f0 > 0
    if voiced.any():
        lf = np.zeros_like(f0)
        lf[voiced] = np.log2(f0[voiced])
        mean = lf[voiced].mean()
        lf[voiced] = mean + shape.shift_st / 12 + shape.range_x * (lf[voiced] - mean)
        t = np.arange(len(f0)) * FRAME_MS / 1000
        if shape.tremor_st:
            lf[voiced] += shape.tremor_st / 12 * np.sin(2 * np.pi * shape.tremor_hz * t[voiced])
        if shape.jitter_st:
            rng = np.random.default_rng(seed)
            lf[voiced] += rng.normal(0, shape.jitter_st / 12, voiced.sum())
        f0 = np.where(voiced, 2**lf, 0.0)
    sp = _tilt(_warp_envelope(sp, shape.warp), fs, shape.tilt_db)
    ap = np.clip(ap + shape.breath * (1 - ap), 0.0, 1.0)
    f0, sp, ap = (_retime(a, shape.tempo) for a in (f0, sp, ap))
    y: np.ndarray = pw.synthesize(
        np.ascontiguousarray(f0), np.ascontiguousarray(sp), np.ascontiguousarray(ap), fs, FRAME_MS
    )
    rms_in = np.sqrt(np.mean(x**2)) + 1e-12
    rms_out = np.sqrt(np.mean(y**2)) + 1e-12
    y = y * (rms_in / rms_out)
    clipped: np.ndarray = np.clip(y, -1.0, 1.0)
    return clipped
