"""Pitch modification without resynthesis: TD-PSOLA (A-11 v3, D-8.6e).

The vocoder in A-11 v1/v2 rebuilds the waveform from pitch / envelope / aperiodicity, which loses
the original phase — heard by the owner as an echo / vibration ("phasey"). TD-PSOLA keeps the
real waveform: two-period, Hann-windowed grains are cut around pitch marks and overlap-added at
the new pitch spacing; unvoiced sound is copied untouched; duration does not change.
Brightness is a separate linear filter (`tilt`), so nothing is resynthesised.
"""

from __future__ import annotations

import numpy as np

FRAME_MS = 5.0


def f0_track(x: np.ndarray, fs: int) -> np.ndarray:
    """Harvest f0 per 5 ms frame (0 = unvoiced). Analysis only — nothing is resynthesised."""
    import pyworld as pw

    f0, _ = pw.harvest(x.astype(np.float64), fs, frame_period=FRAME_MS)
    out: np.ndarray = f0
    return out


def pitch_marks(x: np.ndarray, fs: int, f0: np.ndarray) -> np.ndarray:
    """One mark per period in voiced regions, snapped to the largest |x| near the expected place."""
    hop = int(fs * FRAME_MS / 1000)
    marks: list[int] = []
    t = 0
    n = len(x)
    while t < n:
        fr = min(t // hop, len(f0) - 1)
        if f0[fr] <= 0:
            t += hop // 2 or 1
            continue
        period = int(fs / f0[fr])
        lo, hi = max(0, t - period // 4), min(n, t + period // 4 + 1)
        m = lo + int(np.argmax(np.abs(x[lo:hi]))) if hi > lo else t
        if not marks or m - marks[-1] > period // 2:
            marks.append(m)
        t = m + period
    return np.array(marks, dtype=np.int64)


def psola(x: np.ndarray, fs: int, ratio: np.ndarray, f0: np.ndarray | None = None) -> np.ndarray:
    """Change pitch by `ratio` per 5 ms frame (1 = unchanged), same duration. Unvoiced samples
    and frames with ratio 1 everywhere are returned exactly."""
    x = np.asarray(x, dtype=np.float64)
    if f0 is None:
        f0 = f0_track(x, fs)
    if np.allclose(ratio, 1.0):
        return x.copy()
    hop = int(fs * FRAME_MS / 1000)
    marks = pitch_marks(x, fs, f0)
    if len(marks) < 3:
        return x.copy()
    voiced = np.repeat(f0 > 0, hop)[: len(x)]
    voiced = np.pad(voiced, (0, max(0, len(x) - len(voiced))))
    y = np.zeros_like(x)
    wsum = np.zeros_like(x)
    # synthesis marks: walk the time axis with the new period, reuse the nearest analysis grain
    t = float(marks[0])
    while t < marks[-1]:
        k = int(np.argmin(np.abs(marks - t)))
        m = int(marks[k])
        fr = min(m // hop, len(f0) - 1)
        if f0[fr] <= 0:
            t += hop
            continue
        period = int(fs / f0[fr])
        r = float(ratio[min(int(t) // hop, len(ratio) - 1)])
        lo, hi = m - period, m + period
        if lo < 0 or hi > len(x):
            t += period / max(r, 1e-3)
            continue
        grain = x[lo:hi] * np.hanning(hi - lo)
        c = round(t)
        a, b = c - period, c + period
        if a >= 0 and b <= len(x):
            y[a:b] += grain
            wsum[a:b] += np.hanning(hi - lo)
        t += period / max(r, 1e-3)
    out = np.where(wsum > 1e-3, y / np.maximum(wsum, 1e-3), x)
    # keep unvoiced parts (and voiced parts the grains did not reach) exactly as they were
    out = np.where(voiced & (wsum > 0.2), out, x)
    return out


def tilt(x: np.ndarray, fs: int, db_per_oct: float) -> np.ndarray:
    """Spectral tilt above 1 kHz as a linear-phase FIR filter (no resynthesis)."""
    if db_per_oct == 0.0:
        return np.asarray(x, dtype=np.float64).copy()
    from scipy.signal import firwin2, oaconvolve

    nyq = fs / 2
    freqs = np.linspace(0, nyq, 64)
    gain = 10 ** (db_per_oct * np.log2(np.maximum(freqs, 1000.0) / 1000.0) / 20)
    taps = firwin2(255, freqs / nyq, gain)
    y = oaconvolve(np.asarray(x, dtype=np.float64), taps, mode="same")
    out: np.ndarray = y
    return out
