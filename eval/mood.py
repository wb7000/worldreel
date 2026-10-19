"""Does the generated music follow the tension value? (brief §7 mood layer)

For tension 0, 2, ..., 10: 20 s of `stages.audio.beds.music` at constant tension. Measured from
the audio itself: RMS level (dBFS), spectral centroid (Hz), and tempo — the autocorrelation peak
of the onset envelope between 40 and 200 bpm (0 when no periodic pulse is found).

Usage: python -m eval.mood
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from stages.audio.beds import music
from stages.audio.synth import SAMPLE_RATE


def tempo_bpm(x: np.ndarray) -> float:
    """Tempo from the onset envelope's autocorrelation (40-200 bpm), or 0 when there is no real
    rhythmic signal: onset energy must be a visible share of the envelope and the periodicity
    peak strong (a first version reported 144 bpm for pulse-free pads, D-5.3)."""
    # Energy low-passed at 8 Hz (beats up to 480 bpm survive), then sampled every 11.6 ms. Plain
    # windowed RMS aliased against the pad's 55 Hz pattern and faked a 144 bpm "pulse" (D-5.3).
    hop = 512
    sos = butter(2, 8.0, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    env = np.sqrt(np.maximum(sosfilt(sos, x**2), 0.0))[::hop]
    onset = np.maximum(np.diff(env), 0.0)
    if onset.mean() < 0.01 * env.mean():
        return 0.0
    onset = onset - onset.mean()
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    fps = SAMPLE_RATE / hop
    lags = np.arange(len(ac)) / fps
    sel = (lags >= 60 / 200) & (lags <= 60 / 40)
    if not sel.any() or ac[0] <= 0:
        return 0.0
    best = int(np.argmax(np.where(sel, ac, -np.inf)))
    if ac[best] / ac[0] < 0.5:
        return 0.0
    # Prefer the shortest lag with a comparable peak (avoids picking 2x the beat period).
    for cand in range(int(np.ceil(fps * 60 / 200)), best):
        if ac[cand] >= 0.9 * ac[best] and ac[cand] >= ac[cand - 1] and ac[cand] >= ac[cand + 1]:
            best = cand
            break
    return float(60.0 / lags[best])


def run() -> list[dict[str, Any]]:
    rows = []
    for tension in range(0, 11, 2):
        x = music(20.0, [{"t": 0.0, "tension": tension}]).astype(np.float64)
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
        rows.append(
            {
                "tension": tension,
                "rms_dbfs": round(float(20 * np.log10(np.sqrt((x**2).mean()) + 1e-12)), 1),
                "centroid_hz": round(float((freqs * spec).sum() / spec.sum())),
                "tempo_bpm": round(tempo_bpm(x), 1),
            }
        )
    return rows


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
