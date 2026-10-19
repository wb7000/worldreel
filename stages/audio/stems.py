"""Render the layered stems of a scene and describe them as `.wreel` audio tracks (spec §11).

Layers (brief §7): foley (footstep events mixed down), ambience (wind / rain), music (mood),
effects and dialogue (empty until there is a source — no track is written for an empty layer).
Every stem is a separate mono 16-bit WAV track, so editors and players get them separately; the
mix is only a sum with fixed gains (`MIX_GAINS`), produced on export, never stored.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from stages.audio.beds import ambience, music, place_clips
from stages.audio.synth import SAMPLE_RATE, wav_bytes, wav_samples

F32 = npt.NDArray[np.float32]
MIX_GAINS = {"dialogue": 1.0, "foley": 0.8, "effects": 0.8, "ambience": 0.5, "music": 0.4}


def render_stems(
    duration: float,
    events: list[dict[str, Any]],
    mood: list[dict[str, Any]],
    clips: dict[str, bytes],
    weather: str = "dry",
    seed: int = 0,
) -> dict[str, F32]:
    decoded = {path: wav_samples(data) for path, data in clips.items()}
    stems: dict[str, F32] = {
        "ambience": ambience(duration, mood, weather, seed),
        "music": music(duration, mood, seed),
    }
    foley = place_clips(duration, [e for e in events if e.get("type") == "footstep"], decoded)
    if np.abs(foley).max() > 0:
        stems["foley"] = foley
    return stems


DEFAULT_TARGET_LUFS = -16.0  # the spec's example value (§11); per-scene in `audio.mix`
DUCK_LAYERS = ("music", "ambience")


def loudness_lufs(x: npt.NDArray[np.floating[Any]]) -> float:
    """Integrated loudness, ITU-R BS.1770-4 (pyloudnorm, MIT). -inf for silence / too short."""
    import pyloudnorm

    a = np.asarray(x, np.float64)
    if len(a) < int(0.4 * SAMPLE_RATE) or not np.any(a):
        return float("-inf")
    return float(pyloudnorm.Meter(SAMPLE_RATE).integrated_loudness(a))


def duck(
    bed: F32, key: F32, depth_db: float = 10.0, attack_s: float = 0.05, release_s: float = 0.5
) -> F32:
    """Lower `bed` by up to `depth_db` while `key` (dialogue) is active; smooth attack/release."""
    n = min(len(bed), len(key))
    win = int(0.02 * SAMPLE_RATE)
    frames = max(n // win, 1)
    env = np.sqrt((key[: frames * win].astype(np.float64).reshape(frames, win) ** 2).mean(1))
    active = (env > 0.01).astype(np.float64)  # about -40 dBFS
    gain_db = np.zeros(frames)
    up, down = win / (release_s * SAMPLE_RATE), win / (attack_s * SAMPLE_RATE)
    g = 0.0
    for i, a in enumerate(active):
        target = -depth_db * a
        step = down if target < g else up
        g += (target - g) * min(1.0, step)
        gain_db[i] = g
    per_sample = np.repeat(10 ** (gain_db / 20), win)[:n]
    out = bed.astype(np.float64).copy()
    out[: len(per_sample)] *= per_sample
    return np.asarray(out, np.float32)


def mixdown(
    stems: dict[str, F32], target_lufs: float | None = DEFAULT_TARGET_LUFS, ducking: bool = True
) -> F32:
    """Gain-staged sum of the stems; music/ambience ducked under dialogue; integrated loudness
    normalised to `target_lufs` (None = no normalising); never clips (peak <= 0.99, in which case
    the achieved loudness is below target — see `mix_report`)."""
    n = max((len(s) for s in stems.values()), default=0)
    out = np.zeros(n, np.float64)
    dialogue = stems.get("dialogue")
    for layer, s in stems.items():
        if ducking and dialogue is not None and layer in DUCK_LAYERS:
            s = duck(s, dialogue)
        out[: len(s)] += MIX_GAINS.get(layer, 0.5) * s
    if target_lufs is not None:
        now = loudness_lufs(out)
        if np.isfinite(now):
            out *= 10 ** ((target_lufs - now) / 20)
    peak = float(np.abs(out).max()) if n else 0.0
    if peak > 0.99:  # never clip the mix
        out *= 0.99 / peak
    return np.asarray(out, np.float32)


def mix_report(stems: dict[str, F32], mix: F32, target_lufs: float | None) -> dict[str, Any]:
    return {
        "target_lufs": target_lufs,
        "mix_lufs": round(loudness_lufs(mix), 2),
        "mix_peak_dbfs": round(float(20 * np.log10(max(float(np.abs(mix).max()), 1e-9))), 2),
        "stems_lufs": {k: round(loudness_lufs(v), 2) for k, v in stems.items()},
    }


def stem_tracks(stems: dict[str, F32]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Track entries (spec §11) and WAV entries for the `.wreel`."""
    tracks, entries = [], {}
    for layer, s in sorted(stems.items()):
        path = f"audio/stems/{layer}.wav"
        entries[path] = wav_bytes(s)
        tracks.append(
            {
                "id": f"stem_{layer}",
                "layer": layer,
                "path": path,
                "media_type": "audio/wav",
                "start": 0.0,
                "license": "synthetic",
                "positional": False,
            }
        )
    return tracks, entries
