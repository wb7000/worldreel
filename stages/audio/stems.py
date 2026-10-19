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
from stages.audio.synth import wav_bytes, wav_samples

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


def mixdown(stems: dict[str, F32]) -> F32:
    n = max((len(s) for s in stems.values()), default=0)
    out = np.zeros(n, np.float64)
    for layer, s in stems.items():
        out[: len(s)] += MIX_GAINS.get(layer, 0.5) * s
    peak = float(np.abs(out).max())
    if peak > 0.99:  # never clip the mix
        out *= 0.99 / peak
    return np.asarray(out, np.float32)


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
