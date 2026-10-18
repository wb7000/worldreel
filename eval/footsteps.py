"""Do generated footsteps change with surface and weather? (algorithm A-6, brief Phase 5)

For each surface x {dry, rain}: three seeded variants; per clip the spectral-temporal profile
(energy in 9 octave bands + two decay times, `stages.audio.synth.profile`). (A first version used
four scalar features — centroid, decay, share above 3 kHz, onsets — and could not separate the
conditions: too coarse a description, D-5.1.) A condition is
distinguishable when its closest other condition is farther away (centroid distance) than the
spread of its own variants. Also checks that rain adds high-band energy and a longer tail.

Usage: python -m eval.footsteps [--wav-dir DIR]   (optionally writes the clips to listen to)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from stages.audio.synth import SURFACES, features, footstep, profile, write_wav

KEYS = ("centroid_hz", "decay_ms", "hf_share", "onsets")


def run(wav_dir: Path | None = None, variants: int = 3) -> dict[str, Any]:
    feats: dict[tuple[str, str], list[list[float]]] = {}
    for surface, weather in itertools.product(SURFACES, ("dry", "rain")):
        rows = []
        for v in range(variants):
            clip = footstep({"surface": surface, "weather": weather}, seed=v)
            f = features(clip)
            rows.append(
                [
                    np.log(f["centroid_hz"]),
                    f["decay_ms"],
                    f["hf_share"],
                    f["onsets"],
                    *profile(clip),
                ]
            )
            if wav_dir is not None:
                wav_dir.mkdir(parents=True, exist_ok=True)
                write_wav(wav_dir / f"step_{surface}_{weather}_{v}.wav", clip)
        feats[(surface, weather)] = rows
    # Distances on the profile only (columns 4..); the scalar features are reported for reading.
    allf = np.array([r[4:] for rows in feats.values() for r in rows])
    mu, sd = allf.mean(0), allf.std(0) + 1e-9
    z = {k: (np.array(v)[:, 4:] - mu) / sd for k, v in feats.items()}
    centre = {k: v.mean(0) for k, v in z.items()}
    spread = {k: float(np.linalg.norm(v - centre[k], axis=1).max()) for k, v in z.items()}
    report: dict[str, Any] = {"conditions": {}, "all_distinguishable": True}
    for k in z:
        nearest = min((float(np.linalg.norm(centre[k] - centre[o])), o) for o in z if o != k)
        ok = nearest[0] > 2 * spread[k]
        report["all_distinguishable"] &= ok
        raw = np.array(feats[k]).mean(0)
        report["conditions"][f"{k[0]}/{k[1]}"] = {
            "centroid_hz": round(float(np.exp(raw[0]))),
            "decay_ms": round(float(raw[1])),
            "hf_share": round(float(raw[2]), 3),
            "onsets": round(float(raw[3]), 1),
            "nearest": f"{nearest[1][0]}/{nearest[1][1]}",
            "margin": round(nearest[0] / max(spread[k], 1e-9), 1),
            "distinguishable": ok,
        }
    wet_effect = []
    for s in SURFACES:
        d, w = np.array(feats[(s, "dry")]).mean(0), np.array(feats[(s, "rain")]).mean(0)
        wet_effect.append(
            {"surface": s, "hf_share_up": bool(w[2] > d[2]), "decay_longer": bool(w[1] > d[1])}
        )
    report["rain_vs_dry"] = wet_effect
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wav-dir", type=Path)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.wav_dir), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
