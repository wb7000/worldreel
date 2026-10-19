"""Character bible: explicit, seeded identity codes and distinctness checks (A-8, brief §5).

Identity is never chosen by a generator: each character has a code made from a `seed` and an
optional style request. The code expands deterministically to an appearance vector and a voice
vector, every dimension on a [0, 1] scale; requested styles narrow dimensions to ranges
(`stages.safety.likeness`). The expanded vectors are stored in the `.wscript` so a reader can
check them against the seed.

Distinctness (set before measuring, D-7.4): any two characters of one film must differ by at
least `APPEARANCE_MIN` and `VOICE_MIN` — RMS difference over the dimensions, i.e. on average a
fifth of each dimension's range. This is a check in the identity-parameter space; once faces and
voices are rendered (Phase 8) the same check must run on face / speaker embeddings of the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from stages.safety.likeness import Range, parse_style

APPEARANCE = (
    "age",
    "presentation",
    "height",
    "build",
    "face_width",
    "jaw",
    "brow",
    "eye_size",
    "eye_spacing",
    "nose_length",
    "nose_width",
    "mouth_width",
    "lip_fullness",
    "skin_tone",
    "hair_tone",
    "hair_length",
    "facial_hair",
)
VOICE = ("pitch", "formant", "rate", "breathiness", "roughness")
APPEARANCE_MIN = 0.20
VOICE_MIN = 0.20
CANDIDATES = 32  # best-candidate sampling (A-8); 1 = independent seeds (baseline)


@dataclass(frozen=True)
class Identity:
    seed: int
    requested: str
    appearance: dict[str, float]
    voice: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "requested": self.requested,
            "appearance": {k: round(v, 4) for k, v in self.appearance.items()},
            "voice": {k: round(v, 4) for k, v in self.voice.items()},
        }


def expand(seed: int, requested: str = "") -> Identity:
    """Seed + style request -> identity. Raises ValueError for a refused request."""
    ranges: dict[str, Range] = {}
    if requested.strip():
        req = parse_style(requested)
        if not req.allowed:
            raise ValueError(f"style request refused: {req.reason}")
        ranges = dict(req.ranges)
        # Default, not a rule: a feminine presentation is clean-shaven unless facial hair is asked.
        if ranges.get("presentation", (0.0, 1.0))[1] <= 0.35 and "facial_hair" not in ranges:
            ranges["facial_hair"] = (0.0, 0.1)
    u = np.random.default_rng(seed).random(len(APPEARANCE) + len(VOICE))

    def value(dim: str, x: float) -> float:
        lo, hi = ranges.get(dim, (0.0, 1.0))
        return float(lo + x * (hi - lo))

    names = APPEARANCE + VOICE
    vals = {d: value(d, float(x)) for d, x in zip(names, u, strict=True)}
    return Identity(
        seed,
        requested,
        {d: vals[d] for d in APPEARANCE},
        {d: vals[d] for d in VOICE},
    )


def _vec(values: dict[str, float], dims: tuple[str, ...]) -> np.ndarray:
    return np.array([values[d] for d in dims])


def distance(a: Identity, b: Identity) -> tuple[float, float]:
    """(appearance, voice) RMS difference, each in [0, 1]."""
    da = _vec(a.appearance, APPEARANCE) - _vec(b.appearance, APPEARANCE)
    dv = _vec(a.voice, VOICE) - _vec(b.voice, VOICE)
    return float(np.sqrt(np.mean(da**2))), float(np.sqrt(np.mean(dv**2)))


@dataclass
class DistinctnessReport:
    ok: bool
    min_appearance: float
    min_voice: float
    too_close: list[tuple[str, str, str, float]] = field(default_factory=list)


def check_distinct(cast: dict[str, Identity]) -> DistinctnessReport:
    min_a = min_v = 1.0
    close: list[tuple[str, str, str, float]] = []
    for (ia, a), (ib, b) in combinations(sorted(cast.items()), 2):
        da, dv = distance(a, b)
        min_a, min_v = min(min_a, da), min(min_v, dv)
        if da < APPEARANCE_MIN:
            close.append((ia, ib, "appearance", round(da, 3)))
        if dv < VOICE_MIN:
            close.append((ia, ib, "voice", round(dv, 3)))
    return DistinctnessReport(not close, round(min_a, 4), round(min_v, 4), close)


def sample_cast(
    requests: dict[str, str], cast_seed: int, candidates: int = CANDIDATES
) -> dict[str, Identity]:
    """Pick a seed per character, in key order. Each character takes the best of `candidates`
    seeds: the one whose closest already-chosen character is farthest away (the smaller of the
    appearance and voice distance, each relative to its threshold). Deterministic per cast seed."""
    rng = np.random.default_rng(cast_seed)
    cast: dict[str, Identity] = {}
    for cid in sorted(requests):
        best: Identity | None = None
        best_score = -1.0
        for seed in rng.integers(0, 2**31 - 1, size=candidates):
            cand = expand(int(seed), requests[cid])
            if not cast:
                best = cand
                break
            score = min(
                min(da / APPEARANCE_MIN, dv / VOICE_MIN)
                for da, dv in (distance(cand, other) for other in cast.values())
            )
            if score > best_score:
                best, best_score = cand, score
        assert best is not None
        cast[cid] = best
    return cast


def unique_identities(batch: list[Identity], bins: int = 4) -> int:
    """Collapse counter: identities that stay distinct after quantising every dimension into
    `bins` levels. A generator collapsing to one face gives 1 for any batch size."""
    keys = {
        tuple(
            min(int(v * bins), bins - 1)
            for v in [*_vec(i.appearance, APPEARANCE), *_vec(i.voice, VOICE)]
        )
        for i in batch
    }
    return len(keys)


def verify(identity_json: dict[str, Any]) -> bool:
    """Do the stored vectors match their seed and request? (Detects hand-edited identities.)"""
    again = expand(int(identity_json["seed"]), str(identity_json.get("requested", ""))).to_json()
    return bool(again["appearance"] == identity_json["appearance"]) and bool(
        again["voice"] == identity_json["voice"]
    )


def check_script(dictionary: dict[str, Any]) -> tuple[DistinctnessReport, list[str]]:
    """Distinctness of the characters that carry an `identity`, and the IDs whose stored vectors
    do not match their seed and request."""
    chars = dictionary.get("characters", {})
    cast = {
        cid: expand(int(c["identity"]["seed"]), str(c["identity"].get("requested", "")))
        for cid, c in chars.items()
        if "identity" in c
    }
    tampered = [cid for cid, c in chars.items() if "identity" in c and not verify(c["identity"])]
    return check_distinct(cast), tampered


def assign(dictionary: dict[str, Any], requests: dict[str, str], cast_seed: int) -> None:
    """Give every character in `requests` an identity from `sample_cast` (in place)."""
    for cid, ident in sample_cast(requests, cast_seed).items():
        dictionary["characters"][cid]["identity"] = ident.to_json()
        dictionary["characters"][cid]["identity_seed"] = ident.seed
