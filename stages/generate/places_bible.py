"""Environment bible: seeded, distinct place identity for generated worlds.

Same design principle as character bible (A-8): every location has an explicit
identity code (seed + style request) that expands to appearance dimensions.
Nothing about a place is left to chance — same seed + same request = same place,
always. Distinctness check ensures no two locations in a film are too similar.

This solves the medieval-movie problem: a castle gate, forest path, and chapel
ruin each keep their appearance across every shot, and no two locations look
alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

APPEARANCE_DIMS = (
    # architecture
    "arch_style",  # 0=rounded, 1=pointed, 2=flat, 3=mixed
    "wall_material",  # 0=stone, 1=wood, 2=plaster, 3=brick, 4=mud
    "roof_type",  # 0=none, 1=thatched, 2=tile, 3=slate, 4=shingle
    "door_style",  # 0=plank, 1=iron_bound, 2=arched, 3=carved, 4=hidden
    "window_style",  # 0=arrow_slit, 1=round, 2=rectangular, 3=stained, 4=none
    # lighting
    "light_source",  # 0=torch, 1=lantern, 2=fireplace, 3=window, 4=magic_glow
    "light_level",  # 0=dark, 1=dim, 2=normal, 3=bright, 4=blazing
    # atmosphere
    "weather",  # 0=clear, 1=rain, 2=snow, 3=fog, 4=dust
    "season",  # 0=spring, 1=summer, 2=autumn, 3=winter
    "time_of_day",  # 0=dawn, 1=morning, 2=noon, 3=afternoon, 4=dusk, 5=night
    # size/scale
    "scale",  # 0=tiny, 1=small, 2=medium, 3=large, 4=vast
    "openness",  # 0=enclosed, 1=semi, 2=open, 3=roofless, 4=outdoors
)
APPEARANCE_MIN = 0.20  # minimum RMS difference between any two places

CANDIDATES = 32  # best-candidate sampling


@dataclass(frozen=True)
class PlaceIdentity:
    seed: int
    requested: str
    appearance: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "requested": self.requested,
            "appearance": {k: round(v, 4) for k, v in self.appearance.items()},
        }


def expand(seed: int, requested: str = "") -> PlaceIdentity:
    """Seed + style request -> place identity.

    Place descriptions are more permissive than character style requests: any description
    words are accepted, but likeness requests (names of real places like "Vasai Fort",
    "Eiffel Tower") are refused.
    """
    likeness_words = {
        "fort",
        "tower",
        "bridge",
        "monument",
        "memorial",
        "landmark",
        "statue",
        "temple",
        "church",
        "mosque",
        "cathedral",
        "palace",
        "museum",
        "library",
        "theater",
        "theatre",
        "stadium",
        "arena",
    }

    ranges: dict[str, tuple[float, float]] = {}
    if requested.strip():
        words = requested.lower().split()
        # Check for likeness requests (names of real places)
        for w in words:
            if w in likeness_words:
                raise ValueError(
                    f"place request refused: '{w}' is a place type, not a description. "
                    f"Describe the place instead of naming it."
                )
        # Parse simple dimension keywords
        dim_map = {
            "gate": ("arch_style", 0.0),
            "arch": ("arch_style", 0.0),
            "pointed": ("arch_style", 0.33),
            "flat": ("arch_style", 0.67),
            "mixed": ("arch_style", 1.0),
            "stone": ("wall_material", 0.0),
            "wood": ("wall_material", 0.25),
            "plaster": ("wall_material", 0.5),
            "brick": ("wall_material", 0.75),
            "mud": ("wall_material", 1.0),
            "thatched": ("roof_type", 0.25),
            "tile": ("roof_type", 0.5),
            "slate": ("roof_type", 0.75),
            "shingle": ("roof_type", 1.0),
            "torch": ("light_source", 0.0),
            "lantern": ("light_source", 0.25),
            "fireplace": ("light_source", 0.5),
            "window": ("light_source", 0.75),
            "magic": ("light_source", 1.0),
            "dark": ("light_level", 0.0),
            "dim": ("light_level", 0.25),
            "normal": ("light_level", 0.5),
            "bright": ("light_level", 0.75),
            "blazing": ("light_level", 1.0),
            "clear": ("weather", 0.0),
            "rain": ("weather", 0.25),
            "snow": ("weather", 0.5),
            "fog": ("weather", 0.75),
            "dust": ("weather", 1.0),
            "tiny": ("scale", 0.0),
            "small": ("scale", 0.25),
            "medium": ("scale", 0.5),
            "large": ("scale", 0.75),
            "vast": ("scale", 1.0),
        }
        for w in words:
            if w in dim_map:
                dim, val = dim_map[w]
                ranges[dim] = (val, val)

    u = np.random.default_rng(seed).random(len(APPEARANCE_DIMS))

    def value(dim: str, x: float) -> float:
        lo, hi = ranges.get(dim, (0.0, 1.0))
        return float(lo + x * (hi - lo))

    vals = {d: value(d, float(x)) for d, x in zip(APPEARANCE_DIMS, u, strict=True)}
    return PlaceIdentity(seed, requested, {d: vals[d] for d in APPEARANCE_DIMS})


def _vec(values: dict[str, float], dims: tuple[str, ...]) -> np.ndarray:
    return np.array([values[d] for d in dims])


def distance(a: PlaceIdentity, b: PlaceIdentity) -> float:
    """RMS difference over appearance dimensions, in [0, 1]."""
    da = _vec(a.appearance, APPEARANCE_DIMS) - _vec(b.appearance, APPEARANCE_DIMS)
    return float(np.sqrt(np.mean(da**2)))


@dataclass
class PlaceDistinctnessReport:
    ok: bool
    min_distance: float
    too_close: list[tuple[str, str, float]] = field(default_factory=list)


def check_distinct(places: dict[str, PlaceIdentity]) -> PlaceDistinctnessReport:
    """Check that every pair of places differs by at least APPEARANCE_MIN."""
    min_d = 1.0
    close: list[tuple[str, str, float]] = []
    for (ia, a), (ib, b) in combinations(sorted(places.items()), 2):
        d = distance(a, b)
        min_d = min(min_d, d)
        if d < APPEARANCE_MIN:
            close.append((ia, ib, round(d, 3)))
    return PlaceDistinctnessReport(not close, round(min_d, 4), close)


def sample_cast(
    requests: dict[str, str], cast_seed: int, candidates: int = CANDIDATES
) -> dict[str, PlaceIdentity]:
    """Pick a seed per place, in key order. Each place takes the best of `candidates`
    seeds: the one whose closest already-chosen place is farthest away."""
    rng = np.random.default_rng(cast_seed)
    places: dict[str, PlaceIdentity] = {}
    for pid in sorted(requests):
        best: PlaceIdentity | None = None
        best_score = -1.0
        for seed in rng.integers(0, 2**31 - 1, size=candidates):
            cand = expand(int(seed), requests[pid])
            if not places:
                best = cand
                break
            score = min(distance(cand, other) for other in places.values())
            if score > best_score:
                best, best_score = cand, score
        assert best is not None
        places[pid] = best
    return places


def verify(identity_json: dict[str, Any]) -> bool:
    """Do the stored vectors match their seed and request? (Detects hand-edited identities.)"""
    again = expand(int(identity_json["seed"]), str(identity_json.get("requested", "")))
    return bool(again.appearance == identity_json["appearance"])


def check_bible(dictionary: dict[str, Any]) -> tuple[PlaceDistinctnessReport, list[str]]:
    """Distinctness of places in the bible, plus tamper detection."""
    places = dictionary.get("places", {})
    cast = {
        pid: expand(int(p["identity"]["seed"]), str(p["identity"].get("requested", "")))
        for pid, p in places.items()
        if "identity" in p
    }
    tampered = [pid for pid, p in places.items() if "identity" in p and not verify(p["identity"])]
    return check_distinct(cast), tampered


def assign(dictionary: dict[str, Any], requests: dict[str, str], cast_seed: int) -> None:
    """Give every place in `requests` an identity from `sample_cast` (in place)."""
    for pid, ident in sample_cast(requests, cast_seed).items():
        dictionary["places"][pid]["identity"] = ident.to_json()
        dictionary["places"][pid]["identity_seed"] = ident.seed
