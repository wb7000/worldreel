"""Cosmic place bible: seeded, distinct place identity for non-earth settings (A-9 extension).

Same design principle as the place bible: every location has an explicit identity code
(seed + style request) that expands to appearance dimensions. Nothing about a place is
left to chance — same seed + same request = same place, always. Distinctness check ensures
no two locations in a film are too similar.

Cosmic place types extend earth places with non-earth structures, materials, gravities,
atmospheres and light sources.

Place types:
  planetary_surface - ground you can walk on (planet, moon, asteroid)
  orbital           - station, habitat, ship in orbit
  nebula_interior   - inside a gas cloud or stellar nursery
  dimensional_rift  - folded or pocket space
  megastructure     - ringworld, dyson shell, matrioshka brain
  void              - empty space, no ground, no sky
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

SHARED = (
    "structure",  # 0=ground, 1=floating, 2=orbital, 3=rift, 4=megastructure, 5=void
    "material",  # 0=stone, 1=metal, 2=crystal, 3=bio_organic, 4=energy, 5=void, 6=ice
    "light_source",  # 0=none, 1=star, 2=nebula_glow, 3=biolum, 4=artificial, 5=plasma, 6=magic
    "light_level",  # 0=dark, 1=dim, 2=normal, 3=bright, 4=blazing
    "gravity",  # 0=0.1g, 1=0.5g, 2=1g, 3=2g, 3=3g, 4=zero, 5=inverted
    "atmosphere",  # 0=none, 1=nitrogen, 2=methane, 3=ammonia, 4=plasma, 5=toxic
    "temperature",  # 0=cryogenic, 1=freezing, 2=temperate, 3=molten, 4=stellar
    "scale",  # 0=tiny, 1=small, 2=medium, 3=large, 4=vast, 5=cosmic
    "openness",  # 0=enclosed, 1=semi, 2=open, 3=roofless, 4=outdoors, 5=boundless
)

PLANETARY = (
    "terrain",  # 0=rock, 1=sand, 2=ice, 3=liquid, 4=gas, 5=fungal, 6=metallic
    "sea_level",  # 0=dry, 1=shallow, 2=ocean, 3=global_ocean, 4=supercritical
    "mountain",  # 0=flat, 1=hills, 2=peaks, 3=range, 4=monolithic
    "caves",  # 0=none, 1=few, 2=many, 3=vast_network, 4=hollow
)

ORBITAL = (
    "ring_count",  # 0-4 concentric rings
    "spoke_count",  # 0-8 radial spokes
    "dock_count",  # 0-16 docking bays
    "habitat_rings",  # 0-4 rotating habitat rings
)

NEBULA = (
    "density",  # 0=wisp, 1=thin, 2=normal, 3=thick, 4=opaque
    "color_layers",  # 1-4 overlapping emission layers
    "star_count",  # 0-6 embedded stars
    "pillar_count",  # 0-8 elephant-trunk pillars
)

RIFT = (
    "fold_count",  # 0-4 dimensional folds
    "anchor_count",  # 0-4 reality anchors
    "echo_depth",  # 0=none, 1=faint, 2=clear, 3=strong, 4=overwhelming
    "time_rate",  # 0=stopped, 1=slow, 2=normal, 3=fast, 4=erratic
)

MEGASTRUCTURE = (
    "ring_diameter",  # 0=km, 1=100km, 2=1000km, 3=10000km, 4=stellar
    "wall_height",  # 0=m, 1=km, 2=100km, 3=1000km, 4=atmosphere
    "spoke_width",  # 0=m, 1=km, 2=10km, 3=100km, 4=continent
    "floor_layers",  # 0=single, 2=double, 4=quad, 8=octuple
)

VOID = (
    "drift_speed",  # 0=still, 1=drift, 2=flow, 3=rush, 4=turbulent
    "particle_density",  # 0=none, 1=dust, 2=ice, 3=debris, 4=storm
    "horizon",  # 0=none, 1=near, 2=mid, 3=far, 4=infinite
    "silence",  # 0=noisy, 1=quiet, 2=silent, 3=absolute
)

ALL_DIMS = SHARED + PLANETARY + ORBITAL + NEBULA + RIFT + MEGASTRUCTURE + VOID

PLACE_TYPES = (
    "planetary_surface",
    "orbital",
    "nebula_interior",
    "dimensional_rift",
    "megastructure",
    "void",
)

TYPE_DIMS: dict[str, tuple[str, ...]] = {
    "planetary_surface": SHARED + PLANETARY,
    "orbital": SHARED + ORBITAL,
    "nebula_interior": SHARED + NEBULA,
    "dimensional_rift": SHARED + RIFT,
    "megastructure": SHARED + MEGASTRUCTURE,
    "void": SHARED + VOID,
}

APPEARANCE_MIN = 0.20
CANDIDATES = 32


@dataclass(frozen=True)
class PlaceIdentity:
    seed: int
    requested: str
    place_type: str
    appearance: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "requested": self.requested,
            "place_type": self.place_type,
            "appearance": {k: round(v, 4) for k, v in self.appearance.items()},
        }


def expand(seed: int, requested: str = "", place_type: str = "planetary_surface") -> PlaceIdentity:
    """Seed + style request + place type -> place identity.

    Raises ValueError if place_type is unknown or if the request mentions a real-world
    place name (likeness protection, same rule as character bible).
    """
    if place_type not in PLACE_TYPES:
        raise ValueError(f"unknown place type: {place_type!r}")

    likeness_words = {
        "earth",
        "mars",
        "venus",
        "jupiter",
        "moon",
        "sun",
        "milky_way",
        "andromeda",
        "iss",
        "hubble",
        "voyager",
        "apollo",
        "sputnik",
        "death_star",
        "enterprise",
        "tardis",
        "midgard",
        "asgard",
        "olympus",
        "valhalla",
    }

    ranges: dict[str, tuple[float, float]] = {}
    if requested.strip():
        words = requested.lower().split()
        for w in words:
            if w in likeness_words:
                raise ValueError(
                    f"place request refused: '{w}' is a real-world place name. "
                    f"Describe the place instead of naming it."
                )

    u = np.random.default_rng(seed).random(len(TYPE_DIMS[place_type]))

    def value(dim: str, x: float) -> float:
        lo, hi = ranges.get(dim, (0.0, 1.0))
        return float(lo + x * (hi - lo))

    vals = {d: value(d, float(x)) for d, x in zip(TYPE_DIMS[place_type], u, strict=True)}
    return PlaceIdentity(seed, requested, place_type, {d: vals[d] for d in TYPE_DIMS[place_type]})


def _vec(values: dict[str, float], dims: tuple[str, ...]) -> np.ndarray:
    return np.array([values[d] for d in dims])


def distance(a: PlaceIdentity, b: PlaceIdentity) -> float:
    """RMS difference over shared dims between two places, in [0, 1].

    Only dims present in BOTH types are compared.
    """
    shared = [d for d in TYPE_DIMS[a.place_type] if d in TYPE_DIMS[b.place_type]]
    if not shared:
        return 1.0
    da = _vec(a.appearance, tuple(shared)) - _vec(b.appearance, tuple(shared))
    return float(np.sqrt(np.mean(da**2)))


@dataclass
class PlaceDistinctnessReport:
    ok: bool
    min_distance: float
    too_close: list[tuple[str, str, float]] = field(default_factory=list)


def check_distinct(places: dict[str, PlaceIdentity]) -> PlaceDistinctnessReport:
    """Check that every pair of places differs by at least APPEARANCE_MIN on shared dims."""
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
    seeds: the one whose closest already-chosen place is farthest away (on shared dims).
    Deterministic per cast seed."""
    rng = np.random.default_rng(cast_seed)
    places: dict[str, PlaceIdentity] = {}
    for pid in sorted(requests):
        # place_type comes from the request prefix, e.g. "orbital:glowing ring habitat"
        req = requests[pid]
        ptype = "planetary_surface"
        if ":" in req:
            prefix, rest = req.split(":", 1)
            if prefix in PLACE_TYPES:
                ptype = prefix
                req = rest
        best: PlaceIdentity | None = None
        best_score = -1.0
        for seed in rng.integers(0, 2**31 - 1, size=candidates):
            cand = expand(int(seed), req, ptype)
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
    """Do the stored vectors match their seed, request and place type?"""
    again = expand(
        int(identity_json["seed"]),
        str(identity_json.get("requested", "")),
        str(identity_json.get("place_type", "planetary_surface")),
    )
    again_appearance = {k: round(v, 4) for k, v in again.appearance.items()}
    return bool(again_appearance == identity_json["appearance"]) and bool(
        again.place_type == identity_json["place_type"]
    )


def check_bible(dictionary: dict[str, Any]) -> tuple[PlaceDistinctnessReport, list[str]]:
    """Distinctness of places in the bible, plus tamper detection."""
    places = dictionary.get("places", {})
    cast = {
        pid: expand(
            int(p["identity"]["seed"]),
            str(p["identity"].get("requested", "")),
            str(p["identity"].get("place_type", "planetary_surface")),
        )
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
