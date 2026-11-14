"""Entity bible: seeded, distinct non-human identity codes (A-8 extension).

Same design principle as the character bible: every entity has a code made from a `seed` and an
optional style request. The code expands deterministically to an appearance vector; requested
styles narrow dimensions to ranges. The expanded vectors are stored in the `.wscript` so a reader
can check them against the seed.

Entity types (A-9 cosmic extension):
  human     - flesh-and-blood person (existing bible)
  alien     - non-human biology, exoskeleton or internal skeleton
  robot     - geometric construction, panels and joints
  energy    - plasma / luminous being, no solid surface
  gas       - volumetric cloud, density falloff, no surface
  crystal   - faceted mineral growth, refractive
  giant     - scaled humanoid, 3-100x human height
  tiny      - scaled humanoid, 0.01-0.1x human height

Each type carries its own appearance dimensions; dimensions not relevant to a type are ignored by
its renderer but still stored (so the distinctness check can compare across types on shared dims).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- shared dims
# (all types)

SHARED = (
    "material",  # 0=flesh, 1=metal, 2=crystal, 3=plasma, 4=gas, 5=stone, 6=energy
    "scale",  # 0=tiny(0.05x), 1=small(0.3x), 2=human(1x), 3=large(3x), 4=giant(10x), 5=vast(100x)
    "luminosity",  # 0=none, 1=dim, 2=normal, 3=bright, 4=blazing
    "color_temp",  # 0=cold_blue, 1=cool_white, 2=neutral, 3=warm_orange, 4=hot_red
    "texture",  # 0=smooth, 1=rough, 2=segmented, 3=faceted, 4=furry, 5=glowing
)

# --------------------------------------------------------------------------- type-specific dims

ALIEN = (
    "limb_count",  # 2-8 limbs
    "eye_count",  # 0-8 eyes
    "body_symmetry",  # 0=bilateral, 1=radial, 2=asymmetric
    "skin_hardness",  # 0=soft, 1=leathery, 2=chitinous, 3=armored
)

ROBOT = (
    "panel_count",  # 0=none, 1=few, 2=many, 3=segmented_plates
    "joint_type",  # 0=ball, 1=hinge, 2=sliding, 3=magnetic
    "antenna_count",  # 0-4 antennae/sensors
    "screen_face",  # 0=no face, 1=display, 2=projector, 3=hologram
)

ENERGY = (
    "pulse_rate",  # 0=static, 1=slow, 2=normal, 3=fast, 4=frenetic
    "core_size",  # 0=point, 1=small, 2=normal, 3=large, 4=diffuse
    "tendril_count",  # 0-8 energy tendrils
    "aura_radius",  # 0=none, 1=body, 2=close, 3=room, 4=vast
)

GAS = (
    "density",  # 0=wisp, 1=thin, 2=normal, 3=thick, 4=opaque
    "flow_speed",  # 0=still, 1=drift, 2=flow, 3=rush, 4=turbulent
    "color_layers",  # 1-4 overlapping color layers
    "condensation",  # 0=none, 1=frost, 2=dew, 3=rain, 4=storm
)

CRYSTAL = (
    "facet_count",  # 0=smooth, 1=few, 2=many, 3=intricate, 4=fractal
    "refraction",  # 0=none, 1=low, 2=medium, 3=high, 4=prismatic
    "growth_axis",  # 0=radial, 1=vertical, 2=horizontal, 3=spiral, 4=random
    "inclusion",  # 0=pure, 1=bubbles, 2=veins, 3=impurities, 4=foreign_matter
)

GIANT = (
    "stride",  # 0=slow, 1=measured, 2=normal, 3=long, 4=earthquake
    "reach",  # 0=short, 1=normal, 2=long, 3=very_long, 4=continental
    "footfall",  # 0=silent, 1=thud, 2=boom, 3=crack, 4=shatter
    "shadow",  # 0=none, 1=patch, 2=region, 3=valley, 4=continent
)

TINY = (
    "hop_height",  # 0=none, 1=cm, 2=dm, 3=m, 4=several_m
    "carry_load",  # 0=none, 1=grain, 2=leaf, 3=stone, 4=tool
    "echo",  # 0=none, 1=faint, 2=clear, 3=loud, 4=deafening
    "hide_size",  # 0=none, 1=leaf, 2=stone, 3=log, 4=building
)

ALL_DIMS = SHARED + ALIEN + ROBOT + ENERGY + GAS + CRYSTAL + GIANT + TINY

ENTITY_TYPES = ("human", "alien", "robot", "energy", "gas", "crystal", "giant", "tiny")

# dims per entity type (shared dims always present)
TYPE_DIMS: dict[str, tuple[str, ...]] = {
    "human": SHARED,
    "alien": SHARED + ALIEN,
    "robot": SHARED + ROBOT,
    "energy": SHARED + ENERGY,
    "gas": SHARED + GAS,
    "crystal": SHARED + CRYSTAL,
    "giant": SHARED + GIANT,
    "tiny": SHARED + TINY,
}

APPEARANCE_MIN = 0.20  # minimum RMS difference between any two entities
CANDIDATES = 32  # best-candidate sampling


@dataclass(frozen=True)
class EntityIdentity:
    seed: int
    requested: str
    entity_type: str
    appearance: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "requested": self.requested,
            "entity_type": self.entity_type,
            "appearance": {k: round(v, 4) for k, v in self.appearance.items()},
        }


def expand(seed: int, requested: str = "", entity_type: str = "human") -> EntityIdentity:
    """Seed + style request + entity type -> identity.

    Raises ValueError if entity_type is unknown or if the request mentions a real-world
    entity name (likeness protection, same rule as character bible).
    """
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"unknown entity type: {entity_type!r}")

    likeness_words = {
        "god",
        "gods",
        "deity",
        "angel",
        "demon",
        "devil",
        "satan",
        "zeus",
        "odin",
        "thor",
        "hercules",
        "superman",
        "batman",
        "spiderman",
        "ironman",
        "gandalf",
        "dumbledore",
        "vader",
        "yoda",
        "kirk",
        "picard",
        "doctor",
        "dalek",
        "cyberman",
        "transformer",
        "optimus",
    }

    ranges: dict[str, tuple[float, float]] = {}
    if requested.strip():
        words = requested.lower().split()
        for w in words:
            if w in likeness_words:
                raise ValueError(
                    f"entity request refused: '{w}' is a real-world entity name. "
                    f"Describe the entity instead of naming it."
                )

    u = np.random.default_rng(seed).random(len(TYPE_DIMS[entity_type]))

    def value(dim: str, x: float) -> float:
        lo, hi = ranges.get(dim, (0.0, 1.0))
        return float(lo + x * (hi - lo))

    vals = {d: value(d, float(x)) for d, x in zip(TYPE_DIMS[entity_type], u, strict=True)}
    return EntityIdentity(
        seed, requested, entity_type, {d: vals[d] for d in TYPE_DIMS[entity_type]}
    )


def _vec(values: dict[str, float], dims: tuple[str, ...]) -> np.ndarray:
    return np.array([values[d] for d in dims])


def distance(a: EntityIdentity, b: EntityIdentity) -> float:
    """RMS difference over shared dims between two entities, in [0, 1].

    Only dims present in BOTH types are compared, so an alien and a robot are compared
    on material/scale/luminosity/etc. (the shared dims), not on alien-specific dims.
    """
    shared = [d for d in TYPE_DIMS[a.entity_type] if d in TYPE_DIMS[b.entity_type]]
    if not shared:
        return 1.0  # no common ground: maximally different
    da = _vec(a.appearance, tuple(shared)) - _vec(b.appearance, tuple(shared))
    return float(np.sqrt(np.mean(da**2)))


@dataclass
class EntityDistinctnessReport:
    ok: bool
    min_distance: float
    too_close: list[tuple[str, str, float]] = field(default_factory=list)


def check_distinct(entities: dict[str, EntityIdentity]) -> EntityDistinctnessReport:
    """Check that every pair of entities differs by at least APPEARANCE_MIN on shared dims."""
    min_d = 1.0
    close: list[tuple[str, str, float]] = []
    for (ia, a), (ib, b) in combinations(sorted(entities.items()), 2):
        d = distance(a, b)
        min_d = min(min_d, d)
        if d < APPEARANCE_MIN:
            close.append((ia, ib, round(d, 3)))
    return EntityDistinctnessReport(not close, round(min_d, 4), close)


def sample_cast(
    requests: dict[str, str], cast_seed: int, candidates: int = CANDIDATES
) -> dict[str, EntityIdentity]:
    """Pick a seed per entity, in key order. Each entity takes the best of `candidates`
    seeds: the one whose closest already-chosen entity is farthest away (on shared dims).
    Deterministic per cast seed."""
    rng = np.random.default_rng(cast_seed)
    cast: dict[str, EntityIdentity] = {}
    for eid in sorted(requests):
        # entity_type comes from the request prefix, e.g. "alien:glowing insectoid"
        req = requests[eid]
        etype = "human"
        if ":" in req:
            prefix, rest = req.split(":", 1)
            if prefix in ENTITY_TYPES:
                etype = prefix
                req = rest
        best: EntityIdentity | None = None
        best_score = -1.0
        for seed in rng.integers(0, 2**31 - 1, size=candidates):
            cand = expand(int(seed), req, etype)
            if not cast:
                best = cand
                break
            score = min(distance(cand, other) for other in cast.values())
            if score > best_score:
                best, best_score = cand, score
        assert best is not None
        cast[eid] = best
    return cast


def verify(identity_json: dict[str, Any]) -> bool:
    """Do the stored vectors match their seed, request and entity type?"""
    again = expand(
        int(identity_json["seed"]),
        str(identity_json.get("requested", "")),
        str(identity_json.get("entity_type", "human")),
    )
    again_appearance = {k: round(v, 4) for k, v in again.appearance.items()}
    return bool(again_appearance == identity_json["appearance"]) and bool(
        again.entity_type == identity_json["entity_type"]
    )


def check_bible(dictionary: dict[str, Any]) -> tuple[EntityDistinctnessReport, list[str]]:
    """Distinctness of entities in the bible, plus tamper detection."""
    entities = dictionary.get("entities", {})
    cast = {
        eid: expand(
            int(e["identity"]["seed"]),
            str(e["identity"].get("requested", "")),
            str(e["identity"].get("entity_type", "human")),
        )
        for eid, e in entities.items()
        if "identity" in e
    }
    tampered = [eid for eid, e in entities.items() if "identity" in e and not verify(e["identity"])]
    return check_distinct(cast), tampered


def assign(dictionary: dict[str, Any], requests: dict[str, str], cast_seed: int) -> None:
    """Give every entity in `requests` an identity from `sample_cast` (in place)."""
    for eid, ident in sample_cast(requests, cast_seed).items():
        dictionary["entities"][eid]["identity"] = ident.to_json()
        dictionary["entities"][eid]["identity_seed"] = ident.seed
