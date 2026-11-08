"""Procedural places from location descriptions (A-9).

A place is a parametric point cloud (floor, walls, architectural features) chosen and sized from
the location description words. Every point is `generated` provenance (the whole place is made by
the generator, not observed): nothing in a generated world is `observed`.

The same location id + seed always builds the same place, so a `.wreel` regenerates identically.
Spots are named positions characters can stand on (blocking consumes them).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from formats.wreel import PROVENANCE_CLASSES, PointCloud

GENERATED = PROVENANCE_CLASSES.index("generated")
SPACING = 0.06  # metres between surface points

Vec3 = tuple[float, float, float]
RGB = tuple[int, int, int]
F64 = npt.NDArray[np.float64]
Part = tuple[F64, F64, int, float]  # (positions, colours, provenance class, confidence)
Parts = list[Part]


@dataclass
class Place:
    cloud: PointCloud
    bounds: dict[str, list[float]]  # spec-style {"min": [x,y,z], "max": [x,y,z]}
    spots: dict[str, list[float]] = field(default_factory=dict)
    """Named positions (e.g. "centre", "arch") characters can stand on, in place coords."""


def _box_surface(rng: np.random.Generator, lo: Vec3, hi: Vec3, n: int) -> F64:
    """Points on the surface of an axis-aligned box (lo/hi may be given in either order)."""
    lo_a = np.minimum(np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64))
    hi_a = np.maximum(np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64))
    pts = rng.uniform(lo_a, hi_a, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    side = rng.integers(0, 2, size=n)
    pts[np.arange(n), axis] = np.where(side == 1, hi_a[axis], lo_a[axis])
    return pts


def _floor(rng: np.random.Generator, lo: Vec3, hi: Vec3, n: int, colour: RGB) -> tuple[F64, F64]:
    gx, gz = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[2], hi[2], n))
    pts = np.stack([gx.ravel(), np.full(gx.size, lo[1]), gz.ravel()], axis=1)
    jitter = rng.normal(0, 0.008, pts.shape) * np.array([1.0, 0.0, 1.0])
    return pts + jitter, np.tile(colour, (len(pts), 1))


def _wall(rng: np.random.Generator, lo: Vec3, hi: Vec3, n: int, colour: RGB) -> tuple[F64, F64]:
    pts = _box_surface(rng, lo, hi, n)
    return pts, np.tile(colour, (len(pts), 1))


def _add(
    parts: Parts,
    pts: F64,
    colour: F64 | RGB,
    cls: int = GENERATED,
    conf: float = 1.0,
) -> None:
    parts.append(
        (
            np.asarray(pts, dtype=np.float64),
            np.broadcast_to(np.array(colour, dtype=np.float64), (len(pts), 3)).copy(),
            cls,
            conf,
        )
    )


# --------------------------------------------------------------------------- templates


def _gate(rng: np.random.Generator, size: float) -> Place:
    """Stone arch with a path through it: 'stone arch, iron chain, monsoon path'."""
    w, h, d = size, size * 0.9, size * 0.5  # arch width / height / depth
    parts: Parts = []
    n = max(4, int(2 * size / SPACING))
    floor_pts, floor_col = _floor(rng, (-w, 0, -d), (w, 0, d), n, (110, 100, 90))
    _add(parts, floor_pts, floor_col)
    # two pillars + lintel
    th = 0.14 * size  # pillar thickness
    for sx in (-1.0, 1.0):
        x0 = sx * (w / 2 - th)
        _add(
            parts,
            *_wall(
                rng,
                (x0, 0, -th / 2),
                (x0 + sx * th, h, th / 2),
                int(2 * h / SPACING),
                (150, 140, 130),
            ),
        )
    _add(
        parts,
        *_wall(
            rng,
            (-w / 2 + th, h - th, -th / 2),
            (w / 2 - th, h, th / 2),
            int(2 * w / SPACING),
            (150, 140, 130),
        ),
    )
    bounds = {"min": [-w, 0, -d], "max": [w, h, d]}
    spots = {
        "centre": [0.0, 1.4, 0.0],
        "under_arch": [0.0, 1.4, 0.15],
        "path_far": [0.0, 1.4, -0.4 * d],
    }
    return Place(_build(parts), bounds, spots)


def _wall_walk(rng: np.random.Generator, size: float) -> Place:
    """Raised sea-wall walk with a parapet: 'sea wall walk, wet battlements'."""
    w, h, d = size * 1.6, size * 0.55, size * 0.9  # walkway width / height / depth
    parts: Parts = []
    floor_pts, floor_col = _floor(
        rng, (-w / 2, 0, -d / 2), (w / 2, 0, d / 2), int(2 * size / SPACING), (105, 98, 92)
    )
    _add(parts, floor_pts, floor_col)
    # parapet along both long sides
    th = 0.12 * size
    for sz in (-1.0, 1.0):
        z0 = sz * (d / 2 - th)
        _add(
            parts,
            *_wall(
                rng,
                (-w / 2, 0, z0),
                (w / 2, h, z0 + sz * th),
                int(4 * w / SPACING),
                (130, 122, 114),
            ),
        )
    bounds = {"min": [-w / 2, 0, -d / 2], "max": [w / 2, h, d / 2]}
    spots = {
        "centre": [0.0, 1.45, 0.0],
        "edge_near": [0.0, 1.45, 0.38 * d],
        "edge_far": [0.0, 1.45, -0.38 * d],
    }
    return Place(_build(parts), bounds, spots)


def _ruin(rng: np.random.Generator, size: float) -> Place:
    """Roofless nave with partial walls and a low arch: 'roofless nave, fig roots, low arch'."""
    w, h, d = size * 1.4, size * 0.75, size * 2.0
    parts: Parts = []
    floor_pts, floor_col = _floor(
        rng, (-w / 2, 0, -d / 2), (w / 2, 0, d / 2), int(2 * size / SPACING), (120, 112, 100)
    )
    _add(parts, floor_pts, floor_col)
    # end walls (partial height)
    th = 0.15 * size
    for sz in (-1.0, 1.0):
        z0 = sz * (d / 2 - th)
        _add(
            parts,
            *_wall(
                rng,
                (-w / 2, 0, z0),
                (w / 2, h * 0.55, z0 + sz * th),
                int(4 * w / SPACING),
                (140, 130, 120),
            ),
        )
    # side wall fragments
    for sx in (-1.0, 1.0):
        x0 = sx * (w / 2 - th)
        _add(
            parts,
            *_wall(
                rng,
                (x0, 0, -d / 3),
                (x0 + sx * th, h * 0.4, d / 3),
                int(2 * h / SPACING),
                (140, 130, 120),
            ),
        )
    # low arch in the near end wall
    arch_w, arch_h = w * 0.5, h * 0.45
    _add(
        parts,
        *_wall(
            rng,
            (-arch_w / 2, 0, d / 2 - th),
            (arch_w / 2, arch_h, d / 2 - th + 0.02),
            int(4 * arch_w / SPACING),
            (150, 140, 130),
        ),
    )
    bounds = {"min": [-w / 2, 0, -d / 2], "max": [w / 2, h, d / 2]}
    spots = {
        "centre": [0.0, 1.4, 0.0],
        "arch": [0.0, 1.4, 0.42 * d],
        "nave_far": [0.0, 1.4, -0.35 * d],
    }
    return Place(_build(parts), bounds, spots)


def _room(rng: np.random.Generator, size: float) -> Place:
    """One room, four walls, floor, ceiling: 'one whitewashed room'."""
    w, h, d = size, size * 0.85, size * 0.75
    parts: Parts = []
    floor_pts, floor_col = _floor(
        rng, (-w / 2, 0, -d / 2), (w / 2, 0, d / 2), int(2 * size / SPACING), (240, 238, 232)
    )
    _add(parts, floor_pts, floor_col)
    ceiling = np.stack(
        [
            np.tile(np.linspace(-w / 2, w / 2, int(w / SPACING)), int(d / SPACING)),
            np.full(int(w / SPACING) * int(d / SPACING), h),
            np.repeat(np.linspace(-d / 2, d / 2, int(d / SPACING)), int(w / SPACING)),
        ],
        axis=1,
    )
    _add(parts, ceiling, (245, 245, 240))
    th = 0.12 * size
    for sz in (-1.0, 1.0):
        z0 = sz * (d / 2 - th)
        _add(
            parts,
            *_wall(
                rng,
                (-w / 2, 0, z0),
                (w / 2, h, z0 + sz * th),
                int(4 * w / SPACING),
                (242, 242, 236),
            ),
        )
    for sx in (-1.0, 1.0):
        x0 = sx * (w / 2 - th)
        _add(
            parts,
            *_wall(
                rng,
                (x0, 0, -d / 2),
                (x0 + sx * th, h, d / 2),
                int(4 * d / SPACING),
                (242, 242, 236),
            ),
        )
    bounds = {"min": [-w / 2, 0, -d / 2], "max": [w / 2, h, d / 2]}
    spots = {
        "centre": [0.0, 1.4, 0.0],
        "window": [w * 0.32, 1.4, d / 2],
        "door": [0.0, 1.4, -d / 2],
    }
    return Place(_build(parts), bounds, spots)


def _tunnel(rng: np.random.Generator, size: float) -> Place:
    """Arched corridor stepping into a hill: 'steps into the hill, iron gate, channel'."""
    w, h, d = size * 0.62, size * 0.85, size * 2.4
    parts: Parts = []
    floor_pts, floor_col = _floor(
        rng, (-w / 2, 0, -d / 2), (w / 2, 0, d / 2), int(2 * size / SPACING), (80, 74, 70)
    )
    _add(parts, floor_pts, floor_col)
    # steps along the centre line at the near end
    for i in range(4):
        y = i * 0.12
        z0 = -d / 2 + i * 0.18
        step = _floor(
            rng,
            (-w / 3, y, z0),
            (w / 3, y + 0.001, z0 + 0.18),
            max(4, int(w / SPACING)),
            (90, 84, 80),
        )
        _add(parts, step[0], step[1])
    # arched ceiling ribs (semicircle cross section) every metre
    rib_n = max(2, int(d / 1.0))
    for k in range(rib_n):
        z = -d / 2 + (k + 0.5) * (d / rib_n)
        th = 0.1 * size
        ang = np.linspace(0, math.pi, max(8, int(math.pi * w / SPACING)))
        x = (w / 2 - th) * np.cos(ang)
        y = h * 0.5 + (h * 0.5) * np.sin(ang)  # type: ignore[assignment]
        rib = np.stack([x, y, np.full_like(x, z)], axis=1)
        _add(parts, rib, (95, 88, 84))
    bounds = {"min": [-w / 2, 0, -d / 2], "max": [w / 2, h, d / 2]}
    spots = {
        "centre": [0.0, 1.4, 0.0],
        "entrance": [0.0, 1.4, -d / 2],
        "deep": [0.0, 1.4, 0.35 * d],
    }
    return Place(_build(parts), bounds, spots)


def _build(parts: Parts) -> PointCloud:
    positions = np.concatenate([p for p, _, _, _ in parts]).astype(np.float32)
    colours = np.concatenate([c for _, c, _, _ in parts]).astype(np.uint8)
    n = len(positions)
    return PointCloud(positions, colours, np.full(n, GENERATED, np.uint8), np.ones(n, np.float32))


TemplateFn = Callable[[np.random.Generator, float], Place]


# description word -> (template name, size hint)
_TEMPLATE_KEYS: tuple[tuple[tuple[str, ...], TemplateFn], ...] = (
    (("gate", "arch", "chain", "monsoon", "path"), _gate),
    (("rampart", "battlement", "walk", "sea wall"), _wall_walk),
    (("ruin", "chapel", "nave", "fig", "arch"), _ruin),
    (("room", "hut", "whitewashed", "lamp"), _room),
    (("passage", "tunnel", "hill", "gate", "channel"), _tunnel),
)


def _pick_template(description: str, name: str) -> TemplateFn:
    text = f"{name} {description}".lower()
    for keys, tmpl in _TEMPLATE_KEYS:
        if any(k in text for k in keys):
            return tmpl
    return _room  # default: a room


def _pick(v: float, words: tuple[str, ...]) -> str:
    return words[min(int(v * len(words)), len(words) - 1)]


def _size_for(description: str, name: str) -> float:
    """A scale from the description words; larger places get more points."""
    text = f"{name} {description}".lower()
    big = any(
        k in text
        for k in ("rampart", "walk", "nave", "ruin", "passage", "tunnel", "battlement", "wall")
    )
    return 6.0 if big else 4.0


def build_place(entry: dict[str, Any], seed: int = 0) -> Place:
    """Build one location procedurally from its dictionary entry.

    `entry` has `name` and `description`. The place is deterministic given the location id is
    folded into `seed` by the caller.
    """
    rng = np.random.default_rng(seed)
    tmpl = _pick_template(str(entry.get("description", "")), str(entry.get("name", "")))
    size = _size_for(str(entry.get("description", "")), str(entry.get("name", "")))
    return tmpl(rng, size)


def place_world_offset(place: Place, origin: Vec3) -> F64:
    """Translate a place's points so its bounds' min corner sits at `origin` (x, z)."""
    lo = np.array(place.bounds["min"], dtype=np.float64)
    shift = np.array([origin[0] - lo[0], 0.0, origin[1] - lo[2]], dtype=np.float64)
    return shift
