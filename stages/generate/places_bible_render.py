"""Render place identities to 3D geometry.

Takes a PlaceIdentity (from places_bible.py) and builds a deterministic point-cloud
place. Same identity + same seed = same place, every time. This is what makes a
medieval movie's castle gate, forest path, and chapel ruin look identical across
every shot.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from formats.wreel import PROVENANCE_CLASSES, PointCloud
from stages.generate.places_bible import PlaceIdentity

GENERATED = PROVENANCE_CLASSES.index("generated")
SPACING = 0.06  # metres between surface points


def _fib_sphere(n: int) -> npt.NDArray[np.float64]:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)


def _ellipsoid(
    center: npt.NDArray[np.float64], radii: npt.NDArray[np.float64], n: int
) -> npt.NDArray[np.float64]:
    area = (
        4
        * math.pi
        * (
            (radii[0] * radii[1]) ** 1.6 / 3
            + (radii[0] * radii[2]) ** 1.6 / 3
            + (radii[1] * radii[2]) ** 1.6 / 3
        )
        ** (1 / 1.6)
    )
    return _fib_sphere(max(50, int(area / SPACING**2))) * radii + center


def _cylinder(
    center: npt.NDArray[np.float64],
    radius: float,
    height: float,
    n_around: int,
    n_up: int,
    axis: str = "y",
) -> npt.NDArray[np.float64]:
    circ = 2 * math.pi * radius
    nu = max(8, int(circ / SPACING))
    nv = max(2, int(height / SPACING))
    u, v = np.meshgrid(np.linspace(0, 2 * math.pi, nu, endpoint=False), np.linspace(0, height, nv))
    if axis == "y":
        return np.stack(
            [
                center[0] + radius * np.cos(u).ravel(),
                center[1] + v.ravel(),
                center[2] + radius * np.sin(u).ravel(),
            ],
            1,
        )
    elif axis == "z":
        return np.stack(
            [
                center[0] + radius * np.cos(u).ravel(),
                center[1] + radius * np.sin(u).ravel(),
                center[2] + v.ravel(),
            ],
            1,
        )
    else:  # x
        return np.stack(
            [
                center[1] + v.ravel(),
                center[2] + radius * np.cos(u).ravel(),
                center[3] + radius * np.sin(u).ravel(),
            ],
            1,
        )


def _box(
    center: npt.NDArray[np.float64], sx: float, sy: float, sz: float, n: int
) -> npt.NDArray[np.float64]:
    """Points on surface of box centered at center with half-extents sx, sy, sz."""
    lo = center - np.array([sx, sy, sz])
    hi = center + np.array([sx, sy, sz])
    pts = np.random.default_rng(42).uniform(lo, hi, size=(n, 3))
    axis = np.random.default_rng(43).integers(0, 3, size=n)
    side = np.random.default_rng(44).integers(0, 2, size=n)
    pts[np.arange(n), axis] = np.where(side == 1, hi[axis], lo[axis])
    return pts


_Part = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int, float]


def _add(
    parts: list[_Part],
    pts: npt.NDArray[np.float64],
    color: npt.NDArray[np.float64],
    cls: int = GENERATED,
    conf: float = 1.0,
) -> None:
    parts.append(
        (
            pts.astype(np.float64),
            np.broadcast_to(np.array(color, dtype=np.float64), (len(pts), 3)).copy(),
            cls,
            conf,
        )
    )


def _wall_material_color(v: float) -> npt.NDArray[np.float64]:
    """0=stone grey, 1=wood brown, 2=plaster white, 3=brick red, 4=mud brown."""
    colors = [
        np.array([110, 100, 90], np.float64),  # stone
        np.array([139, 90, 43], np.float64),  # wood
        np.array([220, 220, 210], np.float64),  # plaster
        np.array([178, 34, 34], np.float64),  # brick
        np.array([101, 67, 33], np.float64),  # mud
    ]
    idx = int(v * len(colors))
    idx = min(idx, len(colors) - 1)
    return colors[idx]


def _light_source_color(v: float) -> npt.NDArray[np.float64]:
    """0=torch orange, 1=lantern yellow, 2=fireplace red, 3=window white, 4=magic purple."""
    colors = [
        np.array([255, 140, 0], np.float64),  # torch
        np.array([255, 220, 100], np.float64),  # lantern
        np.array([200, 50, 50], np.float64),  # fireplace
        np.array([240, 240, 240], np.float64),  # window
        np.array([180, 100, 255], np.float64),  # magic
    ]
    idx = int(v * len(colors))
    idx = min(idx, len(colors) - 1)
    return colors[idx]


def render_place(ident: PlaceIdentity, seed: int = 0, size: float = 6.0) -> PointCloud:
    """Render one place identity to a deterministic point cloud.

    Same identity + same seed = same place, always. Uses the appearance
    dimensions to choose architecture, materials, lighting, atmosphere.
    """
    rng = np.random.default_rng(seed)
    a = ident.appearance
    parts: list[_Part] = []

    # Dimensions
    wall_mat = _wall_material_color(a["wall_material"])
    roof_type = a["roof_type"]
    _door_style = a["door_style"]  # TODO: use for door geometry
    _window_style = a["window_style"]  # TODO: use for window geometry
    light_color = _light_source_color(a["light_source"])
    light_level = a["light_level"]
    weather = a["weather"]
    _season = a["season"]  # TODO: use for seasonal vegetation
    _time_of_day = a["time_of_day"]  # TODO: use for lighting angle
    scale = a["scale"]
    openness = a["openness"]

    # Size from scale
    s = 2.0 + scale * 4.0  # 2 to 10 metres base size

    # ---- Floor ----
    _floor_n = max(100, int(s * s / SPACING**2))
    gx, gz = np.meshgrid(
        np.linspace(-s / 2, s / 2, int(s / SPACING)), np.linspace(-s / 2, s / 2, int(s / SPACING))
    )
    floor = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    floor += rng.normal(0, 0.008, floor.shape) * np.array([1.0, 0.0, 1.0])
    _add(parts, floor, wall_mat * 0.7)  # floor slightly darker than walls

    # ---- Walls ----
    wall_h = 3.0 + scale * 2.0  # height grows with scale
    wall_th = 0.5
    corners = [
        (-s / 2, 0, -s / 2),
        (s / 2, 0, -s / 2),
        (s / 2, 0, s / 2),
        (-s / 2, 0, s / 2),
    ]
    for i in range(len(corners)):
        c1 = np.array(corners[i])
        c2 = np.array(corners[(i + 1) % len(corners)])
        _ = (c1 + c2) / 2
        length = np.linalg.norm(c2 - c1)
        n = max(4, int(length / SPACING))
        t = np.linspace(0, 1, n)
        pts = np.stack(
            [c1[0] + t * (c2[0] - c1[0]), np.linspace(0, wall_h, n), c1[2] + t * (c2[2] - c1[2])],
            axis=1,
        )
        # thickness: extrude outward
        normal = np.array([-(c2[2] - c1[2]), 0, c2[0] - c1[0]])
        normal = normal / np.linalg.norm(normal)
        pts_out = pts + normal * wall_th
        pts_all = np.concatenate([pts, pts_out], axis=0)
        _add(parts, pts_all, wall_mat)

    # ---- Door ----
    door_w = 0.9 + scale * 0.3
    door_h = 1.8 + scale * 0.6
    door_center = np.array([corners[0][0] + door_w / 2, 0, corners[0][2] + 0.1])
    door_pts = _box(door_center, door_w / 2, door_h / 2, 0.1, max(4, int(door_w / SPACING)))
    _add(parts, door_pts, np.array([80, 50, 30], np.float64))  # wood door

    # ---- Roof ----
    if roof_type > 0:  # 0 = no roof
        _roof_h = wall_h + 1.0 + scale * 1.0
        roof_colors = [
            np.array([100, 80, 60], np.float64),  # thatched
            np.array([160, 120, 80], np.float64),  # tile
            np.array([120, 120, 130], np.float64),  # slate
            np.array([140, 100, 60], np.float64),  # shingle
        ]
        roof_color = roof_colors[min(int(roof_type) - 1, 3)]
        # Semicircular roof ribs
        rib_n = max(4, int(s / 2.0))
        for k in range(rib_n):
            z = -s / 2 + (k + 0.5) * (s / rib_n)
            th = 0.1 * s
            ang = np.linspace(0, math.pi, max(8, int(math.pi * s / SPACING)))
            x = (s / 2 - th) * np.cos(ang)
            y = wall_h + (s / 2 - th) * np.sin(ang) * 0.6
            rib = np.stack([x, y, np.full_like(x, z)], axis=1)
            _add(parts, rib, roof_color)

    # ---- Light sources ----
    light_n = max(2, int(s / 3.0))
    for _ in range(light_n):
        lx = rng.uniform(-s / 2 + 1, s / 2 - 1)
        lz = rng.uniform(-s / 2 + 1, s / 2 - 1)
        ly = 2.0 + rng.uniform(0, wall_h - 2)
        light_pos = np.array([lx, ly, lz])
        light_color_scaled = light_color * (0.3 + light_level * 0.2)
        _add(parts, light_pos.reshape(1, 3), light_color_scaled, conf=0.8)

    # ---- Atmosphere fog points (guessed, low confidence) ----
    if weather > 0 or openness > 2:
        fog_n = max(50, int(s * s / 4))
        fog_pts = rng.uniform(-s / 2, s / 2, size=(fog_n, 3))
        fog_pts[:, 1] = rng.uniform(0, wall_h, size=fog_n)
        fog_color = np.array([200, 200, 210], np.float64)  # fog grey-white
        fog_conf = 0.2 + weather * 0.15 + openness * 0.1
        _add(parts, fog_pts, fog_color, cls=2, conf=min(fog_conf, 0.5))  # guessed

    # ---- Build cloud ----
    positions = np.concatenate([p for p, _, _, _ in parts]).astype(np.float32)
    colours = np.concatenate([c for _, c, _, _ in parts]).astype(np.uint8)
    n = len(positions)
    return PointCloud(positions, colours, np.full(n, GENERATED, np.uint8), np.ones(n, np.float32))


def place_identity_from_entry(entry: dict[str, Any], seed: int = 0) -> PlaceIdentity:
    """Build a PlaceIdentity from a location dictionary entry."""
    from stages.generate.places_bible import expand

    return expand(seed, str(entry.get("requested", "")))


def render_place_from_entry(entry: dict[str, Any], seed: int = 0, size: float = 6.0) -> PointCloud:
    """Convenience: entry -> identity -> point cloud."""
    ident = place_identity_from_entry(entry, seed)
    return render_place(ident, seed, size)
