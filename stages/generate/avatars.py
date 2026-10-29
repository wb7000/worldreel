"""Character avatars from the bible with a portrait face (A-14).

Local frame as `tools.make_sample_wreel.person_cloud`: feet at the origin, +Y up, facing -Z.
Primitive body parts (legs, torso, arms, neck, ellipsoid head) are sampled as surface points
every ~3 mm, sized from height and build, coloured from skin / hair / a seeded clothing palette;
the chosen portrait's face box is projected onto the front of the head. Points are marked
`generated` (provenance) with full confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from formats.wreel import PROVENANCE_CLASSES, PointCloud
from stages.generate.bible import Identity

GENERATED = PROVENANCE_CLASSES.index("generated")
SPACING = 0.003  # metres between surface points (6 mm blurred faces below SFace detail, D-8.10)
POINT_SIZE = 0.0035
F64 = npt.NDArray[np.float64]

SKIN_LIGHT = np.array([241, 206, 180], np.float64)
SKIN_DEEP = np.array([88, 56, 40], np.float64)
HAIR_YOUNG = [(28, 24, 22), (58, 40, 30), (96, 64, 42), (122, 60, 34), (150, 112, 74),
              (196, 160, 104), (150, 58, 30)]  # fmt: skip
HAIR_OLD = [(28, 24, 22), (58, 40, 30), (96, 64, 42), (150, 112, 74), (196, 160, 104),
            (160, 160, 160), (225, 225, 220)]  # fmt: skip
CLOTHES = [(62, 80, 52), (40, 52, 84), (120, 54, 44), (70, 70, 74), (150, 120, 80),
           (34, 34, 38), (96, 110, 120), (130, 90, 110)]  # fmt: skip


@dataclass(frozen=True)
class Avatar:
    cloud: PointCloud
    height: float
    eye_height: float
    head_center: tuple[float, float, float]
    head_radii: tuple[float, float, float]


def _fib_sphere(n: int) -> F64:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)


def _ellipsoid(center: F64, radii: F64) -> F64:
    a, b, c = radii
    area = 4 * math.pi * ((a * b) ** 1.6 / 3 + (a * c) ** 1.6 / 3 + (b * c) ** 1.6 / 3) ** (1 / 1.6)
    out: F64 = _fib_sphere(max(50, int(area / SPACING**2))) * radii + center
    return out


def _cylinder(x: float, z: float, rx: float, rz: float, y0: float, y1: float) -> F64:
    circ = math.pi * (3 * (rx + rz) - math.sqrt((3 * rx + rz) * (rx + 3 * rz)))
    nu, nv = max(8, int(circ / SPACING)), max(2, int((y1 - y0) / SPACING))
    u, v = np.meshgrid(np.linspace(0, 2 * math.pi, nu, endpoint=False), np.linspace(y0, y1, nv))
    out: F64 = np.stack([x + rx * np.cos(u).ravel(), v.ravel(), z + rz * np.sin(u).ravel()], 1)
    return out


def _pick(v: float, options: list[tuple[int, int, int]]) -> F64:
    return np.array(options[min(int(v * len(options)), len(options) - 1)], np.float64)


def build_avatar(
    ident: Identity,
    portrait: npt.NDArray[np.uint8] | None = None,
    face_box: tuple[float, float, float, float] | None = None,
) -> Avatar:
    """`portrait` is RGB (H, W, 3); `face_box` (x, y, w, h) in its pixels (from YuNet)."""
    a = ident.appearance
    rng = np.random.default_rng(ident.seed)
    height = 1.50 + 0.45 * a["height"]
    k = height / 1.75
    g = 0.85 + 0.40 * a["build"]
    age = 18 + 67 * a["age"]
    skin = SKIN_LIGHT + (SKIN_DEEP - SKIN_LIGHT) * a["skin_tone"]
    hair = _pick(a["hair_tone"], HAIR_OLD if age >= 50 else HAIR_YOUNG)
    top = np.array(CLOTHES[int(rng.integers(len(CLOTHES)))], np.float64)
    bottom = np.array(CLOTHES[int(rng.integers(len(CLOTHES)))], np.float64) * 0.7
    shoe = np.array([30, 28, 26], np.float64)

    parts: list[tuple[F64, F64]] = []

    def add(points: F64, color: F64) -> None:
        parts.append((points, np.broadcast_to(color, points.shape).copy()))

    leg_top = 0.50 * height
    for side in (-1, 1):
        add(
            _cylinder(side * 0.09 * g * k, 0.0, 0.065 * g * k, 0.07 * g * k, 0.06 * k, leg_top),
            bottom,
        )
        add(_cylinder(side * 0.09 * g * k, -0.03 * k, 0.05 * k, 0.09 * k, 0.0, 0.06 * k), shoe)
    torso_top = 0.82 * height
    add(_cylinder(0.0, 0.0, 0.17 * g * k, 0.11 * g * k, leg_top - 0.04 * k, torso_top), top)
    for side in (-1, 1):
        ax = side * (0.17 * g * k + 0.05 * k)
        add(
            _cylinder(ax, 0.0, 0.045 * g * k, 0.045 * g * k, 0.44 * height, torso_top - 0.02 * k),
            top,
        )
        add(
            _ellipsoid(np.array([ax, 0.42 * height, 0.0]), np.array([0.045, 0.06, 0.045]) * k), skin
        )
    rx = 0.075 * k * (0.9 + 0.2 * a["face_width"])
    ry, rz = 0.115 * k, 0.095 * k
    cy = height - ry
    add(_cylinder(0.0, 0.0, 0.05 * k, 0.05 * k, torso_top - 0.01 * k, cy - 0.7 * ry), skin)

    head = _ellipsoid(np.array([0.0, cy, 0.0]), np.array([rx, ry, rz]))
    col = np.broadcast_to(skin, head.shape).copy()
    rel = (head - [0.0, cy, 0.0]) / [rx, ry, rz]
    bald = a["hair_length"] < 0.05
    if not bald:  # hair cap: top and back of the head
        cap = (rel[:, 1] > 0.35) | ((rel[:, 2] > 0.1) & (rel[:, 1] > -0.4))
        col[cap] = hair
    if portrait is not None and face_box is not None:
        h_img, w_img = portrait.shape[:2]
        fx, fy, fw, fh = face_box
        # expand the detector's box to forehead / chin / ears, then map the head's front onto it
        # (a uniform eye-anchored scale was tried: characters rendered less distinct, D-8.10)
        x0, x1 = fx - 0.18 * fw, fx + 1.18 * fw
        y0, y1 = fy - 0.30 * fh, fy + 1.12 * fh
        front = rel[:, 2] < -0.15
        u = x0 + (1 - (rel[:, 0] + 1) / 2) * (x1 - x0)  # camera faces the avatar: image u ~ -x
        v = y0 + (1 - (rel[:, 1] + 1) / 2) * (y1 - y0)
        inside = front & (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
        col[inside] = portrait[v[inside].astype(int), u[inside].astype(int)].astype(np.float64)
    parts.append((head, col))
    if not bald and a["hair_length"] > 0.55:  # long hair falls behind the shoulders
        add(
            _cylinder(
                0.0, 0.03 * k, rx * 0.95, rz * 0.7, cy - ry - 0.25 * k * a["hair_length"], cy
            ),
            hair,
        )

    pos = np.concatenate([p for p, _ in parts]).astype(np.float32)
    rgb = np.clip(np.concatenate([c for _, c in parts]), 0, 255).astype(np.uint8)
    n = len(pos)
    cloud = PointCloud(pos, rgb, np.full(n, GENERATED, np.uint8), np.ones(n, np.float32))
    return Avatar(cloud, round(height, 3), round(cy - 0.1 * ry, 3), (0.0, cy, 0.0), (rx, ry, rz))
