"""Coordinate conversions between Blender and WorldReel. Pure Python (no bpy, no numpy) so the
Blender script and the test suite share exactly the same code.

Blender world: right-handed, +Z up. WorldReel (spec/wreel.md §2): right-handed, +Y up, cameras
look along -Z. Blender cameras also look along their local -Z with local +Y up, so only the world
axes change: (x, y, z)_blender → (x, z, -y)_worldreel, a -90° rotation about X.
"""

from __future__ import annotations

import math

Vec = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # WorldReel order [x, y, z, w]

_S = math.sqrt(0.5)
ZUP_TO_YUP: Quat = (-_S, 0.0, 0.0, _S)


def quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate(q: Quat, v: Vec) -> Vec:
    x, y, z, w = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def vec_zup_to_yup(v: Vec) -> Vec:
    return (v[0], v[2], -v[1])


def quat_zup_to_yup(q_wxyz: tuple[float, float, float, float]) -> Quat:
    """Blender world rotation (mathutils order w, x, y, z) → WorldReel quaternion [x, y, z, w]."""
    w, x, y, z = q_wxyz
    q = quat_mul(ZUP_TO_YUP, (x, y, z, w))
    n = math.sqrt(sum(c * c for c in q))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


ACTOR_LOCAL_FIX: Quat = (_S, 0.0, 0.0, _S)
"""+90° about local X: maps a WorldReel actor's local frame (forward -Z, up +Y) onto a Blender
actor's local frame (forward +Y, up +Z)."""


def actor_quat_zup_to_yup(q_wxyz: tuple[float, float, float, float]) -> Quat:
    """Rotation of a Blender actor that faces its local +Y with +Z up → WorldReel actor rotation
    (faces local -Z with +Y up). An unrotated Blender actor becomes the identity."""
    q = quat_mul(quat_zup_to_yup(q_wxyz), ACTOR_LOCAL_FIX)
    n = math.sqrt(sum(c * c for c in q))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def blender_actor_yaw(direction_x: float, direction_y: float) -> float:
    """Blender Z rotation that points an actor's local +Y along (direction_x, direction_y)."""
    return math.atan2(-direction_x, direction_y)


def intrinsics(
    lens_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    sensor_fit: str,
    width: int,
    height: int,
) -> dict[str, float]:
    """Pinhole intrinsics in pixels for a Blender camera with no shift and square pixels.
    Principal point at the image centre; pixel (0, 0) centre is at (0.5, 0.5) in continuous
    coordinates, so cx = width / 2 - 0.5 in pixel-index coordinates."""
    fit = sensor_fit
    if fit == "AUTO":
        fit = "HORIZONTAL" if width >= height else "VERTICAL"
    if fit == "HORIZONTAL":
        f = lens_mm / sensor_width_mm * width
    else:
        f = lens_mm / sensor_height_mm * height
    return {
        "fx": f,
        "fy": f,
        "cx": width / 2 - 0.5,
        "cy": height / 2 - 0.5,
        "width": width,
        "height": height,
    }


def radial_to_planar(distance: float, u: float, v: float, k: dict[str, float]) -> float:
    """Distance along the pixel ray → depth along the camera's -Z axis."""
    a = (u - k["cx"]) / k["fx"]
    b = (v - k["cy"]) / k["fy"]
    return distance / math.sqrt(1 + a * a + b * b)
