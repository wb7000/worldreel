"""Offline CPU renderer for `.wreel` worlds (Phase 6): RGB + depth + per-actor masks along a
camera path, for the flat MP4 and the `.wedit` render layers.

Mirrors the WorldReel player (player/src/render/splatShaders.ts) so exports look like what the
viewer saw: EWA projection of each Gaussian (0.3 px^2 low-pass, 3-sigma extent), front-to-back
alpha compositing, the same uncertainty fog (spec §10.3: observed with confidence >= 0.5 is
clear, otherwise blended towards a lavender gradient by 1 - confidence, at least 0.6 for
guessed / retrieved / generated, at most 0.92), point layers and actor avatars as opaque discs
with a depth buffer. The compositing loop is compiled with Numba (BSD).

Camera conventions (spec §2): right-handed, +Y up, cameras look along -Z, `fov_deg` vertical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

from eval.geometry import quat_to_matrix
from formats.wreel import GaussianCloud, PointCloud, Wreel

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]
FOG_TOP = np.array([0.86, 0.72, 0.98])
FOG_BOTTOM = np.array([0.56, 0.42, 0.82])
_TO_CV = np.diag([1.0, -1.0, -1.0])  # WorldReel camera axes -> OpenCV camera axes


@dataclass(frozen=True)
class CameraPose:
    position: F64
    rotation: F64  # quaternion [x, y, z, w], WorldReel camera-to-world
    fov_deg: float


@dataclass
class Frame:
    rgb: U8  # H x W x 3
    depth: F32  # H x W, distance along the view axis in scene units; 0 = nothing (sky)
    actor_ids: npt.NDArray[np.uint8]  # H x W, 0 = none, k = manifest actor k (1-based)


# ---- camera paths ----------------------------------------------------------------------------


def _nlerp(a: F64, b: F64, t: float) -> F64:
    if float(a @ b) < 0:
        b = -b
    q = (1 - t) * a + t * b
    return np.asarray(q / np.linalg.norm(q), np.float64)


def _sample(times: list[float], t: float) -> tuple[int, int, float]:
    if t <= times[0] or len(times) == 1:
        return 0, 0, 0.0
    if t >= times[-1]:
        return len(times) - 1, len(times) - 1, 0.0
    i = int(np.searchsorted(times, t))
    a = (t - times[i - 1]) / max(times[i] - times[i - 1], 1e-9)
    return i - 1, i, float(a)


def camera_at(manifest: dict[str, Any], camera_id: str, t: float) -> CameraPose:
    """Pose of a named camera path (spec §8) at time t: positions/fov linear, rotation nlerp."""
    cam = next(c for c in manifest["cameras"] if c["id"] == camera_id)
    keys = cam["keyframes"]
    i, j, a = _sample([float(k["t"]) for k in keys], t)
    ki, kj = keys[i], keys[j]
    pos = (1 - a) * np.asarray(ki["position"], np.float64) + a * np.asarray(
        kj["position"], np.float64
    )
    rot = _nlerp(np.asarray(ki["rotation"], np.float64), np.asarray(kj["rotation"], np.float64), a)
    fov = (1 - a) * float(ki["fov_deg"]) + a * float(kj["fov_deg"])
    return CameraPose(pos, rot, fov)


def actor_pose(track: dict[str, Any], t: float) -> tuple[F64, F64]:
    i, j, a = _sample([float(x) for x in track["times"]], t)
    p = (1 - a) * np.asarray(track["positions"][i], np.float64) + a * np.asarray(
        track["positions"][j], np.float64
    )
    q = _nlerp(
        np.asarray(track["rotations"][i], np.float64),
        np.asarray(track["rotations"][j], np.float64),
        a,
    )
    return p, q


# ---- scene preparation -----------------------------------------------------------------------


@dataclass
class Scene:
    g_pos: F64
    g_cov: F64  # N x 3 x 3 world covariance
    g_rgb: F64  # 0..1
    g_opacity: F64
    g_fog: F64  # fog amount per Gaussian (0..0.92) before strength
    p_pos: F64  # static points
    p_rgb: F64
    p_fog: F64
    p_size: F64  # world diameter
    actors: list[
        tuple[str, PointCloud, float]
    ]  # (actor id, avatar cloud in actor frame, point size)
    tracks: dict[str, dict[str, Any]]
    background: F64


def fog_amount(cls: npt.NDArray[np.integer[Any]], conf: npt.NDArray[np.floating[Any]]) -> F64:
    """Player rule (splatShaders.ts fogAmount)."""
    c = np.clip(np.asarray(conf, np.float64), 0.0, 1.0)
    a = 1.0 - c
    a = np.where(np.asarray(cls) >= 2, np.maximum(a, 0.6), a)
    a = np.where((np.asarray(cls) == 0) & (c >= 0.5), 0.0, a)
    return np.asarray(np.clip(a, 0.0, 0.92), np.float64)


def _srgb(c: npt.NDArray[np.uint8]) -> F64:
    return np.asarray(c, np.float64) / 255.0


def prepare(world: Wreel) -> Scene:
    m, e = world.manifest, world.entries
    gp, gc, gr, go, gf = [], [], [], [], []
    pp, pr, pf, ps = [], [], [], []
    for layer in m["scene"]["layers"]:
        path = sorted(layer["lods"], key=lambda lod: lod["level"])[-1]["path"]
        if layer["kind"] == "gaussians":
            g = GaussianCloud.unpack(e[path])
            rot = np.stack([quat_to_matrix(q) for q in g.rotations.astype(np.float64)])
            s2 = np.asarray(g.scales, np.float64) ** 2
            cov = np.einsum("nij,nj,nkj->nik", rot, s2, rot)
            gp.append(g.positions.astype(np.float64))
            gc.append(cov)
            gr.append(_srgb(g.colors))
            go.append(g.opacity.astype(np.float64))
            gf.append(fog_amount(g.cls, g.confidence))
        elif layer["kind"] == "points":
            pc = PointCloud.unpack(e[path])
            pp.append(pc.positions.astype(np.float64))
            pr.append(_srgb(pc.colors))
            pf.append(fog_amount(pc.cls, pc.confidence))
            ps.append(np.full(len(pc), float(layer.get("point_size", 0.05))))
    actors = []
    for a in m.get("actors", []):
        geo = a["geometry"]
        if geo["kind"] != "points":
            continue
        path = sorted(geo["lods"], key=lambda lod: lod["level"])[-1]["path"]
        actors.append((a["id"], PointCloud.unpack(e[path]), float(geo.get("point_size", 0.035))))
    tracks = {a["id"]: world.track(a["track"]["path"]) for a in m.get("actors", [])}

    def cat(parts: list[Any], shape: tuple[int, ...]) -> F64:
        return np.concatenate(parts) if parts else np.zeros(shape)

    return Scene(
        cat(gp, (0, 3)),
        cat(gc, (0, 3, 3)),
        cat(gr, (0, 3)),
        cat(go, (0,)),
        cat(gf, (0,)),
        cat(pp, (0, 3)),
        cat(pr, (0, 3)),
        cat(pf, (0,)),
        cat(ps, (0,)),
        actors,
        tracks,
        np.asarray(m["scene"].get("background", [0.55, 0.68, 0.85]), np.float64),
    )


# ---- compiled kernels ------------------------------------------------------------------------


@numba.njit(cache=True)
def _composite_splats(order, u, v, ca, cb, cc, rad, rgb, alpha0, depth, out_rgb, out_t, out_depth):  # type: ignore[no-untyped-def]
    h, w = out_t.shape
    for k in range(order.shape[0]):
        i = order[k]
        r = rad[i]
        x0, x1 = max(int(u[i] - r), 0), min(int(u[i] + r) + 1, w)
        y0, y1 = max(int(v[i] - r), 0), min(int(v[i] + r) + 1, h)
        for y in range(y0, y1):
            dy = y + 0.5 - v[i]
            for x in range(x0, x1):
                t = out_t[y, x]
                if t < 1e-3:
                    continue
                dx = x + 0.5 - u[i]
                power = -0.5 * (ca[i] * dx * dx + 2.0 * cb[i] * dx * dy + cc[i] * dy * dy)
                if power < -4.5:  # beyond 3 sigma
                    continue
                a = min(0.99, alpha0[i] * math.exp(power))
                if a < 1.0 / 255.0:
                    continue
                for ch in range(3):
                    out_rgb[y, x, ch] += t * a * rgb[i, ch]
                nt = t * (1.0 - a)
                if t >= 0.5 and nt < 0.5:  # median depth: where coverage passes 50 %
                    out_depth[y, x] = depth[i]
                out_t[y, x] = nt


@numba.njit(cache=True)
def _draw_points(u, v, z, rad, rgb, ids, zbuf, out_rgb, out_ids):  # type: ignore[no-untyped-def]
    h, w = zbuf.shape
    for i in range(u.shape[0]):
        r = rad[i]
        x0, x1 = max(int(u[i] - r), 0), min(int(u[i] + r) + 1, w)
        y0, y1 = max(int(v[i] - r), 0), min(int(v[i] + r) + 1, h)
        for y in range(y0, y1):
            for x in range(x0, x1):
                if (x + 0.5 - u[i]) ** 2 + (y + 0.5 - v[i]) ** 2 > r * r + 0.25:
                    continue
                if z[i] < zbuf[y, x]:
                    zbuf[y, x] = z[i]
                    for ch in range(3):
                        out_rgb[y, x, ch] = rgb[i, ch]
                    out_ids[y, x] = ids[i]


# ---- rendering -------------------------------------------------------------------------------


def _fogged(rgb: F64, fog: F64, screen_y: F64, strength: float) -> F64:
    t = np.clip(screen_y, 0.0, 1.0)[:, None]
    fog_col = FOG_TOP * (1 - t) + FOG_BOTTOM * t
    a = (fog * strength)[:, None]
    return np.asarray(rgb * (1 - a) + fog_col * a, np.float64)


def render(
    scene: Scene, cam: CameraPose, t: float, width: int, height: int, fog_strength: float = 1.0
) -> Frame:
    f = (height / 2) / math.tan(math.radians(cam.fov_deg) / 2)
    cx, cy = width / 2, height / 2
    r_wc = quat_to_matrix(cam.rotation)  # camera-to-world (WorldReel)
    r_cv = _TO_CV @ r_wc.T  # world -> OpenCV camera
    out_rgb = np.zeros((height, width, 3))
    out_t = np.ones((height, width))
    out_depth = np.zeros((height, width))

    if len(scene.g_pos):
        pc = (scene.g_pos - cam.position) @ r_cv.T
        z = pc[:, 2]
        ok = z > 0.02
        pc, zz = pc[ok], z[ok]
        u = cx + f * pc[:, 0] / zz
        v = cy + f * pc[:, 1] / zz
        j = np.zeros((len(zz), 2, 3))
        j[:, 0, 0] = f / zz
        j[:, 0, 2] = -f * pc[:, 0] / zz**2
        j[:, 1, 1] = f / zz
        j[:, 1, 2] = -f * pc[:, 1] / zz**2
        tm = np.einsum("nij,jk->nik", j, r_cv)
        cov2 = np.einsum("nij,njk,nlk->nil", tm, scene.g_cov[ok], tm)
        a = cov2[:, 0, 0] + 0.3
        b = cov2[:, 0, 1]
        c = cov2[:, 1, 1] + 0.3
        det = a * c - b * b
        mid = 0.5 * (a + c)
        l1 = mid + np.sqrt(np.maximum(mid * mid - det, 0.0))
        rad = 3.0 * np.sqrt(l1)
        keep = (
            (det > 1e-12)
            & (rad < 2048)
            & (u + rad >= 0)
            & (u - rad < width)
            & (v + rad >= 0)
            & (v - rad < height)
        )
        idx = np.nonzero(keep)[0]
        rgb = _fogged(scene.g_rgb[ok][idx], scene.g_fog[ok][idx], v[idx] / height, fog_strength)
        order = np.argsort(zz[idx], kind="stable").astype(np.int64)
        _composite_splats(
            order,
            u[idx],
            v[idx],
            c[idx] / det[idx],
            -b[idx] / det[idx],
            a[idx] / det[idx],
            rad[idx],
            rgb,
            scene.g_opacity[ok][idx],
            zz[idx],
            out_rgb,
            out_t,
            out_depth,
        )
    img = out_rgb + out_t[..., None] * scene.background
    depth = np.where(out_depth > 0, out_depth, np.inf)

    # Opaque points: static point layers and actor avatars, depth-tested against the splats.
    parts_pos, parts_rgb, parts_size, parts_id = [], [], [], []
    if len(scene.p_pos):
        parts_pos.append(scene.p_pos)
        parts_rgb.append(scene.p_rgb)  # fog applied below with screen y
        parts_size.append(scene.p_size)
        parts_id.append(np.zeros(len(scene.p_pos), np.uint8))
    p_fog = [scene.p_fog] if len(scene.p_pos) else []
    for k, (aid, cloud, size) in enumerate(scene.actors, start=1):
        tr = scene.tracks.get(aid)
        if tr is None or not (float(tr["times"][0]) <= t <= float(tr["times"][-1])):
            continue
        pos, rot = actor_pose(tr, t)
        world = cloud.positions.astype(np.float64) @ quat_to_matrix(rot).T + pos
        parts_pos.append(world)
        parts_rgb.append(_srgb(cloud.colors))
        parts_size.append(np.full(len(world), size))
        parts_id.append(np.full(len(world), k, np.uint8))
        p_fog.append(fog_amount(cloud.cls, cloud.confidence))
    ids_img = np.zeros((height, width), np.uint8)
    if parts_pos:
        pts = np.concatenate(parts_pos)
        pc = (pts - cam.position) @ r_cv.T
        z = pc[:, 2]
        ok = z > 0.02
        u = cx + f * pc[ok, 0] / z[ok]
        v = cy + f * pc[ok, 1] / z[ok]
        rad = np.maximum(0.7, 0.5 * f * np.concatenate(parts_size)[ok] / z[ok])
        rgb = _fogged(
            np.concatenate(parts_rgb)[ok], np.concatenate(p_fog)[ok], v / height, fog_strength
        )
        zbuf = depth.copy()
        _draw_points(u, v, z[ok], rad, rgb, np.concatenate(parts_id)[ok], zbuf, img, ids_img)
        depth = zbuf
    depth = np.where(np.isfinite(depth), depth, 0.0)
    return Frame(
        rgb=np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8),
        depth=depth.astype(np.float32),
        actor_ids=ids_img,
    )
