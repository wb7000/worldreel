"""Portrait -> MPFB face (A-14 v2, D-8.11).

Blender renders the built human straight on through a fixed orthographic face camera
(`tools/blender/human.py build`). Here the chosen portrait is warped into that render's pixels:
YuNet's five landmarks (eyes, nose tip, mouth corners) on both images drive a similarity fit
plus a thin-plate-spline correction, so the portrait's eyes / nose / mouth land on the mesh's.
The warped portrait gets a feathered face-oval alpha and its colour is matched to the body skin
around the oval's edge. Blender then projects it from the same camera (`... portrait`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

U8 = npt.NDArray[np.uint8]
F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Aligned:
    rgba: U8  # BGRA in the face render's pixels
    iris_rgb: tuple[float, float, float]  # sRGB 0-1, sampled from the portrait
    fit_error_px: float  # mean landmark residual after the similarity fit, before the spline


def landmarks(face: npt.NDArray[np.float32]) -> F64:
    """YuNet row -> 5 x 2 points: right eye, left eye, nose tip, right / left mouth corner."""
    return np.asarray(face[4:14], np.float64).reshape(5, 2)


def _similarity(src: F64, dst: F64) -> F64:
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    out: F64 = np.asarray(m, np.float64)
    return out


def _warp(img: U8, src: F64, dst: F64, size: tuple[int, int]) -> tuple[U8, float]:
    """Similarity, then a thin-plate spline on the residuals (anchored by a fixed frame so the
    correction stays local to the face)."""
    m = _similarity(src, dst)
    w, h = size
    sim = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    moved = src @ m[:, :2].T + m[:, 2]
    frame = np.array(
        [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
         [w / 2, 0], [w / 2, h - 1], [0, h / 2], [w - 1, h / 2]], np.float64
    )  # fmt: skip
    # backward map: each output pixel samples `sim` where the spline sends it
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    grid = np.stack([xx.ravel(), yy.ravel()], 1)
    mapped = tps(np.vstack([dst, frame]), np.vstack([moved, frame]), grid)
    mx = mapped[:, 0].reshape(h, w).astype(np.float32)
    my = mapped[:, 1].reshape(h, w).astype(np.float32)
    out = np.asarray(
        cv2.remap(sim, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), np.uint8
    )
    err = float(np.mean(np.linalg.norm(moved - dst, axis=1)))
    return out, err


def tps(ctrl: F64, values: F64, pts: F64) -> F64:
    """2-D thin-plate spline through `ctrl` -> `values`, evaluated at `pts` (U = r² log r)."""
    n = len(ctrl)

    def u(d: F64) -> F64:
        r2 = d**2
        out: F64 = np.where(r2 > 0, r2 * np.log(np.sqrt(r2) + 1e-12), 0.0)
        return out

    k = u(np.linalg.norm(ctrl[:, None] - ctrl[None], axis=2))
    p = np.hstack([np.ones((n, 1)), ctrl])
    a = np.zeros((n + 3, n + 3))
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
    b = np.zeros((n + 3, 2))
    b[:n] = values
    coef = np.linalg.solve(a, b)
    out: F64 = np.empty((len(pts), 2))
    for s in range(0, len(pts), 1 << 16):  # chunks keep memory small
        q = pts[s : s + (1 << 16)]
        kq = u(np.linalg.norm(q[:, None] - ctrl[None], axis=2))
        out[s : s + len(q)] = kq @ coef[:n] + np.hstack([np.ones((len(q), 1)), q]) @ coef[n:]
    return out


def face_mask(lm: F64, size: tuple[int, int]) -> npt.NDArray[np.float32]:
    """Feathered oval from the landmarks: brow to chin, cheek to cheek (0-1 float)."""
    w, h = size
    eyes = (lm[0] + lm[1]) / 2
    mouth = (lm[3] + lm[4]) / 2
    ipd = float(np.linalg.norm(lm[1] - lm[0]))
    down = mouth - eyes
    centre = eyes + 0.35 * down
    # bottom at the chin (~1.65 x eyes-to-mouth below the eyes): lower, the portrait's neck
    # shadow and collar were painted onto the mesh's neck (D-8.11)
    axes = (int(1.05 * ipd), int(1.3 * np.linalg.norm(down)))
    angle = float(np.degrees(np.arctan2(down[0], down[1]))) * -1
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (int(centre[0]), int(centre[1])), axes, angle, 0, 360, 1.0, -1)
    k = int(0.35 * ipd) | 1
    out = np.asarray(cv2.GaussianBlur(mask, (0, 0), k / 2), np.float32)
    return out


def skin_colour(portrait: U8, lm: F64) -> tuple[float, float, float]:
    """sRGB 0-1 median of the cheeks and forehead (clear of eyes, brows, mouth and beard line)."""
    ipd = float(np.linalg.norm(lm[1] - lm[0]))
    eyes = (lm[0] + lm[1]) / 2
    spots = [lm[0] + (0, 0.55 * ipd), lm[1] + (0, 0.55 * ipd), eyes - (0, 0.55 * ipd)]
    yy, xx = np.mgrid[: portrait.shape[0], : portrait.shape[1]]
    px = np.vstack([portrait[np.hypot(xx - x, yy - y) < 0.15 * ipd] for x, y in spots])
    bgr = np.median(px.astype(np.float64), 0) / 255.0
    return (float(bgr[2]), float(bgr[1]), float(bgr[0]))


PHOTO_TO_ALBEDO = 0.75
DELIGHT = 1.0
FLOOR = 22.0  # Lab lightness units


def hair_colour(
    portrait: U8, lm: F64, wanted: tuple[float, float, float]
) -> tuple[float, float, float]:
    """sRGB 0-1: the bible's hair colour (`wanted`, what the portrait prompt asked for), darkened
    to the portrait's hair lightness when the image model drew it darker (Lio's "light brown"
    came out dark brown). Colours sampled from the photo alone were unreliable — highlights,
    shadowed curls and background through partings turned black hair grey, white hair grey
    and auburn black (D-8.11c)."""
    ipd = float(np.linalg.norm(lm[1] - lm[0]))
    x, y = (lm[0] + lm[1]) / 2 - (0, 1.3 * ipd)
    yy, xx = np.mgrid[: portrait.shape[0], : portrait.shape[1]]
    sel = np.hypot(xx - x, yy - max(y, 0.2 * ipd)) < 0.3 * ipd
    photo_l = float(np.median(cv2.cvtColor(portrait, cv2.COLOR_BGR2LAB)[sel][:, 0]))
    px = np.array([[[wanted[2], wanted[1], wanted[0]]]], np.float32)
    wanted_l = float(cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0, 0])  # float input: L 0-100
    # scale RGB (keeps the hue; lowering Lab lightness clipped red to pure dark red), at most
    # by half
    f = float(np.clip(photo_l * 100 / 255 / max(wanted_l, 1e-3), 0.5, 1.0))
    r, g, b = (float(c * f) for c in wanted)
    return (r, g, b)


def skin_above_eyes(bgr: U8, lm: F64) -> npt.NDArray[np.float32]:
    """1 where a pixel looks like skin, else 0 — only above the eyes, so the portrait's own hair
    at the hairline is not painted onto the forehead (D-8.11c). Below the eyes everything stays
    (moustache and beard are part of the face). Skin = within Lab distance 18 of the forehead /
    cheek median, feathered."""
    ipd = float(np.linalg.norm(lm[1] - lm[0]))
    eyes = (lm[0] + lm[1]) / 2
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    yy, xx = np.mgrid[: bgr.shape[0], : bgr.shape[1]]
    ref_spots = [lm[0] + (0, 0.55 * ipd), lm[1] + (0, 0.55 * ipd), eyes - (0, 0.45 * ipd)]
    ref = np.zeros(bgr.shape[:2], bool)
    for x, y in ref_spots:
        ref |= np.hypot(xx - x, yy - y) < 0.12 * ipd
    skin = np.median(lab[ref], 0)
    dist = np.linalg.norm((lab - skin) * (0.5, 1.0, 1.0), axis=2)  # lightness counts half
    keep = (dist < 18).astype(np.float32)
    keep[yy > eyes[1] - 0.2 * ipd] = 1.0
    out = np.asarray(cv2.GaussianBlur(keep, (0, 0), 0.06 * ipd), np.float32)
    return out


def delight(bgr: U8, lm: F64, strength: float = DELIGHT) -> U8:
    """Remove the photo's broad studio shading (nose-side, cheekbone and jaw shadows): Blender
    lights the 3D face itself, so baked shadows rendered as black patches (owner, D-8.11c).
    Shading = median filter of Lab lightness (0.45 IPD wide, so eyes, brows and the moustache do
    not count as shadow; a plain blur lit up the eye sockets), smoothed; lightness minus shading
    plus its median: wrinkles, pores, stubble, brows and eyes stay."""
    ipd = float(np.linalg.norm(lm[1] - lm[0]))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    small = cv2.resize(lab[..., 0].astype(np.uint8), None, fx=0.25, fy=0.25)
    ksize = max(3, int(0.45 * ipd * 0.25) | 1)
    med = cv2.medianBlur(small, min(ksize, 255)).astype(np.float64)
    med = np.asarray(
        cv2.resize(med, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR), np.float64
    )
    shade = cv2.GaussianBlur(med, (0, 0), 0.15 * ipd)
    eyes = (lm[0] + lm[1]) / 2
    yy, xx = np.mgrid[: bgr.shape[0], : bgr.shape[1]]
    face = np.hypot(xx - eyes[0], yy - eyes[1] - 0.6 * ipd) < 1.0 * ipd
    lab[..., 0] += strength * (np.median(shade[face]) - shade)
    # leftover mid-scale shadows (nose sides, cheek folds) rendered as dark patches: skin may
    # not fall more than FLOOR below the face's median lightness, except where the face really
    # is dark (eyes, brows, nostrils, mouth, moustache)
    feature = np.zeros(bgr.shape[:2], bool)
    for eye in lm[:2]:
        feature |= np.hypot(xx - eye[0], yy - eye[1]) < 0.38 * ipd
    feature |= (np.abs(xx - eyes[0]) < 0.95 * ipd) & (
        np.abs(yy - (eyes[1] - 0.3 * ipd)) < 0.18 * ipd
    )
    mouth = (lm[3] + lm[4]) / 2
    feature |= (
        (np.abs(xx - mouth[0]) < 0.75 * ipd)
        & (yy > lm[2][1] - 0.1 * ipd)
        & (yy < mouth[1] + 0.3 * ipd)
    )
    floor = np.median(lab[..., 0][face]) - FLOOR
    lab[..., 0] = np.where(feature, lab[..., 0], np.maximum(lab[..., 0], floor))
    out = np.asarray(
        cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR), np.uint8
    )
    return out


def curliness(portrait: U8, lm: F64) -> float:
    """0 straight / smooth .. 1 tight curls: band-pass energy of the hair above the forehead
    (scale ~ IPD / 40 .. IPD / 10). Calibrated on the four Lantern Keeper portraits (D-8.11c):
    0.03 short smooth, 0.14-0.15 waves / curls in a dark photo, 0.24 tight curls."""
    ipd = float(np.linalg.norm(lm[1] - lm[0]))
    x, y = (lm[0] + lm[1]) / 2 - (0, 1.3 * ipd)
    r = int(0.3 * ipd)
    cy = int(max(y, r))
    g = cv2.cvtColor(portrait, cv2.COLOR_BGR2GRAY).astype(np.float64)
    patch = g[cy - r : cy + r, int(x) - r : int(x) + r]
    band = cv2.GaussianBlur(patch, (0, 0), ipd / 40) - cv2.GaussianBlur(patch, (0, 0), ipd / 10)
    energy = float(band.std() / (patch.mean() + 1))
    return float(np.clip((energy - 0.05) / 0.15, 0.0, 1.0))


def body_colour(
    texture_rgb: tuple[float, float, float], portrait_rgb: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Skin albedo for body and face (sRGB 0-1): the skin texture's lightness (a real albedo)
    with the hue halfway to the portrait's. The portrait is lit studio skin, far brighter than
    an albedo; used as is it bleached the whole character (D-8.11)."""
    lab = []
    for rgb in (texture_rgb, portrait_rgb):
        px = np.array([[[rgb[2], rgb[1], rgb[0]]]], np.float32)
        lab.append(cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0])
    # lightness: the texture's, but never above the portrait's at studio exposure (~1.33x an
    # albedo) — light skin textures alone rendered chalk-white (D-8.11)
    light = min(lab[0][0], PHOTO_TO_ALBEDO * lab[1][0])
    mixed = np.array([light, (lab[0][1] + lab[1][1]) / 2, (lab[0][2] + lab[1][2]) / 2])
    bgr = cv2.cvtColor(mixed.astype(np.float32).reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]
    b, g, r = (float(np.clip(c, 0, 1)) for c in bgr)
    return (r, g, b)


def _match_colour(src: U8, target_rgb: tuple[float, float, float],
                  region: npt.NDArray[np.bool_]) -> U8:  # fmt: skip
    """Shift the Lab mean of `region` to the target colour; the portrait's own detail stays."""
    a = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float64)
    tgt = np.array([[[target_rgb[2], target_rgb[1], target_rgb[0]]]], np.float32)
    t = cv2.cvtColor(np.round(tgt * 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float64)[
        0, 0
    ]
    for c in range(3):
        a[..., c] += t[c] - np.median(a[..., c][region])
    out = np.asarray(cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR), np.uint8)
    return out


def iris_colour(portrait: U8, lm: F64) -> tuple[float, float, float]:
    """Median colour in a ring around both eye centres, without sclera / highlights / pupil."""
    ipd = float(np.linalg.norm(lm[1] - lm[0]))
    yy, xx = np.mgrid[: portrait.shape[0], : portrait.shape[1]]
    picks = []
    for eye in lm[:2]:
        r = np.hypot(xx - eye[0], yy - eye[1])
        ring = (r > 0.035 * ipd) & (r < 0.08 * ipd)
        px = portrait[ring].astype(np.float64)
        lum = px.mean(1)
        keep = (lum > np.percentile(lum, 20)) & (lum < np.percentile(lum, 70))
        picks.append(px[keep])
    bgr = np.median(np.vstack(picks), 0) / 255.0
    return (float(bgr[2]), float(bgr[1]), float(bgr[0]))


def align(
    portrait: U8,
    p_face: npt.NDArray[np.float32],
    render: U8,
    r_face: npt.NDArray[np.float32],
    body_rgb: tuple[float, float, float],
) -> Aligned:
    """`body_rgb`: the body skin's albedo after tinting (sRGB 0-1); the face is matched to it."""
    size = (render.shape[1], render.shape[0])
    src, dst = landmarks(p_face), landmarks(r_face)
    warped, err = _warp(portrait, src, dst, size)
    warped = delight(warped, dst)
    mask = face_mask(dst, size) * skin_above_eyes(warped, dst)
    # cheeks / forehead of the warped portrait sit where the render's are
    yy, xx = np.mgrid[: size[1], : size[0]]
    ipd = float(np.linalg.norm(dst[1] - dst[0]))
    eyes = (dst[0] + dst[1]) / 2
    spots = [dst[0] + (0, 0.55 * ipd), dst[1] + (0, 0.55 * ipd), eyes - (0, 0.55 * ipd)]
    region = np.zeros(mask.shape, bool)
    for x, y in spots:
        region |= np.hypot(xx - x, yy - y) < 0.15 * ipd
    warped = _match_colour(warped, body_rgb, region)
    rgba = np.dstack([warped, np.round(mask * 255).astype(np.uint8)])
    return Aligned(rgba, iris_colour(portrait, src), err)
