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

from stages.generate.face_landmarks import LIPS_OUTER, NOSE_BASE, OVAL, WARP

U8 = npt.NDArray[np.uint8]
F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Aligned:
    rgba: U8  # BGRA in the face render's pixels
    facial_hair: list[tuple[str, U8]]  # (moustache / stubble / beard, 0-255 mask, render px)
    iris_rgb: tuple[float, float, float]  # sRGB 0-1, sampled from the portrait
    fit_error_px: float  # mean landmark residual after the similarity fit, before the spline


def landmarks(face: npt.NDArray[np.float32]) -> F64:
    """YuNet row -> 5 x 2 points: right eye, left eye, nose tip, right / left mouth corner."""
    return np.asarray(face[4:14], np.float64).reshape(5, 2)


def _similarity(src: F64, dst: F64) -> F64:
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    out: F64 = np.asarray(m, np.float64)
    return out


def _warp(
    img: U8,
    src: F64,
    dst: F64,
    size: tuple[int, int],
    src_dense: F64 | None = None,
    dst_dense: F64 | None = None,
) -> tuple[U8, float]:
    """Similarity on the five YuNet points, then a thin-plate spline on the residuals (anchored
    by a fixed frame so the correction stays local to the face). With dense landmarks (478,
    `face_landmarks.WARP` subset) the spline matches lips, eyes, brows, nose and jaw line
    (D-8.11e: five points left the portrait's lips above / below the mesh's)."""
    m = _similarity(src, dst)
    w, h = size
    sim = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    moved = src @ m[:, :2].T + m[:, 2]
    err = float(np.mean(np.linalg.norm(moved - dst, axis=1)))
    if src_dense is not None and dst_dense is not None:
        idx = list(WARP)
        moved = src_dense[idx] @ m[:, :2].T + m[:, 2]
        dst = dst_dense[idx]
        err = float(np.mean(np.linalg.norm(moved - dst, axis=1)))
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
    # the photo's key light came from one side: left and right halves differed by up to 7 L and
    # the owner saw two different sides (D-8.11d). Even out the large-scale lightness across the
    # face's midline (mirror about the eyes' centre), keeping each side's detail.
    mx = (2 * eyes[0] - xx).astype(np.float32)
    for sigma in (0.5, 0.25):  # broad, then cheek-sized shading; finer detail stays one-sided
        big = cv2.GaussianBlur(lab[..., 0], (0, 0), sigma * ipd).astype(np.float32)
        mirrored = cv2.remap(big, mx, yy.astype(np.float32), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
        lab[..., 0] += (mirrored - big) / 2
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


def eye_texture(portrait: U8, lm: F64, skin_tone: float) -> str:
    """MakeHuman eye texture name (CC0 pack: brown, brownlight, blue, ...). The portraits' irises
    measure nearly colourless (Lab a, b within 6 of neutral: small, lit eyes), so a rule, not a
    colour match: deep skin or dark irises -> brown; a warm tint -> brownlight (hazel); else
    blue. MPFB's procedural eye rendered every iris pale blue-grey (owner, D-8.11d)."""
    ipd = float(np.linalg.norm(lm[1] - lm[0]))
    lab = cv2.cvtColor(portrait, cv2.COLOR_BGR2LAB).astype(np.float64)
    yy, xx = np.mgrid[: portrait.shape[0], : portrait.shape[1]]
    rings = []
    for e in lm[:2]:
        r = np.hypot(xx - e[0], yy - e[1])
        rings.append(lab[(r > 0.04 * ipd) & (r < 0.09 * ipd)])
    px = np.vstack(rings)
    keep = (px[:, 0] > np.percentile(px[:, 0], 30)) & (px[:, 0] < np.percentile(px[:, 0], 80))
    lightness, _, b = np.median(px[keep], 0)
    if skin_tone > 0.6 or lightness < 40:
        return "brown"
    return "brownlight" if b - 128 > 1.5 else "blue"


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


FACIAL_HAIR = ("clean-shaven", "light stubble", "moustache", "full beard")  # as portraits.py
# strand length (m at 1.75 m height) and density (per m2) per piece
STUBBLE = (0.002, 160000.0)  # sparse: dense stubble read as fur
MOUSTACHE = (0.006, 2500000.0)
BEARD = (0.02, 1500000.0)


def facial_hair_kind(appearance: dict[str, float]) -> str:
    """The same four steps the portrait prompt used (portraits.appearance_words), so the
    strands follow what the image model was asked to draw."""
    if appearance["presentation"] < 0.35:
        return "clean-shaven"
    v = appearance["facial_hair"]
    return FACIAL_HAIR[min(int(v * len(FACIAL_HAIR)), len(FACIAL_HAIR) - 1)]


def facial_hair_masks(dense_lm: F64, size: tuple[int, int], kind: str) -> list[tuple[str, U8]]:
    """Template regions from the 478 landmarks (D-8.11e: colour thresholds flooded shaded
    cheeks and chins): `moustache` between the nose base and the upper lip, reaching a little
    past and below the mouth corners; `beard` the face oval below the nose base minus the lips.
    Returns [(piece, 0-255 mask in render pixels)]."""
    w, h = size
    lm = dense_lm
    left, right = lm[61], lm[291]
    mouth_w = float(np.linalg.norm(right - left))
    out: list[tuple[str, U8]] = []
    beard = np.zeros((h, w), np.uint8)
    cv2.fillPoly(beard, [np.round(lm[list(OVAL)]).astype(np.int32)], 255)
    beard[: int(np.max(lm[list(NOSE_BASE)][:, 1])), :] = 0
    lips = np.zeros((h, w), np.uint8)
    cv2.fillPoly(lips, [np.round(lm[list(LIPS_OUTER)]).astype(np.int32)], 255)
    beard[lips > 0] = 0
    blur_s = 0.05 * mouth_w + 1
    if kind in ("moustache", "full beard"):
        # roots end a little above the upper lip's edge (on it the hair curtained over the lip)
        lip_h = float(np.linalg.norm(lm[0] - lm[13]))
        upper = [
            lm[i] - np.array([0.0, 0.35 * lip_h])
            for i in (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
        ]
        tip = 0.12 * mouth_w

        def off(p: F64, dx: float, dy: float) -> F64:
            out: F64 = p + np.array([dx, dy])
            return out

        poly = [off(left, -tip, 0.25 * tip), off(left, -0.6 * tip, -0.8 * tip), lm[98], lm[2],
                lm[327], off(right, 0.6 * tip, -0.8 * tip),
                off(right, tip, 0.25 * tip)]  # fmt: skip
        poly += upper[::-1]
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 255)
        m[lips > 0] = 0
        out.append(("moustache", np.asarray(cv2.GaussianBlur(m, (0, 0), 0.4 * blur_s), np.uint8)))
    if kind == "light stubble":
        out.append(("stubble", np.asarray(cv2.GaussianBlur(beard, (0, 0), blur_s), np.uint8)))
    if kind == "full beard":
        out.append(("beard", np.asarray(cv2.GaussianBlur(beard, (0, 0), blur_s), np.uint8)))
    return out


def align(
    portrait: U8,
    p_face: npt.NDArray[np.float32],
    render: U8,
    r_face: npt.NDArray[np.float32],
    body_rgb: tuple[float, float, float],
    p_dense: F64 | None = None,
    r_dense: F64 | None = None,
    kind: str = "clean-shaven",
) -> Aligned:
    """`body_rgb`: the body skin's albedo after tinting (sRGB 0-1); the face is matched to it.
    `p_dense` / `r_dense`: 478 MediaPipe landmarks of portrait and render (face_landmarks)."""
    size = (render.shape[1], render.shape[0])
    src, dst = landmarks(p_face), landmarks(r_face)
    warped, err = _warp(portrait, src, dst, size, p_dense, r_dense)
    yy, xx = np.mgrid[: size[1], : size[0]]
    ipd = float(np.linalg.norm(dst[1] - dst[0]))
    eyes = (dst[0] + dst[1]) / 2
    spots = [dst[0] + (0, 0.55 * ipd), dst[1] + (0, 0.55 * ipd), eyes - (0, 0.55 * ipd)]
    region = np.zeros(size[::-1], bool)
    for x, y in spots:
        region |= np.hypot(xx - x, yy - y) < 0.15 * ipd
    pieces: list[tuple[str, U8]] = []
    if r_dense is not None and kind != "clean-shaven":
        pieces = facial_hair_masks(r_dense, size, kind)
        for name, m in pieces:
            if name in ("moustache", "beard"):  # long hair: grown as strands, not painted
                hole = (m > 40).astype(np.uint8)
                radius = max(3, int(0.06 * ipd))
                warped = np.asarray(cv2.inpaint(warped, hole, radius, cv2.INPAINT_TELEA), np.uint8)
    warped = delight(warped, dst)
    mask = face_mask(dst, size) * skin_above_eyes(warped, dst)
    warped = _match_colour(warped, body_rgb, region)
    rgba = np.dstack([warped, np.round(mask * 255).astype(np.uint8)])
    return Aligned(rgba, pieces, iris_colour(portrait, src), err)
