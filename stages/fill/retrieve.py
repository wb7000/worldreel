"""Licensed reference images, geometrically verified (algorithm A-4, docs/ALGORITHMS.md).

Candidates come from the owner (a folder of images with licence sidecars) or, when explicitly
enabled, from open-licensed sources (Openverse, Wikimedia Commons). A candidate is used only if
its licence allows derivatives, it shows no people (metadata + face detector) and it registers
geometrically to the reconstruction (SIFT matches -> 2D-3D -> PnP, confirmed in a second frame).
Accepted images colour unseen elements (`retrieved`) and are recorded with URL + licence.

Network access is OFF unless a contact string for the User-Agent is configured (Wikimedia
User-Agent policy) and the item is marked as a public outdoor place. Rooms/homes: never.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from pipeline.io import read_rgb

F64 = npt.NDArray[np.float64]
U8 = npt.NDArray[np.uint8]
_FLIP = np.diag([1.0, -1.0, -1.0])
FACE_MODEL = Path("data/models/face_detection_yunet_2023mar.onnx")  # MIT, OpenCV model zoo
PEOPLE_WORDS = re.compile(
    r"\b(people|person|persons|portrait|portraits|selfie|selfies|crowd|crowds|man|men|woman|"
    r"women|child|children|girl|girls|boy|boys|interior|interiors|bedroom|living room|home|homes|"
    r"house interior|residential)\b",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    image: U8 | None
    url: str
    license: str  # normalised SPDX-style id, e.g. "CC-BY-SA-4.0"
    source: str  # "user", "openverse", "commons"
    attribution: str = ""
    landing_url: str = ""
    metadata_text: str = ""  # titles, tags, categories (for the people/interior filter)


@dataclass
class Verified:
    ok: bool
    reason: str
    r: F64 = field(default_factory=lambda: np.eye(3))  # OpenCV world-to-camera
    t: F64 = field(default_factory=lambda: np.zeros(3))
    k: F64 = field(default_factory=lambda: np.eye(3))
    inliers: int = 0
    inlier_ratio: float = 0.0
    median_px: float = float("inf")
    second_frame_inliers: int = 0
    off_plane_frac: float = 0.0  # inlier 3D points off the dominant plane
    photo_agree: float = 0.0  # share of visible observed points whose colour agrees


# ---- licences -----------------------------------------------------------------------------

_ALLOWED = {"cc0": "CC0-1.0", "pdm": "PDM", "by": "CC-BY", "by-sa": "CC-BY-SA"}


def normalise_license(text: str, version: str = "") -> str | None:
    """Openverse codes (`by-sa`, `cc0`, `pdm`) or Commons short names ("CC BY-SA 4.0",
    "Public domain", "CC0") -> an id, or None when the licence forbids derivatives (ND),
    commercial use (NC) or is unknown/ambiguous."""
    t = text.strip().lower().replace("_", "-")
    if not t or "nc" in re.split(r"[\s\-]", t) or "nd" in re.split(r"[\s\-]", t):
        return None
    if t in _ALLOWED:
        base = _ALLOWED[t]
    elif t.startswith("cc0"):
        base = "CC0-1.0"
    elif t in ("public domain", "pd", "pdm", "public domain mark"):
        base = "PDM"
    elif m := re.fullmatch(r"cc[\s\-]by[\s\-]sa(?:[\s\-]([\d.]+))?", t):
        base, version = "CC-BY-SA", version or (m.group(1) or "")
    elif m := re.fullmatch(r"cc[\s\-]by(?:[\s\-]([\d.]+))?", t):
        base, version = "CC-BY", version or (m.group(1) or "")
    else:
        return None
    return f"{base}-{version}" if version and base.startswith("CC-BY") else base


# ---- people / private-place filters -------------------------------------------------------


def mentions_people(text: str) -> bool:
    return bool(PEOPLE_WORDS.search(text or ""))


def has_face(image: U8, model: Path = FACE_MODEL, score: float = 0.6) -> bool:
    """YuNet face detector (OpenCV FaceDetectorYN). Raises if the model file is missing, so a
    missing filter can never silently let images of people through."""
    if not model.is_file():
        raise FileNotFoundError(f"face model missing: {model} (see docs/MODELS.md)")
    h, w = image.shape[:2]
    det = cv2.FaceDetectorYN.create(str(model), "", (w, h), score)
    _, faces = det.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return faces is not None and len(faces) > 0


# ---- sources ------------------------------------------------------------------------------


def user_candidates(folder: Path) -> list[Candidate]:
    """Owner-provided references: `<image>` + `<image>.info.json` with `license` (required),
    `source` / `url` / `attribution` (optional). Images without an allowed licence are skipped."""
    out = []
    for p in sorted(folder.glob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        side = p.with_name(p.name + ".info.json")
        if not side.is_file():
            continue
        info = json.loads(side.read_text(encoding="utf-8"))
        lic = normalise_license(str(info.get("license", "")))
        if lic is None and str(info.get("license", "")).lower() not in (
            "owner",
            "own work by owner",
        ):
            continue
        out.append(
            Candidate(
                image=read_rgb(p),
                url=str(info.get("url", f"user:{p.name}")),
                license=lic or "owner",
                source="user",
                attribution=str(info.get("attribution", "")),
                metadata_text=str(info.get("notes", "")),
            )
        )
    return out


HttpGet = Callable[[str, dict[str, str]], bytes]


def _default_get(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data: bytes = resp.read()
    return data


def user_agent(contact: str) -> str:
    if not contact.strip():
        raise PermissionError("retrieval needs a contact for the User-Agent (Wikimedia policy)")
    return f"WorldReel/0.4 ({contact.strip()}) python-urllib"


def openverse_search(
    query: str, contact: str, *, page_size: int = 20, get: HttpGet = _default_get
) -> list[Candidate]:
    """Openverse `GET /v1/images/` (anonymous). Licence filter applied server- and client-side."""
    params = {
        "q": query[:200],
        "license": "cc0,pdm,by,by-sa",
        "category": "photograph",
        "mature": "false",
        "page_size": str(page_size),
    }
    url = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(params)
    data = json.loads(get(url, {"User-Agent": user_agent(contact)}))
    out = []
    for r in data.get("results", []):
        lic = normalise_license(str(r.get("license", "")), str(r.get("license_version") or ""))
        if lic is None or r.get("mature"):
            continue
        tags = " ".join(str(t.get("name", "")) for t in r.get("tags") or [])
        out.append(
            Candidate(
                image=None,
                url=str(r.get("url", "")),
                license=lic,
                source="openverse",
                attribution=str(r.get("attribution", "")),
                landing_url=str(r.get("foreign_landing_url", "")),
                metadata_text=f"{r.get('title', '')} {tags}",
            )
        )
    return out


def commons_geosearch(
    lat: float,
    lon: float,
    contact: str,
    *,
    radius_m: int = 500,
    limit: int = 30,
    get: HttpGet = _default_get,
) -> list[Candidate]:
    """Wikimedia Commons files near a coordinate (MediaWiki geosearch generator + imageinfo)."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "geosearch",
        "ggscoord": f"{lat}|{lon}",
        "ggsradius": str(radius_m),
        "ggslimit": str(limit),
        "ggsnamespace": "6",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    return _commons_pages(json.loads(get(url, {"User-Agent": user_agent(contact)})))


def _commons_pages(data: dict[str, Any]) -> list[Candidate]:
    out = []
    for page in (data.get("query", {}).get("pages") or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}

        def val(key: str, meta: dict[str, Any] = meta) -> str:
            return str((meta.get(key) or {}).get("value", ""))

        lic = normalise_license(val("LicenseShortName"))
        if lic is None or "|" in val("LicenseShortName"):  # multi-licensed: unreliable, skip
            continue
        out.append(
            Candidate(
                image=None,
                url=str(info.get("url", "")),
                license=lic,
                source="commons",
                attribution=re.sub(r"<[^>]+>", "", val("Artist")).strip(),
                landing_url=str(info.get("descriptionurl", "")),
                metadata_text=" ".join(
                    [str(page.get("title", "")), val("Categories"), val("ImageDescription")]
                ),
            )
        )
    return out


def commons_search(
    query: str, contact: str, *, limit: int = 30, get: HttpGet = _default_get
) -> list[Candidate]:
    """Wikimedia Commons files whose title or description match `query` (MediaWiki search
    generator in the File namespace, generator parameters carry the `g` prefix) + imageinfo."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query[:200],
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    return _commons_pages(json.loads(get(url, {"User-Agent": user_agent(contact)})))


def local_contact(config: Path = Path("worldreel.local.toml")) -> str:
    """`[retrieval] contact` from the git-ignored local config, or "" (network stays off)."""
    import tomllib

    if not config.is_file():
        return ""
    with config.open("rb") as fh:
        return str(tomllib.load(fh).get("retrieval", {}).get("contact", ""))


# ---- geometric verification ---------------------------------------------------------------


@dataclass(frozen=True)
class FrameView:
    image: U8
    r: F64  # OpenCV world-to-camera, OpenCV world axes
    t: F64
    k: F64


def _sift(image: U8, max_side: int = 800) -> tuple[list[cv2.KeyPoint], Any, float]:
    h, w = image.shape[:2]
    s = min(1.0, max_side / max(h, w))
    img = cv2.resize(image, (round(w * s), round(h * s))) if s < 1 else image
    kp, desc = cv2.SIFT.create(nfeatures=4000).detectAndCompute(
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None
    )
    return list(kp), desc, s


def _match(d1: Any, d2: Any, ratio: float = 0.75) -> list[tuple[int, int]]:
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    return [
        (m.queryIdx, m.trainIdx)
        for m, n in (p for p in pairs if len(p) == 2)
        if m.distance < ratio * n.distance
    ]


def depth_buffer(points_cv: F64, v: FrameView, splat_px: int = 2) -> npt.NDArray[np.float64]:
    """Nearest observed depth per pixel from points (OpenCV world axes), dilated by `splat_px`."""
    h, w = v.image.shape[:2]
    pc = points_cv @ v.r.T + v.t
    z = pc[:, 2]
    ok = z > 1e-3
    u = np.round(v.k[0, 0] * pc[ok, 0] / z[ok] + v.k[0, 2]).astype(np.int64)
    q = np.round(v.k[1, 1] * pc[ok, 1] / z[ok] + v.k[1, 2]).astype(np.int64)
    zz = z[ok]
    inside = (u >= 0) & (u < w) & (q >= 0) & (q < h)
    buf = np.full(h * w, np.inf)
    np.minimum.at(buf, q[inside] * w + u[inside], zz[inside])
    img: npt.NDArray[np.float32] = buf.reshape(h, w).astype(np.float32)
    img[~np.isfinite(img)] = 1e9
    eroded = cv2.erode(img, np.ones((2 * splat_px + 1, 2 * splat_px + 1), np.uint8))
    out = np.asarray(eroded, np.float64)
    out[out >= 1e8] = np.inf
    return out


def _off_plane_fraction(pts: F64, band: float, iterations: int = 300) -> float:
    """1 - share of points on the best RANSAC plane: ~0 when all matches lie on one surface."""
    if len(pts) < 3:
        return 0.0
    rng = np.random.default_rng(0)
    best = 0
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        best = max(best, int((np.abs((pts - a) @ n) < band).sum()))
    return 1.0 - best / len(pts)


def _photo_agreement(
    cand: U8, v: Verified, observed_cv: F64, colors: U8, tol: float = 40.0, max_points: int = 50000
) -> float:
    """Share of observed points visible from the estimated pose whose colour matches the photo
    after a per-channel gain (exposure) correction."""
    if len(observed_cv) > max_points:
        idx = np.random.default_rng(0).choice(len(observed_cv), max_points, replace=False)
        observed_cv, colors = observed_cv[idx], colors[idx]
    fv = FrameView(cand, v.r, v.t, v.k)
    depth = depth_buffer(observed_cv, fv)
    pc = observed_cv @ v.r.T + v.t
    z = pc[:, 2]
    h, w = cand.shape[:2]
    ok = z > 1e-3
    u = np.where(ok, np.round(v.k[0, 0] * pc[:, 0] / np.where(ok, z, 1) + v.k[0, 2]), -1).astype(
        np.int64
    )
    q = np.where(ok, np.round(v.k[1, 1] * pc[:, 1] / np.where(ok, z, 1) + v.k[1, 2]), -1).astype(
        np.int64
    )
    ok &= (u >= 0) & (u < w) & (q >= 0) & (q < h)
    ok[ok] = z[ok] <= depth[q[ok], u[ok]] * 1.05
    if ok.sum() < 50:
        return 0.0
    photo = cand[q[ok], u[ok]].astype(np.float64)
    obs = colors[ok].astype(np.float64)
    gain = np.median(photo, 0) / np.maximum(np.median(obs, 0), 1.0)
    err = np.abs(photo - obs * gain).max(axis=1)
    return float((err < tol).mean())


def verify(
    cand: U8,
    frames: list[FrameView],
    observed_world: F64,
    *,
    observed_colors: U8 | None = None,
    min_off_plane: float = 0.05,  # D-4.4: ground-texture-only matches had 0.0, true views >= 0.085
    min_photo_agree: float = 0.3,  # D-4.4: true views 0.43-0.63
    min_inliers: int = 40,
    min_ratio: float = 0.25,
    max_median_px: float = 2.0,
    min_second: int = 20,
) -> Verified:
    """Register a candidate image to the reconstruction. `observed_world`: observed points in
    WorldReel world axes. Returns the pose (OpenCV conventions) or the reason for rejection."""
    pts_cv = observed_world.astype(np.float64) @ _FLIP.T
    kc, dc, sc = _sift(cand)
    if dc is None or len(kc) < min_inliers:
        return Verified(False, "too few features in candidate")
    scored = []
    for i, fv in enumerate(frames):
        kf, df, sf = _sift(fv.image)
        m = _match(dc, df)
        scored.append((len(m), i, kf, sf, m))
    scored.sort(key=lambda x: -x[0])
    if not scored or scored[0][0] < min_inliers:
        return Verified(False, f"too few matches ({scored[0][0] if scored else 0})")

    def correspondences(
        entry: tuple[int, int, list[cv2.KeyPoint], float, list[tuple[int, int]]],
    ) -> tuple[F64, F64]:
        _, i, kf, sf, m = entry
        fv = frames[i]
        depth = depth_buffer(pts_cv, fv)
        h, w = depth.shape
        obj, img = [], []
        kinv = np.linalg.inv(fv.k)
        for a, b in m:
            u, q = kf[b].pt[0] / sf, kf[b].pt[1] / sf
            iu, iq = round(u), round(q)
            if not (0 <= iu < w and 0 <= iq < h) or not np.isfinite(depth[iq, iu]):
                continue
            x_cam = kinv @ np.array([u, q, 1.0]) * depth[iq, iu]
            obj.append(fv.r.T @ (x_cam - fv.t))
            img.append(np.array(kc[a].pt) / sc)
        return np.array(obj), np.array(img)

    obj, img = correspondences(scored[0])
    if len(obj) < min_inliers:
        return Verified(False, f"too few matches with depth ({len(obj)})")
    h, w = cand.shape[:2]
    best: Verified | None = None
    best_inlier_pts: F64 = np.zeros((0, 3))
    for f_scale in np.linspace(0.6, 1.6, 11):
        f = f_scale * w
        k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, k, None, reprojectionError=4.0, iterationsCount=2000, confidence=0.999
        )
        if not ok or inl is None:
            continue
        r, _ = cv2.Rodrigues(rvec)
        proj = (obj[inl[:, 0]] @ r.T + tvec.ravel()) @ k.T
        err = np.linalg.norm(proj[:, :2] / proj[:, 2:] - img[inl[:, 0]], axis=1)
        cand_v = Verified(
            True,
            "",
            np.asarray(r, np.float64),
            np.asarray(tvec, np.float64).ravel(),
            k,
            len(inl),
            len(inl) / len(obj),
            float(np.median(err)),
        )
        if best is None or cand_v.inliers > best.inliers:
            best = cand_v
            best_inlier_pts = obj[inl[:, 0]]
    if best is None:
        return Verified(False, "PnP failed")
    scale = float(np.percentile(np.linalg.norm(pts_cv - np.median(pts_cv, 0), axis=1), 95))
    best.off_plane_frac = _off_plane_fraction(best_inlier_pts, 0.02 * scale)
    if observed_colors is not None:
        best.photo_agree = _photo_agreement(cand, best, pts_cv, observed_colors)
    # Second-frame consistency: the same pose must explain matches through another frame.
    for entry in scored[1:3]:
        o2, i2 = correspondences(entry)
        if len(o2):
            p2 = (o2 @ best.r.T + best.t) @ best.k.T
            front = p2[:, 2] > 1e-6
            e2 = np.linalg.norm(p2[front, :2] / p2[front, 2:] - i2[front], axis=1)
            best.second_frame_inliers = max(best.second_frame_inliers, int((e2 < 4.0).sum()))
    if best.inliers < min_inliers:
        best.ok, best.reason = False, f"{best.inliers} PnP inliers < {min_inliers}"
    elif best.inlier_ratio < min_ratio:
        best.ok, best.reason = False, f"inlier ratio {best.inlier_ratio:.2f} < {min_ratio}"
    elif best.median_px > max_median_px:
        best.ok, best.reason = (
            False,
            f"median reprojection {best.median_px:.2f} px > {max_median_px}",
        )
    elif best.second_frame_inliers < min_second:
        best.ok, best.reason = False, f"second frame confirms only {best.second_frame_inliers}"
    elif best.off_plane_frac < min_off_plane:
        best.ok, best.reason = (
            False,
            f"matches lie on one surface ({best.off_plane_frac:.2f} off it)",
        )
    elif observed_colors is not None and best.photo_agree < min_photo_agree:
        best.ok, best.reason = False, f"colours disagree (agreement {best.photo_agree:.2f})"
    return best


def colours_from(
    cand: U8,
    v: Verified,
    positions_world: F64,
    occluders_world: F64,
    tol: float = 0.05,
    footprint: float = 0.0,
) -> tuple[npt.NDArray[np.bool_], U8]:
    """Colours of `positions_world` seen in the verified image (z-tested against occluders).

    With `footprint` (world size of the element, e.g. the fill cell), the image is averaged over
    the element's projected size: one element stands for a cell, and a single pixel would carry
    texture noise instead of the cell's colour (measured: 23 vs 15 mean |RGB| error, D-4.4)."""
    fv = FrameView(cand, v.r, v.t, v.k)
    depth = depth_buffer(occluders_world.astype(np.float64) @ _FLIP.T, fv)
    pc = (positions_world.astype(np.float64) @ _FLIP.T) @ v.r.T + v.t
    z = pc[:, 2]
    h, w = cand.shape[:2]
    ok = z > 1e-3
    u = np.full(len(z), -1)
    q = np.full(len(z), -1)
    u[ok] = np.round(v.k[0, 0] * pc[ok, 0] / z[ok] + v.k[0, 2]).astype(np.int64)
    q[ok] = np.round(v.k[1, 1] * pc[ok, 1] / z[ok] + v.k[1, 2]).astype(np.int64)
    ok &= (u >= 0) & (u < w) & (q >= 0) & (q < h)
    vis = ok.copy()
    vis[ok] = z[ok] <= depth[q[ok], u[ok]] * (1 + tol)
    colours = np.zeros((len(z), 3), np.uint8)
    if footprint <= 0:
        colours[vis] = cand[q[vis], u[vis]]
        return vis, colours
    radius_px = np.zeros(len(z))
    radius_px[vis] = 0.5 * v.k[0, 0] * footprint / z[vis]
    levels = [0, 1, 2, 4, 8, 16, 32]
    pick = np.searchsorted(levels, np.clip(radius_px, 0, levels[-1]))
    for li, rad in enumerate(levels):
        sel = vis & (np.minimum(pick, len(levels) - 1) == li)
        if not sel.any():
            continue
        img = cand if rad == 0 else cv2.blur(cand, (2 * rad + 1, 2 * rad + 1))
        colours[sel] = img[q[sel], u[sel]]
    return vis, colours


# ---- applying verified references to a splat world ------------------------------------------

RETRIEVED_CONFIDENCE = 0.5


def apply_references(
    candidates: list[Candidate],
    frames: list[FrameView],
    observed_world: F64,
    observed_colors: U8,
    targets_world: F64,
    target_size: float,
    *,
    log: Callable[[str], None] = print,
) -> tuple[npt.NDArray[np.bool_], U8, list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter, verify and use candidates. Returns (recoloured mask over targets, their colours,
    region records, credits). Candidates with people (metadata or a detected face) are dropped
    before verification; failed verifications are logged and never used."""
    recoloured = np.zeros(len(targets_world), bool)
    colours = np.zeros((len(targets_world), 3), np.uint8)
    regions: list[dict[str, Any]] = []
    credits: list[dict[str, Any]] = []
    for n, cand in enumerate(candidates):
        if cand.image is None:
            log(f"reference {cand.url}: no image data, skipped")
            continue
        if mentions_people(cand.metadata_text) or has_face(cand.image):
            log(f"reference {cand.url}: people / private-place filter, skipped")
            continue
        v = verify(cand.image, frames, observed_world, observed_colors=observed_colors)
        if not v.ok:
            log(f"reference {cand.url}: rejected ({v.reason})")
            continue
        vis, col = colours_from(cand.image, v, targets_world, observed_world, footprint=target_size)
        new = vis & ~recoloured
        colours[new] = col[new]
        recoloured |= new
        log(f"reference {cand.url}: verified ({v.inliers} inliers), recoloured {int(new.sum())}")
        if not new.any():
            continue
        lo, hi = targets_world[new].min(0), targets_world[new].max(0)
        source: dict[str, Any] = {
            "kind": "user" if cand.source == "user" else "retrieval",
            "url": cand.url,
            "license": cand.license,
            "verified": "geometric",
        }
        regions.append(
            {
                "id": f"r_reference_{n}",
                "bounds": {"min": [float(x) for x in lo], "max": [float(x) for x in hi]},
                "class": "retrieved",
                "confidence": RETRIEVED_CONFIDENCE,
                "sources": [source],
            }
        )
        credits.append(
            {
                "url": cand.url,
                "license": cand.license,
                "attribution": cand.attribution,
                "landing_url": cand.landing_url,
                "source": cand.source,
            }
        )
    return recoloured, colours, regions, credits
