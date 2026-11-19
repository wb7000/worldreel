"""Temporal face stabilization for generated clips (D-9.20).

Diagnosis of the 15 s S017 render -- the "character model changes a little
every frame" artifact has two independent causes:

1. **Jitter** ~1 px of real frame-to-frame rigid motion (ECC high-frequency
   component 1.0 px).  YuNet reports 1.6 px, but a known-shift test (translate
   a frame by an exact integer amount, re-detect) puts per-detection noise at
   1.5 px common-mode + ~2 px shape -- so a warp driven by the landmark
   trajectory *injects* detector noise as pixel motion.  Measured: placement
   alone made the fixed-box pixel metric worse (4.0 -> 6.6).
2. **Morph** ECC-aligned consecutive face crops differ by 3.1/255 at lag 1
   and the gap grows with lag (3.1 -> 16.1 over 20 frames): the generated
   face slowly changes shape/texture.  A local temporal window removes the
   white part of this; the drift part is partly real expression/lighting and
   must not be averaged away.

Design:

* Every transform comes from **ECC intensity alignment** against the take's
  middle frame (sub-pixel, detector-free).  One absolute affine W_t per frame
  yields both the jitter trajectory (low-passed in time -> S_t) and the
  pairwise window alignment (W_i o W_k^-1) at no extra cost.
* Placement moves the face oval from W_t onto S_t (jitter removed); YuNet
  landmarks only decide where the take boundaries are and how big the oval is.
* The window frames are blended with Gaussian time weights gated by
  motion-aligned appearance distance (a blink weighs ~0), then the feathered
  oval is composited back: background, hair silhouette and body keep their
  original pixels.
* The trajectory is low-passed per take (`split_at_cuts`), so a cut stays a
  cut instead of being smeared into a pre-bend before it.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from stages.generate.face_swap import FaceSwapper

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"
ECC_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
ECC_MIN_CC = 0.75  # below this ECC lost the face; fall back to the landmark init
# det bounds for accepting the ECC affine.  Measured on the 15 s render:
# det spans 0.71-1.65 and correlates +0.60 with the landmark eye distance,
# i.e. it tracks real face scale (head approaching the camera), so the guard
# must only catch a genuine collapse/blow-up.  The old [0.7, 1.4] rejected 18
# legitimate frames; their fallback was a *glitched* landmark init, which is
# what produced the 35 px placement spike.
ECC_MIN_DET = 0.5
ECC_MAX_DET = 2.0


@dataclass(frozen=True)
class Stats:
    """What the run did, for the CLI to report."""

    takes: list[tuple[int, int]]
    moved_mean: float  # px, translation of the placement correction
    moved_max: float
    cc_mean: float  # mean ECC correlation over all frames
    ecc_fallbacks: int  # frames where ECC was rejected and the init used


def detect_landmarks(frames: Sequence[U8], swapper: FaceSwapper) -> tuple[F64, int]:
    """YuNet's five landmarks per frame, shape (n, 5, 2), plus the gap count.

    A missing detection is linearly interpolated across the gap: a run of
    blanks must not break the trajectory (the 15 s render had 240/240, but a
    chin clipped by the frame edge is enough to lose one frame).
    """
    n = len(frames)
    out = np.full((n, 5, 2), np.nan, dtype=np.float64)
    for i, frame in enumerate(frames):
        face = swapper.detect_face(frame)
        if face is not None:
            out[i] = face[1].astype(np.float64)
    valid = np.isfinite(out[:, 0, 0])
    if valid.all():
        return out, 0
    if not valid.any():
        raise ValueError("no face detected in any frame")
    gaps = int(np.count_nonzero(~valid))
    idx = np.arange(n)
    for k in range(5):
        for c in range(2):
            out[:, k, c] = np.interp(idx, idx[valid], out[valid, k, c])
    return out, gaps


def split_at_cuts(lm: F64, threshold: float = 12.0) -> list[tuple[int, int]]:
    """Split the trajectory where the raw landmarks jump (a take boundary).

    The low-pass must only see motion inside one take: smoothing across a cut
    would drag both takes toward each other in the frames near the boundary
    (v12: cut jumps 21-47 px against 2.8 px mean within-take motion, so 12 px
    separates them cleanly).
    """
    if lm.shape[0] == 0:
        return []
    step = np.linalg.norm(np.diff(lm, axis=0), axis=-1).mean(axis=1)
    cuts = (np.flatnonzero(step > threshold) + 1).tolist()
    bounds = [0, *cuts, lm.shape[0]]
    return list(itertools.pairwise(bounds))


def lowpass(lm: F64, sigma: float) -> F64:
    """Gaussian smoothing along axis 0 (frames), any trailing shape.

    Edges are padded by repetition so the first/last frames keep their
    position instead of being pulled toward the clip centre.
    """
    if sigma <= 0:
        return lm.copy()
    radius = max(1, int(np.ceil(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate(
        [np.repeat(lm[:1], radius, axis=0), lm, np.repeat(lm[-1:], radius, axis=0)],
        axis=0,
    )
    out = np.empty_like(lm)
    for i in range(lm.shape[0]):
        out[i] = np.tensordot(kernel, padded[i : i + 2 * radius + 1], axes=(0, 0))
    return out


def _similarity(src: F64, dst: F64) -> F64:
    """Least-squares similarity (rotation + scale + translation) src -> dst.

    Complex Procrustes: no reflection, exact for a true similarity.
    Returns a 2x3 matrix for cv2.warpAffine.
    """
    s = src - src.mean(axis=0)
    d = dst - dst.mean(axis=0)
    denom = float(np.sum(s**2))
    if denom < 1e-12:
        return np.hstack([np.eye(2), np.zeros((2, 1))])
    num = complex(
        float(np.sum(s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1])),
        float(np.sum(s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0])),
    )
    c = num / denom
    rot = np.array([[c.real, -c.imag], [c.imag, c.real]], dtype=np.float64)
    trans = dst.mean(axis=0) - rot @ src.mean(axis=0)
    return np.hstack([rot, trans[:, None]])


def _local_from_full(m: F64, origin: tuple[int, int]) -> F64:
    """Express a full-frame affine in crop-local coordinates."""
    o = np.array([origin[0], origin[1]], dtype=np.float64)
    out = m.copy()
    out[:, 2] = m[:, :2] @ o + m[:, 2] - o
    return out


def _full_from_local(m: F64, origin: tuple[int, int]) -> F64:
    """The inverse of `_local_from_full`."""
    o = np.array([origin[0], origin[1]], dtype=np.float64)
    out = m.copy()
    out[:, 2] = m[:, 2] - m[:, :2] @ o + o
    return out


def _homog(m: F64) -> F64:
    out = np.eye(3, dtype=np.float64)
    out[:2] = m
    return out


def _ecc_affine(
    template: U8,
    inp: U8,
    init: F64,
    det_bounds: tuple[float, float] = (ECC_MIN_DET, ECC_MAX_DET),
) -> tuple[F64, float, bool]:
    """ECC affine mapping template coordinates -> input coordinates.

    Returns (affine, correlation, ecc_used).  On failure or a implausible
    result (lost face, collapsed scale) the landmark initialisation is kept:
    ECC only has to *refine* the init, never rescue it.  `det_bounds` is the
    plausibility window on det (= scale squared); the cross-take dissolve
    passes a wider one, because two takes of the same chain legitimately
    differ by up to 1.6x in face size where within-take frames never do.
    """
    warp = init.astype(np.float32)
    try:
        cc, found = cv2.findTransformECC(template, inp, warp, cv2.MOTION_AFFINE, ECC_CRIT)
    except cv2.error:
        return init, -1.0, False
    if not np.isfinite(cc):
        return init, -1.0, False
    out = np.asarray(found, dtype=np.float64)
    det = float(np.linalg.det(out[:, :2]))
    if not (cc >= ECC_MIN_CC) or not (det_bounds[0] <= det <= det_bounds[1]):
        return init, float(cc), False
    return out, float(cc), True


def _oval_mask(shape: tuple[int, int], cfg: F64) -> F32:
    """Feathered ellipse covering the face, built from a landmark config.

    Semi-axes are 1.6x the landmark span: the landmark box (eyes to mouth
    corners) is about half the visible face, so the ellipse clears the hair
    line and chin with room for the feather.
    """
    h, w = shape
    cx = float(cfg[:, 0].mean())
    cy = float(cfg[:, 1].mean())
    span_x = max(float(np.ptp(cfg[:, 0])), 8.0)
    span_y = max(float(np.ptp(cfg[:, 1])), 8.0)
    canvas = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        canvas,
        (round(cx), round(cy)),
        (round(1.6 * span_x), round(1.5 * span_y)),
        0.0,
        0.0,
        360.0,
        1.0,
        -1,
    )
    sigma = max(2.0, 0.10 * 1.6 * span_x)
    return np.clip(cv2.GaussianBlur(canvas, (0, 0), sigma), 0.0, 1.0)


def _face_box(cfg: F64, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Crop window around the reference face, padded for in-take pose drift."""
    h, w = shape
    span_x = max(float(np.ptp(cfg[:, 0])), 8.0)
    span_y = max(float(np.ptp(cfg[:, 1])), 8.0)
    x0 = int(max(0, cfg[:, 0].min() - 0.8 * span_x))
    x1 = int(min(w, cfg[:, 0].max() + 0.8 * span_x))
    y0 = int(max(0, cfg[:, 1].min() - 1.0 * span_y))
    y1 = int(min(h, cfg[:, 1].max() + 1.0 * span_y))
    return x0, y0, max(x1, x0 + 8), max(y1, y0 + 8)


def _stabilize_take(
    out: list[U8],
    frames: Sequence[U8],
    lm: F64,
    lo: int,
    hi: int,
    sigma: float,
    window_sigma: float,
    gate_h: float,
    detail: float,
) -> tuple[float, float, float, int]:
    """One take: ECC trajectory, jitter placement, gated temporal blend.

    `detail` reinjects the centre frame's own high frequencies after the
    blend (which averages +-window_sigma frames and therefore low-passes
    the face); the residual is computed from the *placed* centre frame so
    its edges land on the blended edges instead of doubling them.

    Returns (moved_mean, moved_max, cc_mean, ecc_fallbacks).
    """
    h, w = frames[lo].shape[:2]
    n = hi - lo
    ref_i = lo + n // 2
    ref_lm = lm[ref_i]
    x0, y0, x1, y1 = _face_box(ref_lm, (h, w))
    origin = (x0, y0)
    gref = cv2.cvtColor(frames[ref_i][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(
        np.uint8, copy=False
    )

    # 1. absolute trajectory: every frame ECC-aligned to the take's middle
    #    frame, initialised from the landmark similarity (so ECC only refines).
    w_abs = np.empty((n, 2, 3), dtype=np.float64)
    ccs = np.empty(n, dtype=np.float64)
    fallbacks = 0
    for j, t in enumerate(range(lo, hi)):
        init = _local_from_full(_similarity(ref_lm, lm[t]), origin)
        gt = cv2.cvtColor(frames[t][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)
        warp, cc, used = _ecc_affine(gref, gt, init)
        w_abs[j] = _full_from_local(warp, origin)
        ccs[j] = cc
        fallbacks += int(not used)

    # 2. jitter is the high-frequency part of that trajectory
    smooth = lowpass(w_abs.reshape(n, 6), sigma).reshape(n, 2, 3)
    moved = np.linalg.norm(smooth[:, :, 2] - w_abs[:, :, 2], axis=1)

    # 3. blend in raw space (window alignment W_i o W_k^-1), then place the
    #    averaged face onto the smoothed trajectory and composite the oval.
    radius = max(1, round(2.0 * window_sigma))
    for j, t in enumerate(range(lo, hi)):
        final_cfg = ref_lm @ smooth[j, :, :2].T + smooth[j, :, 2]
        oval = _oval_mask((h, w), final_cfg)
        inside = oval > 0.5
        center_gray = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY).astype(np.float32)

        acc = np.zeros((h, w, 3), dtype=np.float32)
        weight_sum = 0.0
        for k in range(max(0, j - radius), min(n, j + radius + 1)):
            weight = float(np.exp(-0.5 * ((k - j) / window_sigma) ** 2))
            if k == j:
                aligned = frames[t]
            else:
                c_ik = _homog(w_abs[j]) @ np.linalg.inv(_homog(w_abs[k]))
                aligned = cv2.warpAffine(
                    frames[lo + k],
                    c_ik[:2].astype(np.float32),
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                ).astype(np.uint8, copy=False)
                if gate_h > 0:
                    # Motion already removed by the alignment, so the distance
                    # is pure appearance change: a blink weighs ~0 instead of
                    # ghosting through the average.
                    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    dist = float(np.mean(np.abs(aligned_gray[inside] - center_gray[inside])))
                    weight *= float(np.exp(-((dist / gate_h) ** 2)))
            if weight < 1e-4:
                continue
            acc += weight * np.asarray(aligned, dtype=np.float32)
            weight_sum += weight
        blended = acc / max(weight_sum, 1e-6)

        c_place = _homog(smooth[j]) @ np.linalg.inv(_homog(w_abs[j]))
        placed = cv2.warpAffine(
            blended,
            c_place[:2].astype(np.float32),
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if detail > 0.0:
            # High frequencies the blend averaged away, taken from this
            # frame after the *same* placement warp so the edges coincide
            # with the blended edges (adding the unwarped residual would
            # double them by the 1-7 px placement offset).
            center = cv2.warpAffine(
                frames[t],
                c_place[:2].astype(np.float32),
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            ).astype(np.float32)
            resid = center - np.asarray(cv2.GaussianBlur(center, (0, 0), 1.0), dtype=np.float32)
            placed = placed + detail * resid
        mixed = frames[t].astype(np.float32) * (1.0 - oval[..., None]) + placed * oval[..., None]
        # Round, do not truncate: truncation darkens the blended oval by 1 grey
        # level whenever the float lands just below an integer.
        out[t] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)

    return float(moved.mean()), float(moved.max()), float(ccs.mean()), fallbacks


def _stabilize_all(
    frames: Sequence[U8],
    lm: F64,
    sigma: float,
    window_sigma: float,
    gate_h: float,
    detail: float = 0.3,
) -> tuple[list[U8], Stats]:
    if len(frames) != lm.shape[0]:
        raise ValueError(f"{len(frames)} frames but {lm.shape[0]} landmark rows")
    segments = split_at_cuts(lm)
    out = [frame.copy() for frame in frames]
    moved_parts: list[float] = []
    moved_max = 0.0
    cc_parts: list[float] = []
    fallbacks = 0
    for lo, hi in segments:
        m_mean, m_max, cc_mean, fb = _stabilize_take(
            out, frames, lm, lo, hi, sigma, window_sigma, gate_h, detail
        )
        moved_parts.append(m_mean)
        moved_max = max(moved_max, m_max)
        cc_parts.append(cc_mean)
        fallbacks += fb
    stats = Stats(
        takes=segments,
        moved_mean=float(np.mean(moved_parts)) if moved_parts else 0.0,
        moved_max=moved_max,
        cc_mean=float(np.mean(cc_parts)) if cc_parts else 0.0,
        ecc_fallbacks=fallbacks,
    )
    return out, stats


def stabilize(
    frames: Sequence[U8],
    lm: F64,
    sigma: float = 2.0,
    window_sigma: float = 1.5,
    gate_h: float = 12.0,
    detail: float = 0.3,
) -> list[U8]:
    """Remove head jitter and average away the per-frame face morph.

    Every frame's face is ECC-aligned to the take's middle frame, placed on
    the low-pass trajectory (sigma, per take) and the face oval is replaced
    by a time-blended, motion-aligned average of the surrounding frames
    (window_sigma, appearance-gated by gate_h).  `detail` puts back the
    centre frame's own high frequencies, which the blend low-passes away.
    """
    return _stabilize_all(frames, lm, sigma, window_sigma, gate_h, detail)[0]


def _read_video(path: str, max_frames: int | None) -> tuple[list[U8], float, tuple[int, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 16.0
    frames: list[U8] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and len(frames) >= max_frames:
            break
        frames.append(np.asarray(frame, dtype=np.uint8))
    cap.release()
    if not frames:
        raise ValueError(f"no frames read from {path}")
    h, w = frames[0].shape[:2]
    return frames, fps, (w, h)


def _write_video(path: str, frames: Sequence[U8], fps: float, size: tuple[int, int]) -> None:
    """Write frames, lossless for `.mkv` (FFV1) and `mp4v` otherwise.

    The stabilize -> cut-smooth chain must not pass through lossy
    intermediates: one mp4v round-trip costs ~20 points of face Laplacian
    variance (66 -> 46 measured on the v12 render), two of them leave the
    face at 27.  FFV1 in matroska round-trips bit-exact and is the format
    for every intermediate; only the delivery encode is lossy, once, at
    the end.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = "FFV1" if path.endswith(".mkv") else "mp4v"
    writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*fourcc), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stabilize face geometry in a generated clip")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Temporal low-pass width in frames (0 = off); 2 keeps head turns, kills jitter",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=1.5,
        help="Averaging window in frames; larger = smoother but softer",
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=12.0,
        help="Appearance gate in grey levels; 0 disables it (never do this: blinks ghost)",
    )
    parser.add_argument(
        "--detail",
        type=float,
        default=0.3,
        help="High-frequency reinjection after the blend (0 = off); restores blend softness",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    src_frames, src_fps, src_size = _read_video(args.video, args.max_frames)
    swapper = FaceSwapper(YUNET, SFACE)
    landmarks, gaps = detect_landmarks(src_frames, swapper)
    out_frames, stats = _stabilize_all(
        src_frames, landmarks, args.sigma, args.window, args.gate, args.detail
    )
    _write_video(args.output, out_frames, src_fps, src_size)

    print(f"stabilized {len(src_frames)} frames -> {args.output}")
    print(f"takes detected: {len(stats.takes)} at frames {[hi for _, hi in stats.takes][:-1]}")
    print(f"landmark gaps interpolated: {gaps}")
    print(f"placement correction: mean={stats.moved_mean:.2f}px max={stats.moved_max:.2f}px")
    print(
        f"ECC trajectory: mean cc={stats.cc_mean:.3f}, "
        f"fallbacks={stats.ecc_fallbacks}/{len(src_frames)}"
    )
