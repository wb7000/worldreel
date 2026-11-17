"""Soften the hard cuts of a chained render (D-9.21).

The five 3 s takes stitched at frames 48/96/144/192 show three jump types at
a cut: an exposure step in the wall light (up to 12 grey levels against 0.2
frame-to-frame inside a take), a content/pose jump (the head teleports
21-47 px against 2.8 px of normal motion), and the seam itself.  Two steps,
in order:

1. **Exposure match** -- one scalar gain per take, from the mean of the side
   columns (wall, mostly subject-free) to the clip's mean wall.  Scalar on
   purpose: deflicker's fitted light field costs 0.035 of identity minimum
   on this clip (0.784 -> 0.748), a uniform gain costs 0.009.
2. **Cross-dissolve** -- `k` output frames replace the last `k` frames of
   the outgoing take and the first `k` of the incoming take, so the clip
   loses `k` frames per cut.  The incoming face is first ECC-registered
   onto the outgoing frame (landmark similarity alone aligns five points,
   not face shape: a similarity-only dissolve ghosted into identity 0.666),
   then both faces are carried onto one interpolating landmark path
   (outgoing config -> incoming config, smoothstep eased) and composited
   through the feathered oval before blending: the head travels in place
   instead of double-imaging, while the backgrounds blend raw.  The path
   starts exactly on the outgoing face and ends exactly on the incoming
   one, so neither window edge introduces a new jump.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from stages.generate.face_stabilize import (
    SFACE,
    YUNET,
    _ecc_affine,
    _face_box,
    _full_from_local,
    _local_from_full,
    _oval_mask,
    _read_video,
    _similarity,
    _write_video,
    detect_landmarks,
    split_at_cuts,
)
from stages.generate.face_swap import FaceSwapper

U8 = npt.NDArray[np.uint8]
F64 = npt.NDArray[np.float64]
WALL_COLS = 0.25  # share of the width on each side, treated as wall
# Plausibility window for the cross-take registration (det = scale^2):
# chain takes legitimately differ by ~1.6x in face size (det ~2.6 measured
# at cut@144), where within-take stabilizer frames stay inside 0.71-1.65.
DISSOLVE_DET = (0.3, 3.0)


@dataclass(frozen=True)
class CutStats:
    """What the run did, for the CLI to report."""

    takes: list[tuple[int, int]]
    cuts: list[int]
    gains: list[float]  # per-take scalar exposure gain
    dissolve: int  # frames consumed per side of each cut
    dissolve_out: list[int]  # first output index of each dissolve window
    frames_out: int


def _wall_mask(shape: tuple[int, int]) -> npt.NDArray[np.bool_]:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[:, : int(WALL_COLS * w)] = True
    mask[:, w - int(WALL_COLS * w) :] = True
    return mask


def match_levels(
    frames: Sequence[U8], takes: Sequence[tuple[int, int]]
) -> tuple[list[U8], list[float]]:
    """One scalar gain per take so the wall level matches over the clip.

    A *scalar* gain leaves face contrast and therefore identity untouched;
    a fitted spatial field (deflicker) does not -- measured cost on this
    clip was 0.035 identity minimum, which is the gate the render exists to
    pass.  Wall means are used instead of global means because the global
    mean moves when the subject moves between takes (content, not exposure).
    """
    if not takes:
        return list(frames), []
    h, w = frames[0].shape[:2]
    wall = _wall_mask((h, w))
    means = []
    for lo, hi in takes:
        acc = 0.0
        for t in range(lo, hi):
            acc += float(cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)[wall].mean())
        means.append(acc / max(hi - lo, 1))
    target = float(np.mean(means))
    gains = [target / m if m > 1e-6 else 1.0 for m in means]
    out: list[U8] = []
    for (lo, hi), gain in zip(takes, gains, strict=True):
        for t in range(lo, hi):
            scaled = np.round(frames[t].astype(np.float32) * gain)
            out.append(np.clip(scaled, 0, 255).astype(np.uint8))
    return out, gains


def _face_warp(frame: U8, cfg_src: F64, cfg_dst: F64) -> U8:
    """Warp `frame` so its face config lands on `cfg_dst`, oval only.

    Only the feathered face oval takes the warped pixels; hair edges,
    background and body keep their original pixels, so the warp cannot drag
    the scene with it.
    """
    h, w = frame.shape[:2]
    m = _similarity(cfg_src, cfg_dst)
    warped = cv2.warpAffine(
        frame,
        m.astype(np.float32),
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.uint8, copy=False)
    oval = _oval_mask((h, w), cfg_dst)[..., None]
    mixed = frame.astype(np.float32) * (1.0 - oval) + warped.astype(np.float32) * oval
    blended: U8 = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return blended


def _incoming_face(a_frame: U8, b_frame: U8, cfg_a: F64, cfg_b: F64, cfg_dst: F64) -> U8:
    """The incoming take's face, ECC-registered onto the outgoing frame first.

    Landmark similarity aligns five points but not face *shape*: with a
    similarity-only dissolve the two renders double-image through the middle
    of the window and identity fell to 0.666 (SFace).  ECC locks the actual
    pixels with a full affine (the landmark fit only initialises it, and the
    guards from the stabilizer fall back to it), after which both
    contributions of every blend pair occupy the same place with sub-pixel
    registration.  A single rigid travel warp then carries the registered
    face onto the interpolating path, so continuity at both window edges is
    unchanged.
    """
    h, w = a_frame.shape[:2]
    x0, y0, x1, y1 = _face_box(np.vstack([cfg_a, cfg_b]), (h, w))
    origin = (x0, y0)
    g_a = cv2.cvtColor(a_frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)
    g_b = cv2.cvtColor(b_frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)
    init = _local_from_full(_similarity(cfg_a, cfg_b), origin)
    w_loc, _cc, _used = _ecc_affine(g_a, g_b, init, DISSOLVE_DET)
    w_full = _full_from_local(w_loc, origin)
    registered = cv2.warpAffine(
        b_frame,
        w_full.astype(np.float32),
        (w, h),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.uint8, copy=False)
    travel = _similarity(cfg_a, cfg_dst)
    moved = cv2.warpAffine(
        registered,
        travel.astype(np.float32),
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.uint8, copy=False)
    oval = _oval_mask((h, w), cfg_dst)[..., None]
    mixed = b_frame.astype(np.float32) * (1.0 - oval) + moved.astype(np.float32) * oval
    blended: U8 = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return blended


def cross_dissolve(
    frames: Sequence[U8], lm: F64, cuts: Sequence[int], k: int = 6
) -> tuple[list[U8], list[int]]:
    """Cross-dissolve `k` frames around every cut, faces travelling in place.

    Returns the new frames and, for each cut, the first output index of its
    dissolve window.  Each window consumes `k` frames from either side of
    the cut and emits `k` frames, so the clip loses `k` frames per cut.
    """
    n = len(frames)
    if lm.shape[0] != n:
        raise ValueError(f"{n} frames but {lm.shape[0]} landmark rows")
    if k < 1:
        raise ValueError(f"dissolve must be >= 1 frame, got {k}")
    prev = 0
    for c in cuts:
        if c - k < prev or c + k > n:
            raise ValueError(
                f"cut at {c} with dissolve {k} overlaps a neighbour or the "
                f"clip edge (valid range {prev + k}..{n - k})"
            )
        prev = c + k

    out: list[U8] = []
    starts: list[int] = []
    pos = 0
    for c in cuts:
        out.extend(frames[pos : c - k])
        start = len(out)
        for i in range(k):
            t = 0.5 if k == 1 else i / (k - 1)
            ease = t * t * (3.0 - 2.0 * t)  # starts and ends at zero velocity
            a_idx, b_idx = c - k + i, c + i
            target = (1.0 - ease) * lm[a_idx] + ease * lm[b_idx]
            oa = _face_warp(frames[a_idx], lm[a_idx], target)
            ob = _incoming_face(frames[a_idx], frames[b_idx], lm[a_idx], lm[b_idx], target)
            alpha = (i + 1) / (k + 1)
            mixed = (1.0 - alpha) * oa.astype(np.float32) + alpha * ob.astype(np.float32)
            out.append(np.clip(np.round(mixed), 0, 255).astype(np.uint8))
        starts.append(start)
        pos = c + k
    out.extend(frames[pos:])
    return out, starts


def smooth_cuts(
    frames: Sequence[U8],
    lm: F64,
    dissolve: int = 6,
    levels: bool = True,
) -> tuple[list[U8], CutStats]:
    """Exposure-match the takes, then cross-dissolve every detected cut."""
    takes = split_at_cuts(lm)
    cuts = [lo for lo, _ in takes[1:]]
    if levels:
        leveled, gains = match_levels(frames, takes)
    else:
        leveled, gains = list(frames), [1.0] * len(takes)
    out, starts = cross_dissolve(leveled, lm, cuts, dissolve)
    stats = CutStats(
        takes=takes,
        cuts=cuts,
        gains=gains,
        dissolve=dissolve,
        dissolve_out=starts,
        frames_out=len(out),
    )
    return out, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soften the hard cuts of a generated clip")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument(
        "--dissolve",
        type=int,
        default=6,
        help="Frames per side of each cut (0 = no dissolve); 6 at 16 fps = 0.375 s",
    )
    parser.add_argument("--no-levels", action="store_true", help="Skip the per-take exposure match")
    parser.add_argument(
        "--bounds",
        type=int,
        nargs="*",
        default=None,
        help="Cut frames (default: detected from landmark jumps)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    src_frames, src_fps, src_size = _read_video(args.video, args.max_frames)
    swapper = FaceSwapper(YUNET, SFACE)
    landmarks, gaps = detect_landmarks(src_frames, swapper)
    if args.bounds is not None:
        bounds = sorted(args.bounds)
        takes = list(itertools.pairwise([0, *bounds, len(src_frames)]))
        cuts = bounds
    else:
        takes = split_at_cuts(landmarks)
        cuts = [lo for lo, _ in takes[1:]]
    if args.no_levels:
        leveled, gains = list(src_frames), [1.0] * len(takes)
    else:
        leveled, gains = match_levels(src_frames, takes)
    if args.dissolve > 0:
        out_frames, starts = cross_dissolve(leveled, landmarks, cuts, args.dissolve)
    else:
        out_frames, starts = leveled, []
    stats = CutStats(
        takes=takes,
        cuts=cuts,
        gains=gains,
        dissolve=args.dissolve,
        dissolve_out=starts,
        frames_out=len(out_frames),
    )
    _write_video(args.output, out_frames, src_fps, src_size)

    print(f"cut-smoothed {len(src_frames)} -> {len(out_frames)} frames -> {args.output}")
    print(f"landmark gaps interpolated: {gaps}")
    print(f"takes: {len(stats.takes)} at source cuts {stats.cuts}")
    print(f"per-take exposure gains: {[f'{g:.4f}' for g in stats.gains]}")
    print(
        f"dissolve: {stats.dissolve} frames/cut at output {stats.dissolve_out}"
        if stats.dissolve
        else "dissolve: off"
    )
