"""Reorder and trim the segments of a joined render into one timeline (D-9.32).

The five takes of a chained render are independent: their seeds carry no
narrative order, so the assembly order is free.  Placed naively (seed order)
the boundaries jump 0.06-0.22 in normalised landmark distance, the face
scale steps 4-6% and the wall light steps up to 6 grey levels -- the "cuts"
a viewer sees even through a cross-dissolve.  A full permutation search over
the pair jump matrix picks the order with the smallest total pose jump
(0.379 -> 0.283 on the D-9.31 footage), and dropping a few frames from each
side of every boundary lands the two faces on matched poses.

Unlike `join_clips.py` this is NOT a stream copy: reorder and frame-accurate
trims cannot be expressed as a concatenation of whole encoded clips, so the
timeline is decoded and written as lossless FFV1 (`.mkv`).  D-9.22/23 still
hold -- FFV1 round-trips bit-exact and only the final delivery encode is
lossy, once, at the end.

Usage:
    python tools/assemble_segments.py --video raw.mp4 --output asm.mkv \
        --bounds 48 96 144 192 --order 3 4 2 1 0 --trims 2:5 5:2 5:5 0:2

`--bounds` splits the input into segments; `--order` names the source
segment for each output position; `--trims` gives one `tail:head` pair per
boundary (frames dropped from the end of the left segment and the start of
the right one).  The tool prints the new cut bounds -- feed them to
`cut_smooth --bounds` so leveling and the dissolve use the real cuts instead
of landmark auto-detection (which misses them once the jumps are small).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_stabilize import _read_video, _write_video

U8 = npt.NDArray[np.uint8]


def assemble(
    frames: Sequence[U8],
    bounds: Sequence[int],
    order: Sequence[int],
    trims: Sequence[tuple[int, int]],
) -> tuple[list[U8], list[int]]:
    """Split at `bounds`, emit segments in `order`, drop `trims` at each boundary.

    `trims[i] = (tail, head)`: `tail` frames leave the end of output segment
    `i`, `head` frames leave the start of output segment `i + 1`.  Returns the
    new frames and the cut bounds in the output timeline.  Every output
    segment keeps at least one frame; anything else is a config error and
    raises rather than silently emitting an empty take.
    """
    n = len(frames)
    cuts = list(bounds)
    if cuts != sorted(set(cuts)) or any(c <= 0 or c >= n for c in cuts):
        raise ValueError(f"bounds must be strictly increasing in (0, {n}), got {cuts}")
    edges = [0, *cuts, n]
    n_seg = len(edges) - 1
    if sorted(order) != list(range(n_seg)):
        raise ValueError(f"order must be a permutation of 0..{n_seg - 1}, got {list(order)}")
    if len(trims) != n_seg - 1:
        raise ValueError(f"{n_seg} segments need {n_seg - 1} trim pairs, got {len(trims)}")
    if any(tail < 0 or head < 0 for tail, head in trims):
        raise ValueError(f"trims must be non-negative, got {list(trims)}")

    out: list[U8] = []
    new_cuts: list[int] = []
    for i, sid in enumerate(order):
        lo, hi = edges[sid], edges[sid + 1]
        start = lo + (trims[i - 1][1] if i > 0 else 0)
        stop = hi - (trims[i][0] if i < n_seg - 1 else 0)
        if stop <= start:
            raise ValueError(f"trims empty segment {sid}: keep [{start}, {stop}) of [{lo}, {hi})")
        if i > 0:
            new_cuts.append(len(out))
        out.extend(frames[start:stop])
    return out, new_cuts


def _parse_trims(specs: Sequence[str]) -> list[tuple[int, int]]:
    """Parse `tail:head` CLI entries into integer pairs."""
    trims: list[tuple[int, int]] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 2:
            raise SystemExit(f"--trims entries must be 'tail:head', got {spec!r}")
        try:
            trims.append((int(parts[0]), int(parts[1])))
        except ValueError:
            raise SystemExit(f"--trims entries must be integers, got {spec!r}") from None
    return trims


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reorder and trim render segments losslessly")
    parser.add_argument("--video", required=True, help="Joined raw timeline to split at --bounds")
    parser.add_argument("--output", required=True, help="Output .mkv (lossless FFV1)")
    parser.add_argument(
        "--bounds",
        type=int,
        nargs="+",
        required=True,
        help="Segment boundaries in the input (e.g. 48 96 144 192)",
    )
    parser.add_argument(
        "--order",
        type=int,
        nargs="+",
        required=True,
        help="Source segment index for each output position (a permutation)",
    )
    parser.add_argument(
        "--trims",
        nargs="*",
        default=[],
        metavar="TAIL:HEAD",
        help="Per-boundary frame drops, one 'tail:head' pair per boundary",
    )
    args = parser.parse_args(argv)

    trims = _parse_trims(args.trims)
    src_frames, fps, size = _read_video(args.video, None)
    out_frames, cuts = assemble(src_frames, args.bounds, args.order, trims)
    _write_video(args.output, out_frames, fps, size)

    edges = [0, *args.bounds, len(src_frames)]
    lengths = [
        (edges[sid + 1] - edges[sid])
        - (trims[i][0] if i < len(args.order) - 1 else 0)
        - (trims[i - 1][1] if i > 0 else 0)
        for i, sid in enumerate(args.order)
    ]
    print(f"assembled {len(src_frames)} -> {len(out_frames)} frames -> {args.output}")
    print(f"order: {args.order}  segment lengths: {lengths}")
    print(f"cut bounds for the next step: {' '.join(str(c) for c in cuts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
