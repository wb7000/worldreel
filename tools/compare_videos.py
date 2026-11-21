"""Compare generated clips on every axis a viewer can see (D-9.26).

The A/B that decides which render ships. Every column is a gate the D-9.25
adapter render failed while still passing every check that existed before it:
its motion was 0.414 grey levels per frame with 211 of 239 consecutive frames
"still" -- a shaking image that passed every identity and geometry check.

Usage:
    python -m tools.compare_videos VIDEO [VIDEO ...]
    python -m tools.compare_videos --portrait PORTRAIT VIDEO [VIDEO ...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.video_metrics import ClipMetrics, measure_clip


def reference_portrait(explicit: str | None) -> Path:
    """Locate the reference portrait (same order as tools/v2_check.py)."""
    candidates = [
        explicit,
        os.environ.get("WORLDREEL_REFERENCE_PORTRAIT"),
        "runs/kaggle/portraits_run3_merged/C01_0.png",
        "C:/Users/li/AppData/Local/Temp/opencode/vacecast/C01_0.png",
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            return Path(cand)
    raise SystemExit(
        "reference portrait not found; pass --portrait or set WORLDREEL_REFERENCE_PORTRAIT"
    )


def row(name: str, m: ClipMetrics) -> str:
    return (
        f"{name:<28} {m.frames:>5}  {m.identity_mean:>7.4f} {m.identity_min:>7.4f} "
        f"{m.identity_below:>5}  {m.sharpness_mean:>8.1f}  {m.motion_mean:>7.3f} "
        f"{m.still_frames:>5}  {m.iod_cv:>6.3f}  {m.faces_multi:>5}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare generated clips on the QA axes")
    parser.add_argument("videos", nargs="+", help="clips to compare")
    parser.add_argument("--portrait", default=None, help="reference portrait")
    parser.add_argument(
        "--identity-stride",
        type=int,
        default=15,
        help="score the identity every Nth frame (15 = the QA default; 1 = every frame)",
    )
    args = parser.parse_args()

    portrait = reference_portrait(args.portrait)
    metrics = [measure_clip(v, portrait, identity_stride=args.identity_stride) for v in args.videos]

    header = (
        f"{'clip':<28} {'frames':>5}  {'id.mean':>7} {'id.min':>7} {'id<.80':>5}  "
        f"{'sharp':>8}  {'motion':>7} {'still':>5}  {'iodcv':>6} {'multi':>5}"
    )
    print(f"reference portrait: {portrait}")
    print(header)
    print("-" * len(header))
    for video, m in zip(args.videos, metrics, strict=True):
        print(row(video, m))
    print()
    for video, m in zip(args.videos, metrics, strict=True):
        flags = []
        if m.is_static:
            flags.append("STATIC (a shaking still image is not a video)")
        if m.faces_multi:
            flags.append(f"{m.faces_multi} frames with more than one face")
        if m.faces_none:
            flags.append(f"{m.faces_none} frames without a face")
        if m.identity_min < 0.363:
            flags.append(f"identity below the same-person threshold ({m.identity_min:.4f})")
        print(f"{video}: {', '.join(flags) if flags else 'no issues'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
