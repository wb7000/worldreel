"""Quantitative QA for a generated clip: one face per frame, stable geometry,
sharpness, identity AND MOTION.

Replaces eyeballing extracted frames. Reports:
- detections per frame (a pasted/ghost second face shows up as 2 or as
  landmark jitter between frames)
- inter-ocular distance and face-height spread across the clip (a swap overlay
  makes the geometry pop between two populations)
- identity cosine per sampled frame against the reference portrait
- Laplacian variance (sharpness) inside the face oval
- motion: the mean frame-to-frame difference and the share of consecutive frames
  that barely change (D-9.26). The D-9.25 adapter render passed every other
  check here -- one face, stable geometry, identity 0.93 -- and was still a
  "still image that shakes": 211 of 239 consecutive frames differed by less than
  0.5 grey levels (the base model: 0 of 239). A clip that does not move is not a
  video, so motion belongs in this gate.

The measurements live in tools/video_metrics.py; this is the single-clip
front-end (tools/compare_videos.py is the multi-clip one).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.video_metrics import SAME_PERSON, STILL_DIFF, measure_clip


def _reference_portrait() -> Path:
    """Locate the reference portrait.

    Order: $WORLDREEL_REFERENCE_PORTRAIT, then the fetched vacecast copy under
    runs/, then the old temp path. The temp path alone was a trap: the temp
    directory gets wiped between sessions and the tool then failed with a
    confusing "missing reference" assert.
    """
    candidates = [
        os.environ.get("WORLDREEL_REFERENCE_PORTRAIT"),
        "runs/kaggle/portraits_run3_merged/C01_0.png",
        "C:/Users/li/AppData/Local/Temp/opencode/vacecast/C01_0.png",
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            return Path(cand)
    raise SystemExit(
        "reference portrait not found; set WORLDREEL_REFERENCE_PORTRAIT or fetch "
        "runs/kaggle/portraits_run3_merged/C01_0.png"
    )


def main() -> int:
    video = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ("runs/kaggle/character_consistency_v2/out/face_consistency.mp4")
    )
    m = measure_clip(video, _reference_portrait())

    print(f"video: {video}")
    print(f"frames={m.frames} size={m.width}x{m.height}")
    print(f"frames read          : {m.frames}")
    print(f"frames with 0 faces  : {m.faces_none}")
    print(f"frames with >1 face  : {m.faces_multi}  <-- a ghost/overlapping face shows here")
    if m.iod:
        print(
            f"inter-ocular px      : mean={m.iod_mean:.1f} sd={m.iod_sd:.1f} "
            f"cv={m.iod_cv:.3f} min={min(m.iod):.1f} max={max(m.iod):.1f}"
        )
        print(f"face height px       : mean={m.height_mean:.1f} cv={m.height_cv:.3f}")
    if m.sharpness:
        print(f"face sharpness (Lap) : mean={m.sharpness_mean:.1f} min={min(m.sharpness):.1f}")
    if m.identity:
        print(
            f"identity cosine      : mean={m.identity_mean:.4f} "
            f"min={m.identity_min:.4f} (SAME_PERSON threshold {SAME_PERSON})"
        )
    if m.motion:
        print(
            f"motion (frame diff)  : mean={m.motion_mean:.3f} median={m.motion_median:.3f} "
            f"max={max(m.motion):.3f}"
        )
        print(
            f"still frames (<{STILL_DIFF}) : {m.still_frames}/{len(m.motion)}  "
            f"<-- a shaking still image shows here (D-9.25: 211/239)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
