"""Quantitative QA for the generated clips: one face per frame, stable geometry,
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
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_align import to_uint8
from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"

# A frame-to-frame difference below this (grey levels) counts as "not moving".
STILL_DIFF = 0.5


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


REF = _reference_portrait()


def main() -> int:
    video = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ("runs/kaggle/character_consistency_v2/out/face_consistency.mp4")
    )
    sw = FaceSwapper(YUNET, SFACE)
    loaded = cv2.imread(str(REF))
    assert loaded is not None, f"missing reference {REF}"
    ref_emb = sw.compute_embedding(to_uint8(loaded))
    assert ref_emb is not None

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"cannot open {video}")
        return 1

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    print(f"video: {video}")
    print(f"frames={n} size={w}x{h}")

    counts: list[int] = []
    iods: list[float] = []
    heights: list[float] = []
    sims: list[float] = []
    sharp: list[float] = []
    diffs: list[float] = []
    prev: np.ndarray | None = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = to_uint8(frame)
        if prev is not None:
            diffs.append(float(np.abs(prev.astype(np.float32) - frame.astype(np.float32)).mean()))
        prev = frame
        # Ask YuNet for up to 2 faces so a ghost overlay is not silently merged
        # into one detection.
        # Ask for every detection (not just the most confident one) so a ghost
        # second face is counted instead of silently merged away.
        det = sw._detector(frame).detect(frame)[1]
        rows = [r for r in det] if det is not None else []
        count = len(rows)
        counts.append(count)

        faces = []
        for r in rows:
            lm = np.asarray(r[4:14], dtype=np.float32).reshape(5, 2)
            bbox = np.asarray(r[0:4], dtype=np.float32)
            faces.append((bbox, lm))
        if faces:
            bbox, lm = faces[0]
            iod_px = float(np.linalg.norm(lm[0] - lm[1]))
            iods.append(iod_px)
            heights.append(float(bbox[3]))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x0, y0 = max(0, int(bbox[0])), max(0, int(bbox[1]))
            x1, y1 = min(w, int(bbox[0] + bbox[2])), min(h, int(bbox[1] + bbox[3]))
            roi = gray[y0:y1, x0:x1]
            if roi.size:
                sharp.append(float(cv2.Laplacian(roi, cv2.CV_64F).var()))
            if i % 15 == 0:
                emb = sw.compute_embedding(frame)
                sims.append(float(emb @ ref_emb) if emb is not None else float("nan"))
        i += 1
    cap.release()

    if not counts:
        print("no frames read")
        return 1

    multi = sum(1 for c in counts if c > 1)
    none = sum(1 for c in counts if c == 0)
    iod = np.asarray(iods, dtype=np.float64)
    fh = np.asarray(heights, dtype=np.float64)
    print(f"frames read          : {len(counts)}")
    print(f"frames with 0 faces  : {none}")
    print(f"frames with >1 face  : {multi}  <-- a ghost/overlapping face shows here")
    if iod.size:
        print(
            f"inter-ocular px      : mean={iod.mean():.1f} sd={iod.std():.1f} "
            f"cv={iod.std() / iod.mean():.3f} min={iod.min():.1f} max={iod.max():.1f}"
        )
        print(
            f"face height px       : mean={fh.mean():.1f} sd={fh.std():.1f} "
            f"cv={fh.std() / fh.mean():.3f}"
        )
    if sharp:
        s = np.asarray(sharp, dtype=np.float64)
        print(f"face sharpness (Lap) : mean={s.mean():.1f} min={s.min():.1f}")
    if sims:
        v = np.asarray(sims, dtype=np.float64)
        print(
            f"identity cosine      : mean={np.nanmean(v):.4f} "
            f"min={np.nanmin(v):.4f} (SAME_PERSON threshold 0.363)"
        )
    if diffs:
        d = np.asarray(diffs, dtype=np.float64)
        still = int(np.sum(d < STILL_DIFF))
        print(
            f"motion (frame diff)  : mean={d.mean():.3f} median={np.median(d):.3f} "
            f"max={d.max():.3f}"
        )
        print(
            f"still frames (<{STILL_DIFF}) : {still}/{len(d)}  "
            f"<-- a shaking still image shows here (D-9.25: 211/239)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
