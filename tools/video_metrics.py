"""One implementation of the clip metrics, shared by the QA front-ends.

The D-9.25 adapter render passed every check that existed at the time -- one
face per frame, stable geometry, identity 0.93 -- and was still a "still image
that shakes": 211 of 239 consecutive frames differed by less than 0.5 grey
levels. The metric that would have caught it (motion) now lives here, next to
the identity, sharpness and geometry metrics, so no QA front-end can forget it.

`tools/v2_check.py` prints one clip; `tools/compare_videos.py` prints a table
for several clips (the A/B that decides which render ships).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_align import to_uint8
from stages.generate.face_swap import FaceSwapper

U8 = NDArray[np.uint8]

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"

# A frame-to-frame difference below this (grey levels) counts as "not moving".
STILL_DIFF = 0.5
# SFace's SAME_PERSON threshold for the 128-d embeddings used here.
SAME_PERSON = 0.363


@dataclass
class ClipMetrics:
    """Everything the QA gates measure on one clip."""

    path: str
    frames: int = 0
    width: int = 0
    height: int = 0
    faces_none: int = 0
    faces_multi: int = 0
    iod: list[float] = field(default_factory=list)
    face_heights: list[float] = field(default_factory=list)
    sharpness: list[float] = field(default_factory=list)
    identity: list[float] = field(default_factory=list)
    motion: list[float] = field(default_factory=list)

    @property
    def identity_mean(self) -> float:
        return float(np.nanmean(self.identity)) if self.identity else float("nan")

    @property
    def identity_min(self) -> float:
        return float(np.nanmin(self.identity)) if self.identity else float("nan")

    @property
    def identity_below(self) -> int:
        return int(sum(1 for v in self.identity if v < 0.80))

    @property
    def sharpness_mean(self) -> float:
        return float(np.mean(self.sharpness)) if self.sharpness else float("nan")

    @property
    def iod_mean(self) -> float:
        return float(np.mean(self.iod)) if self.iod else float("nan")

    @property
    def iod_sd(self) -> float:
        return float(np.std(self.iod)) if self.iod else float("nan")

    @property
    def height_mean(self) -> float:
        return float(np.mean(self.face_heights)) if self.face_heights else float("nan")

    @property
    def iod_cv(self) -> float:
        if not self.iod:
            return float("nan")
        arr = np.asarray(self.iod, dtype=np.float64)
        return float(arr.std() / arr.mean())

    @property
    def height_cv(self) -> float:
        if not self.face_heights:
            return float("nan")
        arr = np.asarray(self.face_heights, dtype=np.float64)
        return float(arr.std() / arr.mean())

    @property
    def motion_mean(self) -> float:
        return float(np.mean(self.motion)) if self.motion else 0.0

    @property
    def motion_median(self) -> float:
        return float(np.median(self.motion)) if self.motion else 0.0

    @property
    def still_frames(self) -> int:
        return int(sum(1 for v in self.motion if v < STILL_DIFF))

    @property
    def still_fraction(self) -> float:
        return self.still_frames / len(self.motion) if self.motion else 0.0

    @property
    def is_static(self) -> bool:
        """A clip that barely moves is not a video (D-9.25)."""
        return self.still_fraction > 0.5


def measure_clip(path: str, portrait_path: str | Path, identity_stride: int = 15) -> ClipMetrics:
    """Measure one clip against the reference portrait.

    `identity_stride` samples the identity every Nth frame (15 = the QA default,
    one embedding per ~1 s of video). A ranking probe wants every frame, so it
    passes 1; the default keeps the published QA numbers unchanged.
    """
    swapper = FaceSwapper(YUNET, SFACE)
    loaded = cv2.imread(str(portrait_path))
    if loaded is None:
        raise FileNotFoundError(f"missing reference portrait {portrait_path}")
    ref_emb = swapper.compute_embedding(to_uint8(loaded))
    if ref_emb is None:
        raise ValueError(f"no face in the reference portrait {portrait_path}")

    metrics = ClipMetrics(path=path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    metrics.width = int(cap.get(3))
    metrics.height = int(cap.get(4))

    prev: U8 | None = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = to_uint8(frame)
        if prev is not None:
            metrics.motion.append(
                float(np.abs(prev.astype(np.float32) - frame.astype(np.float32)).mean())
            )
        prev = frame

        # Ask for every detection (not just the most confident one) so a ghost
        # second face is counted instead of silently merged away.
        detected = swapper._detector(frame).detect(frame)[1]
        rows = [r for r in detected] if detected is not None else []
        if len(rows) == 0:
            metrics.faces_none += 1
        elif len(rows) > 1:
            metrics.faces_multi += 1

        if rows:
            best = rows[0]
            lm = np.asarray(best[4:14], dtype=np.float32).reshape(5, 2)
            bbox = np.asarray(best[0:4], dtype=np.float32)
            metrics.iod.append(float(np.linalg.norm(lm[0] - lm[1])))
            metrics.face_heights.append(float(bbox[3]))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x0, y0 = max(0, int(bbox[0])), max(0, int(bbox[1]))
            x1 = min(metrics.width, int(bbox[0] + bbox[2]))
            y1 = min(metrics.height, int(bbox[1] + bbox[3]))
            roi = gray[y0:y1, x0:x1]
            if roi.size:
                metrics.sharpness.append(float(cv2.Laplacian(roi, cv2.CV_64F).var()))
            if index % identity_stride == 0:
                emb = swapper.compute_embedding(frame)
                metrics.identity.append(float(emb @ ref_emb) if emb is not None else float("nan"))
        metrics.frames += 1
        index += 1
    cap.release()
    return metrics
