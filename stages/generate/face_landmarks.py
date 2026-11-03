"""Dense face landmarks (MediaPipe Face Landmarker, Face Mesh V2: 478 points; model card
"Licensed under Apache License, Version 2.0"; `data/models/face_landmarker.task`, D-8.11e).

YuNet's five points (eyes, nose tip, mouth corners) never fix the lips' height or the jaw, so
the projected portrait's lips sat above / below the mesh's lips (owner: "lips fucked up").
"""

from __future__ import annotations

from functools import cache

import cv2
import numpy as np
import numpy.typing as npt

MODEL = "data/models/face_landmarker.task"
F64 = npt.NDArray[np.float64]

# Face Mesh indices (MediaPipe's canonical topology)
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39,
              40, 185)  # fmt: skip
LIPS_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81,
              80, 191)  # fmt: skip
EYES = (33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380)
BROWS = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)
NOSE = (1, 2, 98, 327, 168, 6, 48, 278)
OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67,
        109)  # fmt: skip
WARP = LIPS_OUTER + LIPS_INNER + EYES + BROWS + NOSE + OVAL
NOSE_BASE = (2, 98, 327)
CHIN = 152


@cache
def _landmarker():  # type: ignore[no-untyped-def]
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL), num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(opts)


def dense(bgr: npt.NDArray[np.uint8]) -> F64 | None:
    """478 x 2 pixel coordinates, or None when no face is found."""
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    out: F64 = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], np.float64)
    return out
