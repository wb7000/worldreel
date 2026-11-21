"""How consistent is the character through a clip, measured against ITSELF?

The QA numbers compare every frame with the reference PHOTO (SFace cosine).
That measures two different things at once:

  * "is this the same person as in the photo" -- an absolute identity score,
  * "does the face morph into somebody else during the clip" -- consistency.

A clip can score 0.78 against the photo and still be perfectly consistent (the
photo has a different pose/light than the clip). The user's question is the
second one, so measure it: every frame against the clip's own median face
embedding, and against its own first frame.

Usage:
    python tools/clip_consistency.py VIDEO [VIDEO ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_align import to_uint8
from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"


def embeddings(path: Path) -> tuple[list[np.ndarray], list[int]]:
    swapper = FaceSwapper(YUNET, SFACE)
    cap = cv2.VideoCapture(str(path))
    embs: list[np.ndarray] = []
    indices: list[int] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        emb = swapper.compute_embedding(to_uint8(frame))
        if emb is not None:
            embs.append(np.asarray(emb, dtype=np.float64))
            indices.append(index)
        index += 1
    cap.release()
    return embs, indices


def report(path: Path) -> None:
    embs, indices = embeddings(path)
    if not embs:
        print(f"{path}: no face embeddings")
        return
    matrix = np.stack(embs)
    median = np.median(matrix, axis=0)
    median /= np.linalg.norm(median)
    first = matrix[0] / np.linalg.norm(matrix[0])
    to_median = matrix @ median
    to_first = matrix @ first
    pairwise = matrix @ matrix.T
    off = pairwise[np.triu_indices(len(matrix), k=1)]
    print(f"{path}")
    print(f"  frames with a face : {len(matrix)}")
    print(f"  vs own median face : mean={to_median.mean():.4f} min={to_median.min():.4f} "
          f"(frame {indices[int(np.argmin(to_median))]})")
    print(f"  vs own first frame : mean={to_first.mean():.4f} min={to_first.min():.4f} "
          f"(frame {indices[int(np.argmin(to_first))]})")
    print(f"  frame-to-frame pair: mean={off.mean():.4f} min={off.min():.4f}")
    drift = [float(v) for v in to_median]
    step = max(1, len(drift) // 12)
    print("  vs median over time: " + " ".join(f"{v:.3f}" for v in drift[::step]))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        report(Path(arg))
