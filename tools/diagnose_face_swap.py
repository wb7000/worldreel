"""Diagnostic: does the face swapper preserve identity?

Swaps C01_0 (reference) onto other frames of the same character and reports the SFace
cosine similarity before and after the swap. A correct swap raises similarity toward 1.0.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"


def load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return np.asarray(img, dtype=np.uint8)


def main() -> int:
    ref_path = sys.argv[1]
    targets = sys.argv[2:]

    swapper = FaceSwapper(YUNET, SFACE)
    ref = load(ref_path)
    ref_emb = swapper.compute_embedding(ref)
    if ref_emb is None:
        print("FAIL: no face in reference")
        return 2

    print(f"reference: {ref_path}")
    print(f"{'target':<22} {'pre':>7} {'post':>7}  delta")
    print("-" * 50)
    for path in targets:
        target = load(path)
        pre_emb = swapper.compute_embedding(target)
        pre = swapper.similarity(pre_emb, ref_emb) if pre_emb is not None else float("nan")
        swapped = swapper.swap_face(target, ref)
        post_emb = swapper.compute_embedding(swapped)
        post = swapper.similarity(post_emb, ref_emb) if post_emb is not None else float("nan")
        flag = "  <-- WORSE" if post < pre else ""
        print(f"{Path(path).name:<22} {pre:>7.4f} {post:>7.4f}  {post - pre:+.4f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
