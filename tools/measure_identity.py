"""Measure the REAL face-identity similarity of a video against a reference portrait.

The Kaggle job reported `sim=1.0` because it only checked that a face was still detectable
after the swap. This script computes the actual cosine similarity between the SFace
embedding of the reference portrait and the SFace embedding of the face in each video frame.
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
    video_path, portrait_path = sys.argv[1], sys.argv[2]
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    swapper = FaceSwapper(YUNET, SFACE)

    portrait = load(portrait_path)
    print(f"portrait {portrait_path}: {portrait.shape[1]}x{portrait.shape[0]}")
    ref_face = swapper.detect_face(portrait)
    if ref_face is None:
        print("FAIL: no face detected in the reference portrait")
        return 2
    ref_bbox, _ = ref_face
    print(f"reference face bbox: {ref_bbox.tolist()} (w={ref_bbox[2]}, h={ref_bbox[3]})")
    ref_emb = swapper.compute_embedding(portrait)
    if ref_emb is None:
        print("FAIL: no reference embedding")
        return 2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"FAIL: cannot open {video_path}")
        return 2

    sims: list[float] = []
    detected = 0
    total = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every == 0:
            total += 1
            frame_img = np.asarray(frame, dtype=np.uint8)
            emb = swapper.compute_embedding(frame_img)
            if emb is not None:
                detected += 1
                sims.append(swapper.similarity(emb, ref_emb))
        idx += 1
    cap.release()

    if not sims:
        print("FAIL: no faces detected in the video")
        return 2

    values = np.array(sims, dtype=np.float64)
    print(f"\nframes sampled: {total}, faces detected: {detected}")
    print(
        f"REAL similarity to reference: mean={values.mean():.4f} min={values.min():.4f} "
        f"max={values.max():.4f} std={values.std():.4f}"
    )
    print(f"frames below 0.85 threshold: {int(np.sum(values < 0.85))}/{len(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
