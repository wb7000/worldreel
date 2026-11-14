"""Visual side-by-side of a face swap: target | reference | swapped."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_align import to_uint8
from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"


def load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return to_uint8(img)


def crop_face(img: np.ndarray, pad: int = 30) -> np.ndarray:
    sw = FaceSwapper(YUNET, SFACE)
    face = sw.detect_face(img)
    if face is None:
        return img
    x, y, w, h = face[0]
    return img[
        max(0, y - pad) : min(img.shape[0], y + h + pad),
        max(0, x - pad) : min(img.shape[1], x + w + pad),
    ]


def main() -> int:
    video, ref_path, frame_no, out_path = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        sys.argv[4],
    )
    sw = FaceSwapper(YUNET, SFACE)
    ref = load(ref_path)

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("FAIL: cannot read frame")
        return 2
    target = to_uint8(frame)
    swapped = sw.swap_face(target, ref)

    pre_emb = sw.compute_embedding(target)
    post_emb = sw.compute_embedding(swapped)
    ref_emb = sw.compute_embedding(ref)
    if pre_emb is None or post_emb is None or ref_emb is None:
        print("FAIL: no face detected in target, reference or swapped frame")
        return 2
    print(f"pre  sim={sw.similarity(pre_emb, ref_emb):.4f}")
    print(f"post sim={sw.similarity(post_emb, ref_emb):.4f}")

    panels = [crop_face(target), crop_face(ref), crop_face(swapped)]
    height = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height)) for p in panels]
    cv2.imwrite(out_path, np.hstack(panels))
    print(f"wrote {out_path} (target | reference | swapped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
