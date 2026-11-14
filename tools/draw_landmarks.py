"""Draw the detected YuNet landmarks on an image so the detection can be eyeballed."""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_align import to_uint8
from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"


def main() -> int:
    path = sys.argv[1]
    out = sys.argv[2]
    loaded = cv2.imread(path)
    if loaded is None:
        print(f"FAIL: cannot read {path}")
        return 2
    img = to_uint8(loaded)
    sw = FaceSwapper(YUNET, SFACE)
    face = sw.detect_face(img)
    if face is None:
        print("FAIL: no face detected")
        return 2
    bbox, lm = face
    print(f"image {img.shape[1]}x{img.shape[0]}")
    print(f"bbox {bbox.tolist()}")
    labels = ["R-eye", "L-eye", "nose", "R-mouth", "L-mouth"]
    for (x, y), label in zip(lm.tolist(), labels, strict=True):
        print(f"  {label:<8} ({x}, {y})")
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(
            img, label, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
        )
    x, y, w, h = bbox
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.imwrite(out, img)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
