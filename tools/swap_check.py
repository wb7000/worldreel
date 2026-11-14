"""Visual + numeric check of the new template-based face swap."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_align import to_uint8
from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"
TMP = Path("C:/Users/li/AppData/Local/Temp/opencode")
REF = TMP / "vacecast" / "C01_0.png"


def main() -> int:
    sw = FaceSwapper(YUNET, SFACE)
    loaded_ref = cv2.imread(str(REF))
    assert loaded_ref is not None
    ref = to_uint8(loaded_ref)

    cap = cv2.VideoCapture("runs/kaggle/hero_chain_run1/out/chain.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
    ret, frame = cap.read()
    cap.release()
    assert ret
    frame = to_uint8(frame)

    swapped = sw.swap_face(frame, ref)

    ref_emb = sw.compute_embedding(ref)
    pre_emb = sw.compute_embedding(frame)
    post_emb = sw.compute_embedding(swapped)
    assert ref_emb is not None and pre_emb is not None and post_emb is not None
    print(f"pre  vs ref: {sw.similarity(pre_emb, ref_emb):.4f}")
    print(f"post vs ref: {sw.similarity(post_emb, ref_emb):.4f}")
    print(f"post vs pre: {sw.similarity(post_emb, pre_emb):.4f}")
    changed = float(np.abs(swapped.astype(np.int32) - frame.astype(np.int32)).mean())
    print(f"mean abs pixel delta: {changed:.2f}")

    # Side-by-side strip: frame | swapped | ref
    strip = np.hstack(
        [
            cv2.resize(frame, (320, 320)),
            cv2.resize(swapped, (320, 320)),
            cv2.resize(ref, (320, 320)),
        ]
    )
    cv2.putText(strip, "ORIG", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(strip, "SWAPPED", (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(strip, "REF C01_0", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imwrite(str(TMP / "swap_check.png"), strip)

    # Same check with the reference's own face region only.
    face = sw.detect_face(frame)
    if face is not None:
        x, y, w, h = face[0]
        pad = 40
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        crop = np.hstack([frame[y0:y1, x0:x1], swapped[y0:y1, x0:x1]])
        crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3))
        cv2.imwrite(str(TMP / "swap_check_crop.png"), crop)
    print("wrote", TMP / "swap_check.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
