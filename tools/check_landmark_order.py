"""Determine the correct landmark order for the SFace/ArcFace 5-point template.

YuNet emits:  right eye, left eye, nose, right mouth corner, left mouth corner
SFace template expects: left eye, right eye, nose, mouth left corner, mouth right corner

Evaluated on a labelled set:
  same character   (C01_0 vs C01_1..5)  -> similarity should be HIGH
  different people (C01_0 vs C02/C03/C04) -> similarity should be LOW

The order that separates the two groups is the correct one.

This tool uses the canonical template alignment (stages/generate/face_align.py),
NOT cv2.FaceRecognizerSF.alignCrop, which returns an all-black crop on OpenCV
5.x and therefore produces the same embedding for every image.

    python tools/check_landmark_order.py C:/path/to/vacecast
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_align import align_face
from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"


def load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(str(path))
    return np.asarray(img, dtype=np.uint8)


def embedding(sw: FaceSwapper, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
    aligned = align_face(img, landmarks.astype(np.float32))
    if aligned is None:
        return None
    feat = sw.rec.feature(aligned)
    if feat is None:
        return None
    e = feat.ravel().astype(np.float32)
    norm = float(np.linalg.norm(e))
    if norm <= 0.0:
        return None
    return e / norm


def main() -> int:
    root = Path(sys.argv[1])
    sw = FaceSwapper(YUNET, SFACE)

    # YuNet: right eye, left eye, nose, right mouth, left mouth
    # Template: left eye, right eye, nose, left mouth, right mouth
    swap_eyes_mouth = [1, 0, 2, 4, 3]

    ref_img = load(root / "C01_0.png")
    ref_face = sw.detect_face(ref_img)
    if ref_face is None:
        print(f"FAIL: no face in {root / 'C01_0.png'}")
        return 2
    ref_lm = ref_face[1]

    groups = {
        "same character": [root / f"C01_{i}.png" for i in range(1, 6)],
        "different people": [root / f"C0{n}_0.png" for n in (2, 3, 4)],
    }

    for label, order in (("YuNet order", [0, 1, 2, 3, 4]), ("SFace order", swap_eyes_mouth)):
        ref_emb = embedding(sw, ref_img, ref_lm[order])
        if ref_emb is None:
            print(f"\n=== {label} ===\n  could not embed reference")
            continue
        print(f"\n=== {label} ===")
        means = {}
        for group, paths in groups.items():
            sims = []
            for p in paths:
                if not p.exists():
                    continue
                img = load(p)
                face = sw.detect_face(img)
                if face is None:
                    continue
                emb = embedding(sw, img, face[1][order])
                if emb is not None:
                    sims.append(sw.similarity(emb, ref_emb))
            if sims:
                arr = np.array(sims)
                means[group] = float(arr.mean())
                print(
                    f"  {group:<18} mean={arr.mean():.4f} min={arr.min():.4f} "
                    f"max={arr.max():.4f}  (n={len(sims)})"
                )
        if len(means) == 2:
            gap = means["same character"] - means["different people"]
            print(f"  separation (same - different) = {gap:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
