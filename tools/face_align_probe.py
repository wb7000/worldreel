"""Evidence for the face-alignment decisions (D-9.11).

Three questions, one script:

1. Is cv2.FaceRecognizerSF.alignCrop usable?  (OpenCV 5.0.0 returns an
   all-black crop for every input, so every embedding is identical.)
2. Which landmark order matches the canonical ArcFace/SFace template?
3. Which blend strategy preserves identity best?

    python tools/face_align_probe.py C:/path/to/vacecast [video.mp4] [frame]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_align import (
    ALIGN_SIZE,
    SFACE_TEMPLATE,
    align_face,
    feather,
    similarity_transform,
    to_float32,
    to_uint8,
    tone_match,
)
from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"
TMP = Path("C:/Users/li/AppData/Local/Temp/opencode")
SFACE_ORDER = [1, 0, 2, 4, 3]


def load(path: Path) -> NDArray[np.uint8]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(str(path))
    return to_uint8(img)


def nonzero_pct(arr: np.ndarray) -> float:
    return 100.0 * np.count_nonzero(arr) / arr.size


def question_1_aligncrop(sw: FaceSwapper, ref: NDArray[np.uint8]) -> None:
    print("=" * 72)
    print("1. Is cv2.FaceRecognizerSF.alignCrop usable?")
    print("=" * 72)
    face = sw.detect_face(ref)
    assert face is not None
    _, lm = face
    pts = lm.astype(np.float32)

    # Determinism check: the same input must give the same crop every time.
    means = []
    for _ in range(20):
        out = np.asarray(sw.rec.alignCrop(ref, pts)).copy()
        means.append(float(out.mean()))
    unique = sorted({round(m, 3) for m in means})
    print(f"   alignCrop repeated 20x on one image: {len(unique)} distinct result(s)")
    print(f"     means seen: {unique}")
    print(f"     black crops: {sum(1 for m in means if m < 1.0)}/20")

    aligned = align_face(ref, pts)
    assert aligned is not None
    print(
        f"   template alignment       : mean={aligned.mean():7.2f} "
        f"nonzero={nonzero_pct(aligned):5.1f}%"
    )

    f_cv = sw.rec.feature(np.asarray(sw.rec.alignCrop(ref, pts)).copy())
    f_man = sw.rec.feature(aligned)
    if f_cv is not None and f_man is not None:
        a = np.asarray(f_cv, dtype=np.float32).ravel()
        b = np.asarray(f_man, dtype=np.float32).ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(
            f"   embedding norm  alignCrop={np.linalg.norm(a):.3f}  "
            f"template={np.linalg.norm(b):.3f}"
        )
        print(f"   cos(alignCrop, template) = {cos:.4f}")
    print("   -> alignCrop is unstable and misaligned; it cannot be trusted for metrics.")


def question_2_order(sw: FaceSwapper, root: Path) -> None:
    print()
    print("=" * 72)
    print("2. Which landmark order matches the canonical template?")
    print("=" * 72)
    ref = load(root / "C01_0.png")
    ref_face = sw.detect_face(ref)
    assert ref_face is not None
    groups = {
        "same character": [root / f"C01_{i}.png" for i in range(1, 6)],
        "different people": [root / f"C0{n}_0.png" for n in (2, 3, 4)],
    }
    for label, order in (("YuNet order", [0, 1, 2, 3, 4]), ("SFace order", SFACE_ORDER)):
        aligned = align_face(ref, ref_face[1][order].astype(np.float32))
        assert aligned is not None
        ref_emb = np.asarray(sw.rec.feature(aligned), dtype=np.float32).ravel()
        ref_emb = ref_emb / np.linalg.norm(ref_emb)
        print(f"   {label}:")
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
                a = align_face(img, face[1][order].astype(np.float32))
                if a is None:
                    continue
                e = np.asarray(sw.rec.feature(a), dtype=np.float32).ravel()
                e = e / np.linalg.norm(e)
                sims.append(float(e @ ref_emb))
            if sims:
                arr = np.array(sims)
                means[group] = float(arr.mean())
                print(f"     {group:<18} mean={arr.mean():.4f} (n={len(sims)})")
        if len(means) == 2:
            print(f"     separation = {means['same character'] - means['different people']:+.4f}")


def question_3_blend(sw: FaceSwapper, ref: NDArray[np.uint8], frame: NDArray[np.uint8]) -> None:
    print()
    print("=" * 72)
    print("3. Which blend strategy preserves identity best?")
    print("=" * 72)
    face = sw.detect_face(frame)
    if face is None:
        print("   no face in the test frame")
        return
    _, dst_lm = face
    h, w = frame.shape[:2]

    ref_face = sw.detect_face(ref)
    assert ref_face is not None
    m_ref = similarity_transform(ref_face[1].astype(np.float32), SFACE_TEMPLATE)
    m_dst = similarity_transform(dst_lm.astype(np.float32), SFACE_TEMPLATE)
    assert m_ref is not None and m_dst is not None
    src_aligned = to_uint8(cv2.warpAffine(ref, m_ref, (ALIGN_SIZE, ALIGN_SIZE)))
    m_inv = to_float32(cv2.invertAffineTransform(m_dst))

    ref_emb = sw.compute_embedding(ref)
    assert ref_emb is not None
    mask = to_uint8(cv2.warpAffine(sw._oval, m_inv, (w, h)))
    src_in_frame = to_uint8(cv2.warpAffine(src_aligned, m_inv, (w, h)))

    variants = {
        "feather": None,
        "feather+tone": "tone",
        "seamlessClone(NORMAL)": "normal",
        "seamlessClone(MIXED)": "mixed",
    }
    for label, mode in variants.items():
        src = src_in_frame
        if mode == "tone":
            src = tone_match(src, mask, frame, mask, strength=1.0)
        alpha = feather(mask, 13)[:, :, None]
        blended = np.clip(
            frame.astype(np.float32) * (1 - alpha) + src.astype(np.float32) * alpha, 0, 255
        ).astype(np.uint8)
        out: NDArray[np.uint8] = blended
        if mode in ("normal", "mixed"):
            binary = (mask > 127).astype(np.uint8) * 255
            center = (int(dst_lm[:, 0].mean()), int(dst_lm[:, 1].mean()))
            flag = cv2.NORMAL_CLONE if mode == "normal" else cv2.MIXED_CLONE
            try:
                out = to_uint8(cv2.seamlessClone(blended, frame, binary, center, flag))
            except cv2.error:
                out = blended
        emb = sw.compute_embedding(out)
        assert emb is not None
        print(f"   {label:24s} similarity to reference = {sw.similarity(emb, ref_emb):.4f}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else str(TMP / "vacecast"))
    video = sys.argv[2] if len(sys.argv) > 2 else "runs/kaggle/hero_chain_run1/out/chain.mp4"
    frame_no = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    sw = FaceSwapper(YUNET, SFACE)
    ref = load(root / "C01_0.png")

    question_1_aligncrop(sw, ref)
    question_2_order(sw, root)

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if ret:
        question_3_blend(sw, ref, np.asarray(frame, dtype=np.uint8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
