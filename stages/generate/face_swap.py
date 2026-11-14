"""Face swap utility for character-consistent video generation.

Uses YuNet face detection + SFace recognition + canonical-template affine
warping + OpenCV seamlessClone for photorealistic face swapping.

D-9.11 supersedes the original Delaunay-triangulation warp: the triangle mesh
built from five landmarks only covered the mid-face (the convex hull of eyes,
nose and mouth corners), so the forehead, jaw and chin kept the *original*
identity and the SFace embedding barely moved. Aligning both faces to the
canonical 112x112 template and warping the reference back through the
destination transform covers the whole face oval instead.

The template alignment also replaces `cv2.FaceRecognizerSF.alignCrop`, which is
unreliable in OpenCV 5.0.0 (it returns an all-black crop, collapsing every
embedding to the same constant). See `stages/generate/face_align.py`.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from stages.generate.face_align import (
    ALIGN_SIZE,
    SFACE_TEMPLATE,
    align_face,
    face_oval_mask,
    feather,
    similarity_transform,
    to_float32,
    to_uint8,
    tone_match,
)

BBox = NDArray[np.int64]
Landmarks = NDArray[np.int64]


class FaceSwapper:
    """Swap faces between images using landmark-based warping + seamless cloning."""

    def __init__(self, yunet_path: str, sface_path: str):
        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.rec = cv2.FaceRecognizerSF.create(sface_path, "")
        self._det: cv2.FaceDetectorYN | None = None
        self._det_size: tuple[int, int] | None = None
        self._oval = face_oval_mask()

    def _detector(self, img_bgr: NDArray[np.uint8]) -> cv2.FaceDetectorYN:
        """Get or recreate face detector matching image size."""
        h, w = img_bgr.shape[:2]
        if self._det is None or self._det_size != (w, h):
            self._det = cv2.FaceDetectorYN.create(self.yunet_path, "", (w, h), 0.6)
            self._det_size = (w, h)
        return self._det

    def detect_face(self, img_bgr: NDArray[np.uint8]) -> tuple[BBox, Landmarks] | None:
        """Detect the most confident face. Returns (bbox, landmarks) or None."""
        det = self._detector(img_bgr)
        _, faces = det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        detected = np.asarray(faces)
        f = detected[int(np.argmax(detected[:, -1]))]
        bbox = f[:4].astype(np.int64)  # x, y, w, h
        landmarks = f[4:14].reshape(5, 2).astype(np.int64)
        return bbox, landmarks

    def compute_embedding(self, img_bgr: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        """Compute SFace face embedding (L2-normalised).

        The face is first aligned to the canonical 112x112 template. We do NOT
        use `FaceRecognizerSF.alignCrop`: on OpenCV 5.0.0 it returns an
        all-black crop for every input, so `feature()` returns the same
        degenerate vector for every face and every similarity collapses to 1.0.
        """
        face = self.detect_face(img_bgr)
        if face is None:
            return None
        _, lm = face
        aligned = align_face(img_bgr, lm.astype(np.float32))
        if aligned is None:
            return None
        feat = self.rec.feature(aligned)
        if feat is None:
            return None
        emb = feat.ravel().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm <= 0.0:
            return None
        return emb / norm

    def similarity(self, emb_a: NDArray[np.float32], emb_b: NDArray[np.float32]) -> float:
        """Cosine similarity between two face embeddings."""
        return float(np.dot(emb_a, emb_b))

    def swap_face(
        self,
        frame_bgr: NDArray[np.uint8],
        ref_bgr: NDArray[np.uint8],
        clone: bool = False,
        tone: float = 1.0,
    ) -> NDArray[np.uint8]:
        """Swap the face in *frame_bgr* with the face from *ref_bgr*.

        Both faces are aligned to the canonical 112x112 template. The reference
        face is then warped back through the inverse of the destination
        transform, so it lands exactly on the destination's face geometry
        (position, scale, in-plane rotation), and blended inside the face oval.

        `clone` enables OpenCV's Poisson seamlessClone. It is OFF by default:
        Poisson blending copies the *destination's* image gradient into the
        result, which re-imprints strong source features (a moustache, a shadow)
        on top of the swapped face and measurably lowers identity similarity
        (0.796 vs 0.840 for the feathered blend on the test frame).
        `tone` scales the per-channel Reinhard tone match (0 disables it).
        """
        ref_face = self.detect_face(ref_bgr)
        if ref_face is None:
            return frame_bgr
        _, ref_lm = ref_face

        face = self.detect_face(frame_bgr)
        if face is None:
            return frame_bgr
        _, dst_lm = face

        h, w = frame_bgr.shape[:2]

        m_ref = similarity_transform(ref_lm.astype(np.float32), SFACE_TEMPLATE)
        m_dst = similarity_transform(dst_lm.astype(np.float32), SFACE_TEMPLATE)
        if m_ref is None or m_dst is None:
            return frame_bgr

        # Reference face in template space.
        src_aligned = to_uint8(
            cv2.warpAffine(
                to_uint8(ref_bgr),
                m_ref,
                (ALIGN_SIZE, ALIGN_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        )

        # Template space -> destination frame space.
        m_dst_inv = to_float32(cv2.invertAffineTransform(m_dst))
        src_in_frame = to_uint8(
            cv2.warpAffine(
                src_aligned,
                m_dst_inv,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        )
        mask_in_frame = to_uint8(
            cv2.warpAffine(
                self._oval,
                m_dst_inv,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        )

        if tone > 0.0:
            src_in_frame = tone_match(
                src_in_frame,
                mask_in_frame,
                to_uint8(frame_bgr),
                mask_in_frame,
                strength=tone,
            )

        alpha = feather(mask_in_frame, blur=13)[:, :, None]
        blended = to_float32(frame_bgr) * (1.0 - alpha)
        blended += to_float32(src_in_frame) * alpha
        blended = to_uint8(np.clip(blended, 0, 255))

        result = blended
        if clone:
            binary = (mask_in_frame > 127).astype(np.uint8) * 255
            center = (int(dst_lm[:, 0].mean()), int(dst_lm[:, 1].mean()))
            center = (max(10, min(w - 10, center[0])), max(10, min(h - 10, center[1])))
            if np.any(binary):
                try:
                    result = to_uint8(
                        cv2.seamlessClone(blended, frame_bgr, binary, center, cv2.NORMAL_CLONE)
                    )
                except cv2.error:
                    result = blended

        return to_uint8(result)


def test_swap(
    ref_path: str, frame_path: str, output_path: str, yunet_path: str, sface_path: str
) -> None:
    """Swap face from ref into frame and save result."""
    swapper = FaceSwapper(yunet_path, sface_path)

    ref_bgr = cv2.imread(ref_path)
    frame_bgr = cv2.imread(frame_path)

    if ref_bgr is None:
        raise FileNotFoundError(f"Cannot read ref: {ref_path}")
    if frame_bgr is None:
        raise FileNotFoundError(f"Cannot read frame: {frame_path}")

    ref_img: NDArray[np.uint8] = np.asarray(ref_bgr, dtype=np.uint8)
    frame_img: NDArray[np.uint8] = np.asarray(frame_bgr, dtype=np.uint8)

    result = swapper.swap_face(frame_img, ref_img)
    cv2.imwrite(output_path, result)

    # Report similarity
    ref_emb = swapper.compute_embedding(ref_img)
    frame_emb = swapper.compute_embedding(frame_img)
    result_emb = swapper.compute_embedding(result)

    print(f"Ref shape: {ref_img.shape}, Frame shape: {frame_img.shape}")
    print(f"Result saved: {output_path}")
    if ref_emb is not None:
        print(f"Ref embedding norm: {float(np.linalg.norm(ref_emb)):.4f}")
    if frame_emb is not None:
        print(f"Frame embedding norm: {float(np.linalg.norm(frame_emb)):.4f}")
    if result_emb is not None and ref_emb is not None:
        sim = swapper.similarity(result_emb, ref_emb)
        print(f"Result vs Ref similarity: {sim:.4f}")
    if result_emb is not None and frame_emb is not None:
        sim = swapper.similarity(result_emb, frame_emb)
        print(f"Result vs Frame similarity: {sim:.4f}")


if __name__ == "__main__":
    import sys

    yunet = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
    sface = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"

    # Use shot images as test data
    ref = "runs/kaggle/character_consistency/out/shot0_start.jpg"
    frame = "runs/kaggle/character_consistency/out/shot1_start.jpg"
    out = "training/kaggle/jobs/test_swap_output.jpg"

    if len(sys.argv) >= 4:
        ref, frame, out = sys.argv[1], sys.argv[2], sys.argv[3]

    test_swap(ref, frame, out, yunet, sface)
