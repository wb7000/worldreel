"""Debug: overlay the swap mask + aligned reference on a generated frame."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_align import (
    SFACE_TEMPLATE,
    face_oval_mask,
    similarity_transform,
    to_uint8,
)
from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"
REF = Path("C:/Users/li/AppData/Local/Temp/opencode/vacecast/C01_0.png")
TMP = Path("C:/Users/li/AppData/Local/Temp/opencode")


def main() -> int:
    sw = FaceSwapper(YUNET, SFACE)
    loaded = cv2.imread(str(REF))
    assert loaded is not None
    ref = to_uint8(loaded)

    cap = cv2.VideoCapture("runs/kaggle/character_consistency_v2/out/face_consistency.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 45)
    ok, frame = cap.read()
    cap.release()
    assert ok
    frame = to_uint8(frame)

    ref_face = sw.detect_face(ref)
    dst_face = sw.detect_face(frame)
    assert ref_face is not None and dst_face is not None
    _, ref_lm = ref_face
    _, dst_lm = dst_face

    m_ref = similarity_transform(ref_lm.astype(np.float32), SFACE_TEMPLATE)
    m_dst = similarity_transform(dst_lm.astype(np.float32), SFACE_TEMPLATE)
    assert m_ref is not None and m_dst is not None

    h, w = frame.shape[:2]
    src_aligned = to_uint8(
        cv2.warpAffine(ref, m_ref, (112, 112), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    )
    m_inv = np.asarray(cv2.invertAffineTransform(m_dst), dtype=np.float32)
    src_in_frame = to_uint8(cv2.warpAffine(src_aligned, m_inv, (w, h), borderValue=(0, 0, 0)))
    mask = to_uint8(cv2.warpAffine(face_oval_mask(), m_inv, (w, h), borderValue=0))

    # Report geometry
    print(f"frame {w}x{h}")
    print(f"dst bbox {dst_face[0].tolist()}")
    print(f"dst lm  {dst_lm.tolist()}")
    print(f"ref bbox {ref_face[0].tolist()}")
    # Inter-ocular distance and face height as a scale proxy
    iod_dst = float(np.linalg.norm(dst_lm[0] - dst_lm[1]))
    face_h_dst = float(dst_face[0][3])
    iod_ref = float(np.linalg.norm(ref_lm[0] - ref_lm[1]))
    face_h_ref = float(ref_face[0][3])
    print(f"dst iod={iod_dst:.1f} face_h={face_h_dst:.1f} ratio={iod_dst / face_h_dst:.3f}")
    print(f"ref iod={iod_ref:.1f} face_h={face_h_ref:.1f} ratio={iod_ref / face_h_ref:.3f}")
    # yaw proxy: horizontal offset of nose tip from eye midpoint
    eye_mid_dst = (dst_lm[0] + dst_lm[1]) / 2.0
    yaw_dst = float(dst_lm[2][0] - eye_mid_dst[0])
    eye_mid_ref = (ref_lm[0] + ref_lm[1]) / 2.0
    yaw_ref = float(ref_lm[2][0] - eye_mid_ref[0])
    print(f"yaw dst={yaw_dst:.1f} ref={yaw_ref:.1f} (px, nose vs eye mid)")

    # Build overlays
    # 1) mask contour on original frame
    vis1 = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis1, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(TMP / "dbg_mask.png"), vis1)

    # 2) aligned reference (template space) upscaled
    cv2.imwrite(str(TMP / "dbg_ref_aligned.png"), cv2.resize(src_aligned, (336, 336)))

    # 3) reference warped into frame + alpha overlay
    vis3 = frame.copy()
    alpha = cv2.GaussianBlur(mask, (13, 13), 0).astype(np.float32) / 255.0
    vis3 = (
        frame.astype(np.float32) * (1 - alpha[:, :, None])
        + src_in_frame.astype(np.float32) * alpha[:, :, None]
    ).astype(np.uint8)
    cv2.imwrite(str(TMP / "dbg_blended.png"), vis3)

    # 4) side-by-side: original | blended | mask
    strip = np.hstack(
        [
            cv2.resize(frame, (400, 225)),
            cv2.resize(vis3, (400, 225)),
            cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (400, 225)),
        ]
    )
    cv2.imwrite(str(TMP / "dbg_strip.png"), strip)
    print("wrote dbg_mask.png dbg_ref_aligned.png dbg_blended.png dbg_strip.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
