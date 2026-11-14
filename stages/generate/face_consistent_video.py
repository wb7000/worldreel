"""
Character-Consistent Photorealistic Video Pipeline
===================================================

Takes a generated video and applies face-swap post-processing using
YuNet + Delaunay triangulation + seamlessClone to ensure the same
character identity appears in every frame.

Usage:
    python stages/generate/face_consistent_video.py \
        --video input.mp4 \
        --portrait reference.png \
        --output output.mp4 \
        --yunet path/to/yunet.onnx \
        --sface path/to/sface.onnx
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stages.generate.face_swap import FaceSwapper


def process_video_with_face_swap(
    video_path: str,
    reference_portrait_path: str,
    output_path: str,
    yunet_path: str,
    sface_path: str,
    max_frames: int | None = None,
    report_path: str | None = None,
) -> dict[str, Any]:
    """
    Process every frame of a video with face swap for character consistency.

    Returns a report dict with per-frame metrics.
    """
    swapper = FaceSwapper(yunet_path, sface_path)

    # Load reference portrait
    loaded_ref = cv2.imread(reference_portrait_path)
    if loaded_ref is None:
        raise FileNotFoundError(f"Cannot read reference portrait: {reference_portrait_path}")
    ref_img: NDArray[np.uint8] = np.asarray(loaded_ref, dtype=np.uint8)

    ref_emb = swapper.compute_embedding(ref_img)
    if ref_emb is None:
        raise RuntimeError(f"No face found in reference portrait: {reference_portrait_path}")
    print(f"Reference face embedding computed (norm={float(np.linalg.norm(ref_emb)):.4f})")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    if max_frames:
        total_frames = min(total_frames, max_frames)

    # Setup output video writer
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    report: dict[str, Any] = {
        "reference_portrait": reference_portrait_path,
        "input_video": video_path,
        "output_video": output_path,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": 0,
        "frames_processed": 0,
        "faces_found": 0,
        "faces_swapped": 0,
        "avg_similarity_to_ref": 0.0,
        "frame_reports": [],
    }
    frame_reports: list[dict[str, Any]] = []

    frame_idx = 0
    sim_sum = 0.0
    sim_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        frame_img: NDArray[np.uint8] = np.asarray(frame, dtype=np.uint8)

        # Apply face swap
        result = swapper.swap_face(frame_img, ref_img)

        # Compute similarity
        result_emb = swapper.compute_embedding(result)
        frame_emb = swapper.compute_embedding(frame_img)

        face_found = frame_emb is not None
        face_swapped = not np.array_equal(result, frame_img)

        sim_to_ref = 0.0
        if result_emb is not None:
            sim_to_ref = swapper.similarity(result_emb, ref_emb)
            sim_sum += sim_to_ref
            sim_count += 1

        if face_found:
            report["faces_found"] = int(report["faces_found"]) + 1
        if face_swapped:
            report["faces_swapped"] = int(report["faces_swapped"]) + 1

        frame_reports.append(
            {
                "frame": frame_idx,
                "face_found": face_found,
                "face_swapped": face_swapped,
                "similarity_to_ref": round(sim_to_ref, 4),
            }
        )

        out_writer.write(result)
        report["frames_processed"] = int(report["frames_processed"]) + 1
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out_writer.release()

    report["frame_reports"] = frame_reports
    report["total_frames"] = frame_idx
    report["avg_similarity_to_ref"] = (
        round(sim_sum / max(sim_count, 1), 4) if sim_count > 0 else 0.0
    )

    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
        print(f"Report saved: {report_path}")

    print(f"\nDone! Processed {report['frames_processed']} frames")
    print(f"  Faces found: {report['faces_found']}")
    print(f"  Faces swapped: {report['faces_swapped']}")
    print(f"  Avg similarity to ref: {report['avg_similarity_to_ref']:.4f}")
    print(f"  Output: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face-consistent video post-processing")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--portrait", required=True, help="Reference portrait path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--yunet", default="training/kaggle/jobs/face_detection_yunet_2023mar.onnx")
    parser.add_argument(
        "--sface", default="training/kaggle/jobs/face_recognition_sface_2021dec.onnx"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--report", default=None, help="Report output path")

    args = parser.parse_args()

    process_video_with_face_swap(
        video_path=args.video,
        reference_portrait_path=args.portrait,
        output_path=args.output,
        yunet_path=args.yunet,
        sface_path=args.sface,
        max_frames=args.max_frames,
        report_path=args.report,
    )
