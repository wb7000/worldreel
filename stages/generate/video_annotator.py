"""
Video Annotator — Visualizes invisible errors on video frames
============================================================

Creates an annotated MP4 where problem areas are highlighted with overlays:
- Red boxes: face regions with blur/artifacts
- Blue arrows: optical flow vectors (motion)
- Yellow heatmap: frame-to-frame difference
- Green: face landmarks
- Text overlay: per-frame metrics (PSNR, SSIM, blur, face similarity)

Usage:
    python stages/generate/video_annotator.py \
        --video input.mp4 \
        [--reference reference.mp4] \
        [--portrait reference_portrait.jpg] \
        [--output annotated.mp4] \
        [--sample-every 5]
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stages.generate.video_analyzer import VideoAnalyzer

Frame = NDArray[np.uint8]


class VideoAnnotator:
    """Draws invisible-error annotations on video frames."""

    def __init__(self, yunet_path: str, sface_path: str):
        self.analyzer = VideoAnalyzer(yunet_path, sface_path)

    def annotate_frame(
        self,
        frame: Frame,
        prev_frame: Frame | None,
        face_info: dict[str, Any] | None,
        flow: dict[str, Any] | None,
        diff: dict[str, Any] | None,
        frame_idx: int,
        metrics: dict[str, Any] | None,
    ) -> Frame:
        """Draw all annotations on a single frame."""
        h, w = frame.shape[:2]
        out = frame.copy()

        # --- Face region annotations ---
        if face_info and face_info.get("bbox"):
            x, y, fw, fh = face_info["bbox"]
            blur = face_info.get("blur", 0)
            edge_density = face_info.get("edge_density", 0)

            # Color based on blur level. The thresholds match the analyzer's
            # subject-sharpness rule so the two tools never contradict each other.
            if blur < 50:
                box_color = (0, 0, 255)  # red = blurry
                label = "BLURRY FACE"
            elif blur < 100:
                box_color = (0, 165, 255)  # orange = moderate
                label = "MODERATE BLUR"
            else:
                box_color = (0, 255, 0)  # green = good
                label = "FACE OK"

            cv2.rectangle(out, (x, y), (x + fw, y + fh), box_color, 3)
            cv2.putText(out, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.putText(
                out,
                f"blur={blur:.0f}",
                (x, y + fh + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                1,
            )

            # Draw landmarks
            if "landmarks" in face_info:
                for lx, ly in face_info["landmarks"]:
                    cv2.circle(out, (int(lx), int(ly)), 3, (255, 255, 0), -1)

            # Edge density warning
            if edge_density > 0.08:
                cv2.putText(
                    out,
                    "HIGH EDGE DENSITY (artifacts?)",
                    (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        # --- Optical flow arrows ---
        if flow and prev_frame is not None:
            pg = np.asarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
            cg = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
            # Preallocate the flow field (CV_32FC2): OpenCV accepts None here, the stubs do not.
            flow_buf = np.zeros((pg.shape[0], pg.shape[1], 2), dtype=np.float32)
            flow_arr = cv2.calcOpticalFlowFarneback(pg, cg, flow_buf, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Draw flow vectors on a grid (subsample for clarity)
            step = 16
            for i in range(step // 2, h, step):
                for j in range(step // 2, w, step):
                    fx = flow_arr[i, j, 0]
                    fy = flow_arr[i, j, 1]
                    mag = np.sqrt(fx * fx + fy * fy)
                    if mag > 1.0:
                        # Color by magnitude (green=low, red=high)
                        color = (0, int(255 * min(mag / 10, 1)), int(255 * (1 - min(mag / 10, 1))))
                        cv2.arrowedLine(
                            out, (j, i), (int(j + fx), int(i + fy)), color, 1, tipLength=0.3
                        )

            # Motion magnitude text
            if flow.get("mean_motion") is not None:
                cv2.putText(
                    out,
                    f"motion={flow['mean_motion']:.2f}",
                    (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # --- Frame difference heatmap ---
        if diff and prev_frame is not None:
            diff_img = np.abs(frame.astype(np.float64) - prev_frame.astype(np.float64))
            diff_gray = diff_img.mean(axis=2).astype(np.uint8)
            diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            # Blend with original (30% heatmap)
            out = np.asarray(cv2.addWeighted(out, 0.7, diff_color, 0.3, 0), dtype=np.uint8)

            cv2.putText(
                out,
                f"diff={diff.get('mean_diff', 0):.1f}",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # --- Metrics overlay ---
        if metrics:
            y_offset = 30
            for key, val in metrics.items():
                text = f"{key}: {val:.3f}" if isinstance(val, float) else f"{key}: {val}"
                cv2.putText(
                    out, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                y_offset += 20

        # --- Frame index ---
        cv2.putText(
            out,
            f"Frame: {frame_idx}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return out

    def annotate_video(
        self,
        video_path: str,
        output_path: str,
        portrait_path: str | None = None,
        sample_every: int = 5,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        """Create annotated video with invisible-error highlights."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total = min(total, max_frames)

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        prev_frame: Frame | None = None
        prev_face_info: dict[str, Any] | None = None
        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx % sample_every == 0:
                frame_img: Frame = np.asarray(frame, dtype=np.uint8)

                # Get face info
                face_info: dict[str, Any] | None = None
                if portrait_path:
                    face = self.analyzer.face_swapper.detect_face(frame_img)
                    if face:
                        bbox, lm = face
                        emb = self.analyzer.face_swapper.compute_embedding(frame_img)
                        face_info = {
                            "bbox": bbox.tolist(),
                            "landmarks": lm.tolist(),
                            # Use the same measure as the analyzer's subject-sharpness rule
                            # (detail_lap_variance). The plain Laplacian variance of a face
                            # region is dominated by smooth skin, so a perfectly sharp face
                            # reads as "blurry" and the annotator contradicts the analyzer.
                            "blur": self.analyzer.detail_lap_variance(
                                frame_img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                            ),
                            "embedding_sim": None,  # filled if prev exists
                        }
                        if (
                            prev_face_info
                            and prev_face_info.get("embedding") is not None
                            and emb is not None
                        ):
                            face_info["embedding_sim"] = self.analyzer.face_swapper.similarity(
                                emb, prev_face_info["embedding"]
                            )
                        if emb is not None:
                            face_info["embedding"] = emb

                # Get flow + diff
                flow: dict[str, Any] | None = None
                diff: dict[str, Any] | None = None
                if prev_frame is not None:
                    flow = self.analyzer.optical_flow_magnitude(prev_frame, frame_img)
                    diff = self.analyzer.frame_difference(prev_frame, frame_img)

                # Build metrics overlay
                metrics: dict[str, Any] = {}
                if flow:
                    metrics["motion"] = flow["mean_motion"]
                if diff:
                    metrics["frame_diff"] = diff["mean_diff"]
                if face_info and face_info.get("blur") is not None:
                    metrics["face_blur"] = face_info["blur"]
                if face_info and face_info.get("embedding_sim") is not None:
                    metrics["face_sim"] = face_info["embedding_sim"]

                annotated = self.annotate_frame(
                    frame_img, prev_frame, face_info, flow, diff, frame_idx, metrics
                )
                out_writer.write(annotated)
                saved += 1

                prev_frame = frame_img.copy()
                prev_face_info = face_info

            frame_idx += 1

        cap.release()
        out_writer.release()

        print(f"Annotated video: {saved} frames written to {output_path}")
        return {"output": output_path, "frames": saved}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate video with invisible-error overlays")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--portrait", default=None, help="Reference portrait for face analysis")
    parser.add_argument("--output", required=True, help="Output annotated video path")
    parser.add_argument("--sample-every", type=int, default=5, help="Annotate every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")

    args = parser.parse_args()

    yunet = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
    sface = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"

    annotator = VideoAnnotator(yunet, sface)
    annotator.annotate_video(
        video_path=args.video,
        output_path=args.output,
        portrait_path=args.portrait,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
    )
