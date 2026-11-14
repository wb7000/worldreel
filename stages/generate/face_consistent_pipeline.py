"""
End-to-End Character-Consistent Photorealistic Video Pipeline
=============================================================

Complete pipeline for generating photorealistic, face-consistent .wreel videos:

1. Generate video with Wan2.1-VACE + face embedding injection (Kaggle GPU)
2. Post-process with face swap (YuNet + Delaunay + seamlessClone) for guaranteed consistency
3. Package as signed .wreel with embedded video media

Usage:
    # Step 1: Run on Kaggle GPU
    python -m tools.kaggle_runner run training/kaggle/jobs/character_consistency_v2.py \
        --scene data:vacecast --scene data:facemodels --wait

    # Step 2: Process locally with face swap
    python stages/generate/face_consistent_pipeline.py --process-video \
        --video runs/kaggle/character_consistency_final/out/face_consistency.mp4 \
        --portrait training/kaggle/jobs/reference_portraits/C01_ref.png \
        --output training/kaggle/jobs/face_consistent_processed.mp4

    # Step 3: Package as .wreel
    python stages/generate/face_consistent_pipeline.py --package-wreel \
        --video training/kaggle/jobs/face_consistent_processed.mp4 \
        --portrait training/kaggle/jobs/reference_portraits/C01_ref.png \
        --output runs/generated/S012_face_consistent.wreel

    # Step 4: Validate
    python -m formats validate runs/generated/S012_face_consistent.wreel
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stages.generate.face_consistent_video import process_video_with_face_swap
from stages.generate.package_wreel_video import package_wreel_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end face-consistent photorealistic video pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- process-video subcommand ---
    p_proc = sub.add_parser("process-video", help="Apply face swap to video frames")
    p_proc.add_argument("--video", required=True, help="Input video path (MP4)")
    p_proc.add_argument("--portrait", required=True, help="Reference portrait path")
    p_proc.add_argument("--output", required=True, help="Output video path (MP4)")
    p_proc.add_argument("--yunet", default="training/kaggle/jobs/face_detection_yunet_2023mar.onnx")
    p_proc.add_argument(
        "--sface", default="training/kaggle/jobs/face_recognition_sface_2021dec.onnx"
    )
    p_proc.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    p_proc.add_argument("--report", default=None, help="Report output path (JSON)")

    # --- package-wreel subcommand ---
    p_wreel = sub.add_parser("package-wreel", help="Package video as signed .wreel")
    p_wreel.add_argument("--video", required=True, help="Input video path (MP4)")
    p_wreel.add_argument("--portrait", required=True, help="Reference portrait path")
    p_wreel.add_argument("--output", required=True, help="Output .wreel path")
    p_wreel.add_argument("--title", default="Face-Consistent Photorealistic Video")
    p_wreel.add_argument(
        "--yunet", default="training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
    )
    p_wreel.add_argument(
        "--sface", default="training/kaggle/jobs/face_recognition_sface_2021dec.onnx"
    )
    p_wreel.add_argument("--identity-seed", type=int, default=42)

    # --- full subcommand (process + package) ---
    p_full = sub.add_parser("full", help="Run process-video + package-wreel in one go")
    p_full.add_argument("--video", required=True, help="Input video path (MP4)")
    p_full.add_argument("--portrait", required=True, help="Reference portrait path")
    p_full.add_argument("--video-output", required=True, help="Processed video path (MP4)")
    p_full.add_argument("--wreel-output", required=True, help="Output .wreel path")
    p_full.add_argument("--title", default="Face-Consistent Photorealistic Video")
    p_full.add_argument("--yunet", default="training/kaggle/jobs/face_detection_yunet_2023mar.onnx")
    p_full.add_argument(
        "--sface", default="training/kaggle/jobs/face_recognition_sface_2021dec.onnx"
    )
    p_full.add_argument("--max-frames", type=int, default=None)
    p_full.add_argument("--identity-seed", type=int, default=42)

    args = parser.parse_args()

    yunet = args.yunet
    sface = args.sface

    if args.command == "process-video":
        report = process_video_with_face_swap(
            video_path=args.video,
            reference_portrait_path=args.portrait,
            output_path=args.output,
            yunet_path=yunet,
            sface_path=sface,
            max_frames=args.max_frames,
            report_path=args.report,
        )
        print(f"\n✓ Face-consistent video: {args.output}")
        print(f"  Frames: {report['frames_processed']}, Faces swapped: {report['faces_swapped']}")
        print(f"  Avg similarity to ref: {report['avg_similarity_to_ref']:.4f}")

    elif args.command == "package-wreel":
        report = package_wreel_video(
            video_path=args.video,
            reference_portrait_path=args.portrait,
            output_path=args.output,
            title=args.title,
            yunet_path=yunet,
            sface_path=sface,
            identity_seed=args.identity_seed,
        )
        print(f"\n✓ .wreel package: {args.output}")
        print(f"  Size: {report['size_bytes'] / 1024:.1f} KB")
        print(
            f"  Duration: {report['duration']:.2f}s, "
            f"{report['width']}x{report['height']} @ {report['fps']}fps"
        )

    elif args.command == "full":
        # Step 1: Process video with face swap
        print("=" * 60)
        print("STEP 1: Face swap post-processing")
        print("=" * 60)
        report1 = process_video_with_face_swap(
            video_path=args.video,
            reference_portrait_path=args.portrait,
            output_path=args.video_output,
            yunet_path=yunet,
            sface_path=sface,
            max_frames=args.max_frames,
        )
        print()

        # Step 2: Package as .wreel
        print("=" * 60)
        print("STEP 2: Package as signed .wreel")
        print("=" * 60)
        report2 = package_wreel_video(
            video_path=args.video_output,
            reference_portrait_path=args.portrait,
            output_path=args.wreel_output,
            title=args.title,
            yunet_path=yunet,
            sface_path=sface,
            identity_seed=args.identity_seed,
        )
        print()

        # Step 3: Summary
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Processed video: {args.video_output}")
        print(f"  .wreel package:  {args.wreel_output}")
        print(f"  .wreel size:     {report2['size_bytes'] / 1024:.1f} KB")
        print(f"  Duration:        {report2['duration']:.2f}s")
        print(f"  Resolution:      {report2['width']}x{report2['height']}")
        print(f"  Faces swapped:   {report1['faces_swapped']}/{report1['frames_processed']}")
        print(f"  Avg similarity:  {report1['avg_similarity_to_ref']:.4f}")
        print()
        print("Next: python -m formats validate", args.wreel_output)


if __name__ == "__main__":
    main()
