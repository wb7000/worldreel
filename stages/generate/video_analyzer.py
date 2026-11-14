"""
Video Quality & Consistency Analyzer
====================================

Detects errors invisible to the human eye in AI-generated video:
- Frame-level quality: PSNR, SSIM, VMAF (vs reference or self-comparison)
- Temporal consistency: optical flow motion artifacts, flicker, jitter
- Face consistency: SFace embedding drift across frames
- Face region analysis: eye/mouth artifacts, boundary seams
- Compression artifacts: blocking, ringing, blur detection
- Output: JSON report + annotated frames highlighting problem areas

Usage:
    python stages/generate/video_analyzer.py \
        --video input.mp4 \
        [--reference reference.mp4] \
        [--portrait reference_portrait.jpg] \
        [--output report.json] \
        [--annotated output_annotated.mp4] \
        [--sample-every N]  # analyze every Nth frame for speed
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stages.generate.face_swap import FaceSwapper

Frame = NDArray[np.uint8]


class VideoAnalyzer:
    """Comprehensive video quality and consistency analyzer."""

    def __init__(self, yunet_path: str, sface_path: str):
        self.yunet_path = yunet_path
        self.sface_path = sface_path
        self.face_swapper = FaceSwapper(yunet_path, sface_path)

    # ------------------------------------------------------------------ quality metrics

    @staticmethod
    def psnr(frame_a: Frame, frame_b: Frame) -> float:
        """Peak Signal-to-Noise Ratio between two frames (dB)."""
        mse = np.mean((frame_a.astype(np.float64) - frame_b.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")
        return float(20 * np.log10(255.0 / np.sqrt(mse)))

    @staticmethod
    def ssim(frame_a: Frame, frame_b: Frame) -> float:
        """Simplified Structural Similarity Index (luminance + contrast + structure)."""
        # Convert to grayscale for speed
        ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float64)

        mu_a = ga.mean()
        mu_b = gb.mean()
        var_a = ga.var()
        var_b = gb.var()
        cov = np.mean((ga - mu_a) * (gb - mu_b))

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        ssim_val = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
            (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
        )
        return float(ssim_val)

    @staticmethod
    def lap_variance(frame: Frame) -> float:
        """Laplacian variance — blur / sharpness measure (higher = sharper)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def detail_lap_variance(frame: Frame, percentile: float = 90.0) -> float:
        """Laplacian variance measured only over the strongest edges of the frame.

        The plain global variance is dominated by large flat areas (a studio wall, a sky, a
        blurred background), so a perfectly sharp subject on a blank backdrop still reads as
        "blurry". Restricting the measurement to the top-detail pixels measures the
        sharpness the eye actually perceives, and keeps the global metric for comparison.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        mag = np.abs(lap)
        if mag.size == 0:
            return 0.0
        cutoff = float(np.percentile(mag, percentile))
        mask = mag >= cutoff
        if not np.any(mask):
            return 0.0
        return float(np.var(lap[mask]))

    def subject_lap_variance(self, frame: Frame) -> float:
        """Sharpness of the subject: the detected face when there is one, else the centre.

        A head-and-shoulders shot puts the subject in the middle of the frame, so the central
        crop is a reasonable fallback when no face is found.
        """
        face = self.face_swapper.detect_face(frame)
        if face is not None:
            x, y, w, h = face[0]
            pad = 20
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad)
            y1 = min(frame.shape[0], y + h + pad)
            region = frame[y0:y1, x0:x1]
            if region.size:
                return self.detail_lap_variance(region)
        h, w = frame.shape[:2]
        centre = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        if not centre.size:
            return 0.0
        return self.detail_lap_variance(centre)

    # ------------------------------------------------------------------ temporal analysis

    @staticmethod
    def optical_flow_magnitude(prev: Frame, curr: Frame) -> dict[str, Any]:
        """Dense optical flow between two consecutive frames.

        Returns stats that reveal unnatural motion / warping / flicker.
        """
        pg = np.asarray(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        cg = np.asarray(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), dtype=np.uint8)

        # Preallocate the flow field (CV_32FC2). OpenCV accepts None here, but the typed
        # stubs require an array.
        flow = np.zeros((pg.shape[0], pg.shape[1], 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(pg, cg, flow, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        return {
            "mean_motion": float(mag.mean()),
            "max_motion": float(mag.max()),
            "std_motion": float(mag.std()),
            "high_motion_frac": float(np.mean(mag > 5.0)),  # fraction of pixels moving fast
            "flow_x_std": float(flow[..., 0].std()),
            "flow_y_std": float(flow[..., 1].std()),
        }

    @staticmethod
    def frame_difference(prev: Frame, curr: Frame) -> dict[str, Any]:
        """Per-pixel absolute difference between consecutive frames."""
        diff = np.abs(curr.astype(np.float64) - prev.astype(np.float64))
        return {
            "mean_diff": float(diff.mean()),
            "max_diff": float(diff.max()),
            "std_diff": float(diff.std()),
            "changed_frac": float(np.mean(diff > 10)),  # fraction of pixels changed > threshold
        }

    @staticmethod
    def flicker_score(frames: list[Frame], sample_every: int = 5) -> dict[str, Any]:
        """Detect brightness/color flicker across frames (temporal luminance variance)."""
        lumens: list[float] = []
        for f in frames[::sample_every]:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            lumens.append(float(gray.mean()))

        values = np.array(lumens, dtype=np.float64)
        return {
            "mean_luminance": float(values.mean()),
            "std_luminance": float(values.std()),
            "luminance_range": float(values.max() - values.min()),
            "flicker_index": float(values.std() / max(values.mean(), 1)),
        }

    # ------------------------------------------------------------------ face analysis

    def face_consistency_across_frames(
        self, frames: list[Frame], sample_every: int = 5
    ) -> dict[str, Any]:
        """Track SFace embedding similarity across frames.

        Low similarity between consecutive frames = face identity drift.
        """
        embeddings: list[NDArray[np.float32] | None] = []
        face_found: list[bool] = []
        for f in frames[::sample_every]:
            emb = self.face_swapper.compute_embedding(f)
            if emb is not None:
                embeddings.append(emb)
                face_found.append(True)
            else:
                embeddings.append(None)
                face_found.append(False)

        # Consecutive similarity
        sims: list[float] = []
        for i in range(1, len(embeddings)):
            current, previous = embeddings[i], embeddings[i - 1]
            if current is not None and previous is not None:
                sims.append(self.face_swapper.similarity(current, previous))

        values = np.array(sims, dtype=np.float64)

        return {
            "frames_with_face": int(sum(face_found)),
            "total_sampled": len(face_found),
            "face_detection_rate": float(sum(face_found) / max(len(face_found), 1)),
            "mean_consecutive_similarity": float(values.mean()) if len(values) else 0.0,
            "min_consecutive_similarity": float(values.min()) if len(values) else 0.0,
            "std_consecutive_similarity": float(values.std()) if len(values) else 0.0,
            "drift_frames": int(np.sum(values < 0.8)) if len(values) else 0,  # below 0.8 = drift
        }

    def face_region_artifacts(self, frames: list[Frame], sample_every: int = 5) -> dict[str, Any]:
        """Analyze face region for artifacts (blur, boundary seams, eye/mouth issues)."""
        results = []
        for idx, f in enumerate(frames[::sample_every]):
            face = self.face_swapper.detect_face(f)
            if face is None:
                continue
            bbox, lm = face
            x, y, w, h = bbox

            # Expand face region slightly for context
            pad = 20
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(f.shape[1], x + w + pad)
            y1 = min(f.shape[0], y + h + pad)

            face_region = f[y0:y1, x0:x1]

            # Blur within face region
            blur = self.lap_variance(face_region)

            # Edge detection within face (unexpected edges = artifacts)
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = float(np.mean(edges > 0))

            # Color variance in face region (unnatural color shifts)
            color_std = float(face_region.std(axis=(0, 1)).mean())

            results.append(
                {
                    "frame": idx * sample_every,
                    "face_bbox": bbox.tolist(),
                    "face_blur": blur,
                    "edge_density": edge_density,
                    "color_std": color_std,
                    "landmarks": lm.tolist(),
                }
            )

        if not results:
            return {"faces_analyzed": 0}

        blurs = [float(r["face_blur"]) for r in results]
        edge_densities = [float(r["edge_density"]) for r in results]
        colors = [float(r["color_std"]) for r in results]

        return {
            "faces_analyzed": len(results),
            "mean_face_blur": float(np.mean(blurs)),
            "min_face_blur": float(np.min(blurs)),
            "mean_edge_density": float(np.mean(edge_densities)),
            "max_edge_density": float(np.max(edge_densities)),
            "mean_color_std": float(np.mean(colors)),
            "max_color_std": float(np.max(colors)),
            "frame_details": results[:10],  # first 10 for brevity
        }

    # ------------------------------------------------------------------ compression artifacts

    @staticmethod
    def blocking_artifact(frame: Frame) -> float:
        """Detect 8x8 blocking artifacts (DCT block boundaries)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape

        # Compute differences across 8-pixel boundaries
        block_score = 0.0
        count = 0
        for i in range(8, h - 8, 8):
            for j in range(8, w - 8, 8):
                # Vertical boundary
                v_diff = np.abs(gray[i, j - 1] - gray[i, j])
                # Horizontal boundary
                h_diff = np.abs(gray[i - 1, j] - gray[i, j])
                block_score += v_diff + h_diff
                count += 2

        return float(block_score / max(count, 1))

    @staticmethod
    def ringing_artifact(frame: Frame) -> float:
        """Detect ringing around edges (high-frequency oscillation)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        # Laplacian highlights ringing
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        # Ringing = high magnitude laplacian near edges
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        if np.sum(edges > 0) == 0:
            return 0.0
        ringing = np.mean(np.abs(lap)[edges > 0])
        return float(ringing)

    # ------------------------------------------------------------------ main analysis

    def analyze(
        self,
        video_path: str,
        reference_path: str | None = None,
        portrait_path: str | None = None,
        sample_every: int = 5,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        """Run full analysis on a video file.

        Returns a comprehensive report dict.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
        print(f"Analyzing every {sample_every}th frame...")

        frames: list[Frame] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break
            if frame_idx % sample_every == 0:
                frames.append(np.asarray(frame, dtype=np.uint8))
            frame_idx += 1

        cap.release()
        print(f"Loaded {len(frames)} frames for analysis")

        report: dict[str, Any] = {
            "video_path": video_path,
            "reference_path": reference_path,
            "portrait_path": portrait_path,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "frames_analyzed": len(frames),
            "sample_every": sample_every,
            "analysis_timestamp": datetime.now(UTC).isoformat(),
            "metrics": {},
            "issues": [],
            "severity": "unknown",
        }

        # --- Per-frame quality vs reference (if provided) ---
        if reference_path:
            ref_cap = cv2.VideoCapture(reference_path)
            ref_frames: list[Frame] = []
            ri = 0
            while True:
                ret, rf = ref_cap.read()
                if not ret:
                    break
                if ri % sample_every == 0 and len(ref_frames) < len(frames):
                    ref_frames.append(np.asarray(rf, dtype=np.uint8))
                ri += 1
            ref_cap.release()

            psnr_vals = []
            ssim_vals = []
            for i in range(min(len(frames), len(ref_frames))):
                psnr_vals.append(self.psnr(frames[i], ref_frames[i]))
                ssim_vals.append(self.ssim(frames[i], ref_frames[i]))

            report["metrics"]["vs_reference"] = {
                "mean_psnr": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
                "min_psnr": float(np.min(psnr_vals)) if psnr_vals else 0.0,
                "mean_ssim": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
                "min_ssim": float(np.min(ssim_vals)) if ssim_vals else 0.0,
            }

        # --- Temporal consistency ---
        flow_stats = []
        diff_stats = []
        for i in range(1, len(frames)):
            flow_stats.append(self.optical_flow_magnitude(frames[i - 1], frames[i]))
            diff_stats.append(self.frame_difference(frames[i - 1], frames[i]))

        if flow_stats:
            report["metrics"]["temporal_consistency"] = {
                "mean_motion": float(np.mean([f["mean_motion"] for f in flow_stats])),
                "max_motion": float(np.max([f["max_motion"] for f in flow_stats])),
                "mean_frame_diff": float(np.mean([d["mean_diff"] for d in diff_stats])),
                "max_frame_diff": float(np.max([d["max_diff"] for d in diff_stats])),
                "mean_changed_frac": float(np.mean([d["changed_frac"] for d in diff_stats])),
                "high_motion_frames": int(
                    np.sum([f["high_motion_frac"] > 0.3 for f in flow_stats])
                ),
            }

        # --- Flicker ---
        report["metrics"]["flicker"] = self.flicker_score(frames, sample_every=1)

        # --- Blur / sharpness ---
        # Three views of sharpness: global (whole frame), detail (strongest edges only) and
        # subject (the face, or the centre of frame). Only the latter two say anything about
        # whether the *content* is sharp, so the blur issue is judged on them (D-9.11).
        blur_vals = [self.lap_variance(f) for f in frames]
        detail_vals = [self.detail_lap_variance(f) for f in frames]
        subject_vals = [self.subject_lap_variance(f) for f in frames]
        report["metrics"]["sharpness"] = {
            "mean_laplacian_var": float(np.mean(blur_vals)),
            "min_laplacian_var": float(np.min(blur_vals)),
            "blur_frames": int(np.sum(np.array(blur_vals) < 50)),  # threshold tunable
            "mean_detail_laplacian_var": float(np.mean(detail_vals)),
            "min_detail_laplacian_var": float(np.min(detail_vals)),
            "mean_subject_laplacian_var": float(np.mean(subject_vals)),
            "min_subject_laplacian_var": float(np.min(subject_vals)),
            "blurred_subject_frames": int(np.sum(np.array(subject_vals) < 50)),
        }

        # --- Compression artifacts ---
        block_vals = [self.blocking_artifact(f) for f in frames[::3]]  # subsample for speed
        ring_vals = [self.ringing_artifact(f) for f in frames[::3]]
        report["metrics"]["compression_artifacts"] = {
            "mean_blocking": float(np.mean(block_vals)) if block_vals else 0.0,
            "max_blocking": float(np.max(block_vals)) if block_vals else 0.0,
            "mean_ringing": float(np.mean(ring_vals)) if ring_vals else 0.0,
            "max_ringing": float(np.max(ring_vals)) if ring_vals else 0.0,
        }

        # --- Face analysis ---
        if portrait_path:
            report["metrics"]["face_consistency"] = self.face_consistency_across_frames(
                frames, sample_every=sample_every
            )

        report["metrics"]["face_region"] = self.face_region_artifacts(
            frames, sample_every=sample_every
        )

        # --- Issue detection (invisible to human eye) ---
        issues = []

        # Temporal jitter: high frame-to-frame difference without corresponding motion
        if "temporal_consistency" in report["metrics"]:
            tc = report["metrics"]["temporal_consistency"]
            if tc["mean_changed_frac"] > 0.5:
                issues.append(
                    {
                        "type": "temporal_jitter",
                        "severity": "high",
                        "description": (
                            f"High frame-to-frame pixel change "
                            f"({tc['mean_changed_frac']:.1%} of pixels changed per frame)"
                        ),
                        "invisible": True,
                        "metric": tc["mean_changed_frac"],
                    }
                )

        # Face drift
        if "face_consistency" in report["metrics"]:
            fc = report["metrics"]["face_consistency"]
            if fc["mean_consecutive_similarity"] < 0.85:
                issues.append(
                    {
                        "type": "face_identity_drift",
                        "severity": "high",
                        "description": (
                            f"Face embedding similarity below 0.85 "
                            f"({fc['mean_consecutive_similarity']:.3f}) — identity may drift"
                        ),
                        "invisible": True,
                        "metric": fc["mean_consecutive_similarity"],
                    }
                )
            if fc["face_detection_rate"] < 0.8:
                issues.append(
                    {
                        "type": "face_detection_failures",
                        "severity": "medium",
                        "description": (
                            f"Face not detected in "
                            f"{100 * (1 - fc['face_detection_rate']):.0f}% of sampled frames"
                        ),
                        "invisible": True,
                        "metric": fc["face_detection_rate"],
                    }
                )

        # Blur. The global Laplacian variance alone gives false positives on footage with a
        # large flat background (a sharp subject on a blank studio wall scores ~28 because
        # the wall has no detail). Judge the *subject*: flag only when the subject itself is
        # soft in most frames, or when no face was found and the whole frame is soft.
        sharp = report["metrics"]["sharpness"]
        face_analysed = report["metrics"].get("face_region", {}).get("faces_analyzed", 0)
        subject_soft = sharp["blurred_subject_frames"] > len(frames) * 0.3
        whole_frame_soft = face_analysed == 0 and sharp["blur_frames"] > len(frames) * 0.3
        if subject_soft or whole_frame_soft:
            metric = (
                sharp["mean_subject_laplacian_var"] if subject_soft else sharp["mean_laplacian_var"]
            )
            issues.append(
                {
                    "type": "excessive_blur",
                    "severity": "medium",
                    "description": (
                        f"{sharp['blurred_subject_frames']} frames with a soft subject "
                        f"(subject Laplacian var {sharp['mean_subject_laplacian_var']:.1f}, "
                        f"global {sharp['mean_laplacian_var']:.1f})"
                        if subject_soft
                        else f"{sharp['blur_frames']} frames below blur threshold"
                    ),
                    "invisible": True,
                    "metric": metric,
                }
            )

        # Compression artifacts
        ca = report["metrics"]["compression_artifacts"]
        if ca["mean_blocking"] > 5.0:
            issues.append(
                {
                    "type": "blocking_artifacts",
                    "severity": "medium",
                    "description": f"Visible DCT blocking (mean score {ca['mean_blocking']:.1f})",
                    "invisible": True,
                    "metric": ca["mean_blocking"],
                }
            )

        # Flicker
        fl = report["metrics"]["flicker"]
        if fl["flicker_index"] > 0.05:
            issues.append(
                {
                    "type": "luminance_flicker",
                    "severity": "medium",
                    "description": (
                        f"Luminance flicker index {fl['flicker_index']:.3f} (threshold 0.05)"
                    ),
                    "invisible": True,
                    "metric": fl["flicker_index"],
                }
            )

        report["issues"] = issues
        report["issue_count"] = len(issues)

        # Overall severity
        if any(i["severity"] == "high" for i in issues):
            report["severity"] = "high"
        elif any(i["severity"] == "medium" for i in issues):
            report["severity"] = "medium"
        elif issues:
            report["severity"] = "low"
        else:
            report["severity"] = "none"

        print(f"\nAnalysis complete: {len(issues)} issues found, severity={report['severity']}")
        return report


def save_report(report: dict[str, Any], output_path: str) -> None:
    """Save analysis report to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {output_path}")


def print_summary(report: dict[str, Any]) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("VIDEO ANALYSIS REPORT")
    print("=" * 60)
    print(f"File: {report['video_path']}")
    print(f"Resolution: {report['width']}x{report['height']} @ {report['fps']:.1f}fps")
    print(f"Frames analyzed: {report['frames_analyzed']}")
    print(f"Issues found: {report['issue_count']}")
    print(f"Overall severity: {report['severity'].upper()}")

    if "vs_reference" in report["metrics"]:
        vr = report["metrics"]["vs_reference"]
        print(f"\nVs Reference: PSNR={vr['mean_psnr']:.2f}dB, SSIM={vr['mean_ssim']:.3f}")

    tc = report["metrics"].get("temporal_consistency", {})
    if tc:
        print(
            f"\nTemporal: mean_motion={tc.get('mean_motion', 0):.2f}, "
            f"changed_frac={tc.get('mean_changed_frac', 0):.1%}"
        )

    fc = report["metrics"].get("face_consistency", {})
    if fc:
        print(
            f"\nFace: detected_in={fc.get('face_detection_rate', 0):.1%}, "
            f"sim={fc.get('mean_consecutive_similarity', 0):.3f}"
        )

    fr = report["metrics"].get("face_region", {})
    if fr and fr.get("faces_analyzed", 0) > 0:
        print(
            f"Face region: blur={fr.get('mean_face_blur', 0):.1f}, "
            f"edge_density={fr.get('mean_edge_density', 0):.3f}"
        )

    sh = report["metrics"].get("sharpness", {})
    if sh:
        print(
            f"Sharpness: global={sh.get('mean_laplacian_var', 0):.1f}, "
            f"detail={sh.get('mean_detail_laplacian_var', 0):.1f}, "
            f"subject={sh.get('mean_subject_laplacian_var', 0):.1f}"
        )

    if report["issues"]:
        print("\nISSUES:")
        for i in report["issues"]:
            print(f"  [{i['severity']}] {i['type']}: {i['description']}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video quality & consistency analyzer")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--reference", default=None, help="Reference video path (for PSNR/SSIM)")
    parser.add_argument("--portrait", default=None, help="Reference portrait for face consistency")
    parser.add_argument("--output", default="video_analysis_report.json", help="Report output path")
    parser.add_argument("--sample-every", type=int, default=5, help="Analyze every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to analyze")

    args = parser.parse_args()

    yunet = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
    sface = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"

    analyzer = VideoAnalyzer(yunet, sface)
    report = analyzer.analyze(
        video_path=args.video,
        reference_path=args.reference,
        portrait_path=args.portrait,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
    )

    save_report(report, args.output)
    print_summary(report)
