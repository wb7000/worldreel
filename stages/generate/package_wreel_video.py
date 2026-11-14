"""
Package a face-consistent video as a signed `.wreel` file.

The `.wreel` format stores 3D scenes (Gaussian splats + actors + cameras).
To embed photorealistic video, we:
1. Add the MP4 video as a media file in the container
2. Create a minimal Gaussian "screen" layer (a few gaussians as a placeholder)
3. Add an actor that represents the video character
4. Sign with Ed25519

Usage:
    python stages/generate/package_wreel_video.py \
        --video training/kaggle/jobs/test_face_consistent.mp4 \
        --portrait training/kaggle/jobs/reference_portraits/C01_ref.png \
        --output runs/generated/S011_face_consistent.wreel \
        --title "Face-Consistent Photorealistic Video"
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from formats.common import load_or_create_signing_key
from formats.wreel import write_wreel
from stages.generate.face_swap import FaceSwapper


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def create_gaussian_screen(width: int, height: int, n_gaussians: int = 500) -> bytes:
    """
    Create a Gaussian splat "screen" - a flat plane of gaussians
    that can serve as a video surface.
    """
    positions: list[list[float]] = []
    scales: list[list[float]] = []
    rotations: list[list[float]] = []
    colors: list[list[float]] = []
    opacity: list[float] = []
    cls: list[int] = []
    confidence: list[float] = []

    # Create a grid of gaussians on a screen plane (Z = -1, facing camera)
    grid_size = int(np.ceil(np.sqrt(n_gaussians)))
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) >= n_gaussians:
                break
            x = (i - grid_size / 2) * 0.002  # ~1 metre wide screen
            y = (j - grid_size / 2) * 0.002
            z = -1.0  # 1 metre in front of camera

            positions.append([x, y, z])
            scales.append([0.001, 0.001, 0.001])  # tiny gaussians
            rotations.append([0.0, 0.0, 0.0, 1.0])  # identity quaternion
            colors.append([128, 128, 128])  # grey placeholder
            opacity.append(255)
            cls.append(4)  # "generated" provenance class
            confidence.append(1.0)

    dt = np.dtype(
        [
            ("position", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("rotation", "<f4", (4,)),
            ("rgb", "u1", (3,)),
            ("opacity", "u1"),
            ("cls", "u1"),
            ("reserved", "u1", (3,)),
            ("conf", "<f4"),
        ]
    )

    records = np.zeros(len(positions), dtype=dt)
    records["position"] = np.array(positions, dtype=np.float32)
    records["scale"] = np.array(scales, dtype=np.float32)
    records["rotation"] = np.array(rotations, dtype=np.float32)
    records["rgb"] = np.array(colors, dtype=np.uint8)
    records["opacity"] = np.array(opacity, dtype=np.uint8)
    records["cls"] = np.array(cls, dtype=np.uint8)
    records["conf"] = np.array(confidence, dtype=np.float32)

    return records.tobytes()


def package_wreel_video(
    video_path: str,
    reference_portrait_path: str,
    output_path: str,
    title: str = "Face-Consistent Video",
    yunet_path: str = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx",
    sface_path: str = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx",
    identity_seed: int = 42,
) -> dict[str, Any]:
    """
    Package a face-consistent video as a signed `.wreel` file.

    Returns a report dict.
    """
    video_file = Path(video_path)
    portrait_file = Path(reference_portrait_path)
    out_file = Path(output_path)

    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")
    if not portrait_file.exists():
        raise FileNotFoundError(f"Portrait not found: {portrait_file}")

    # --- Get video properties ---
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames, {duration:.2f}s")

    # --- Compute face embedding for identity record ---
    swapper = FaceSwapper(yunet_path, sface_path)
    loaded_ref = cv2.imread(str(portrait_file))
    ref_emb = (
        swapper.compute_embedding(np.asarray(loaded_ref, dtype=np.uint8))
        if loaded_ref is not None
        else None
    )

    # --- Build manifest ---
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Video file entry
    video_bytes = video_file.read_bytes()
    video_sha256 = hashlib.sha256(video_bytes).hexdigest()

    # Gaussian screen entry
    screen_bytes = create_gaussian_screen(width, height, n_gaussians=500)
    screen_sha256 = hashlib.sha256(screen_bytes).hexdigest()

    # Track file for the video actor (simple static track)
    track_data = {
        "times": [0.0, duration],
        "positions": [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        "rotations": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
    }
    track_bytes = json.dumps(track_data, indent=2).encode("utf-8")
    track_sha256 = hashlib.sha256(track_bytes).hexdigest()

    manifest = {
        "format": "wreel",
        "format_version": "1.1",
        "title": title,
        "duration": duration,
        "provenance": {
            "ai_label": "AI-generated-and-edited",
            "notice": (
                "AI-generated photorealistic video with face-consistent character identity. "
                "Not a recording."
            ),
            "tool": "worldreel",
            "tool_version": "1.0.0",
            "created_at": now,
            "source": {
                "kind": "video",
                "sha256": video_sha256,
            },
            "identity_policy": "synthetic-replacement",
            "rights": {
                "upload_rights": "synthetic",
                "license": "synthetic",
                "takedown": {"status": "none", "contact": ""},
            },
        },
        "generation": {
            "seeds": {"face_swap": identity_seed},
            "stages": [
                {
                    "name": "face_swap",
                    "version": "1.0.0",
                    "config_sha256": hashlib.sha256(b"face_swap_v1").hexdigest(),
                },
                {
                    "name": "wreel_package",
                    "version": "1.0.0",
                    "config_sha256": hashlib.sha256(b"wreel_package_v1").hexdigest(),
                },
            ],
            "models": [
                {"name": "Wan2.1-VACE-1.3B", "version": "1.0", "license": "research"},
                {"name": "OpenCV-YuNet", "version": "2023mar", "license": "Apache-2.0"},
                {"name": "OpenCV-SFace", "version": "2021dec", "license": "Apache-2.0"},
            ],
            "inputs": [
                {"role": "reference_portrait", "sha256": sha256_file(portrait_file)},
                {"role": "generated_video", "sha256": video_sha256},
            ],
        },
        "scene": {
            "bounds": {"min": [-10.0, -10.0, -10.0], "max": [10.0, 10.0, 10.0]},
            "background": [0.1, 0.1, 0.1],
            "lighting": {
                "ambient": {"color": [0.5, 0.5, 0.5], "intensity": 0.5},
                "lights": [
                    {
                        "id": "key_light",
                        "type": "directional",
                        "color": [1.0, 1.0, 1.0],
                        "intensity": 1.0,
                        "direction": [0.0, -1.0, -1.0],
                    }
                ],
            },
            "layers": [
                {
                    "id": "video_screen",
                    "kind": "gaussians",
                    "point_size": 0.002,
                    "lods": [
                        {
                            "level": 0,
                            "path": "scene/video_screen/lod0.gsp",
                            "count": 500,
                        }
                    ],
                }
            ],
        },
        "actors": [
            {
                "id": "C01",
                "type": "person",
                "label": "Face-consistent character",
                "character_id": "C01",
                "identity": {
                    "policy": "synthetic-replacement",
                    "identity_seed": identity_seed,
                },
                "geometry": {
                    "id": "avatar_C01",
                    "kind": "gaussians",
                    "point_size": 0.002,
                    "lods": [
                        {
                            "level": 0,
                            "path": "actors/C01/avatar/lod0.gsp",
                            "count": 500,
                        }
                    ],
                },
                "eye_height": 1.6,
                "track": {"path": "actors/C01/track.json"},
            }
        ],
        "cameras": [
            {
                "id": "cam0",
                "name": "Main camera",
                "kind": "director",
                "keyframes": [
                    {
                        "t": 0.0,
                        "position": [0.0, 1.6, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                        "fov_deg": 60.0,
                    },
                    {
                        "t": duration,
                        "position": [0.0, 1.6, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                        "fov_deg": 60.0,
                    },
                ],
            }
        ],
        "pov_presets": [
            {
                "id": "pov0",
                "name": "Director's cut",
                "mode": "directors-cut",
                "camera_id": "cam0",
            }
        ],
        "regions": [],
        "audio": {
            "tracks": [],
            "events": [],
            "mood": [],
        },
        "interaction": {
            "rules": [],
            "branches": [],
        },
        "files": {
            "video/face_consistent.mp4": {
                "sha256": video_sha256,
                "bytes": len(video_bytes),
                "media_type": "video/mp4",
            },
            "scene/video_screen/lod0.gsp": {
                "sha256": screen_sha256,
                "bytes": len(screen_bytes),
                "media_type": "application/octet-stream",
            },
            "actors/C01/avatar/lod0.gsp": {
                "sha256": screen_sha256,
                "bytes": len(screen_bytes),
                "media_type": "application/octet-stream",
            },
            "actors/C01/track.json": {
                "sha256": track_sha256,
                "bytes": len(track_bytes),
                "media_type": "application/json",
            },
        },
        "extensions": {},
    }

    # --- Build entries ---
    entries = {
        "video/face_consistent.mp4": video_bytes,
        "scene/video_screen/lod0.gsp": screen_bytes,
        "actors/C01/avatar/lod0.gsp": screen_bytes,
        "actors/C01/track.json": track_bytes,
    }

    # --- Sign and write ---
    key = load_or_create_signing_key()

    # write_wreel builds the manifest, signs it and writes the container itself.
    write_wreel(
        out_file,
        manifest,
        entries,
        key,
        validate=True,
    )

    print(f"\n.wreel written: {out_file}")
    print(f"  Size: {out_file.stat().st_size / 1024:.1f} KB")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Resolution: {width}x{height}")
    if ref_emb is not None:
        print(f"  Reference face embedding norm: {float(np.linalg.norm(ref_emb)):.4f}")

    return {
        "output_path": str(out_file),
        "size_bytes": out_file.stat().st_size,
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "manifest": manifest,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Package face-consistent video as .wreel")
    parser.add_argument("--video", required=True, help="Input video path (MP4)")
    parser.add_argument("--portrait", required=True, help="Reference portrait path")
    parser.add_argument("--output", required=True, help="Output .wreel path")
    parser.add_argument("--title", default="Face-Consistent Video", help="Video title")
    parser.add_argument("--yunet", default="training/kaggle/jobs/face_detection_yunet_2023mar.onnx")
    parser.add_argument(
        "--sface", default="training/kaggle/jobs/face_recognition_sface_2021dec.onnx"
    )
    parser.add_argument("--identity-seed", type=int, default=42)

    args = parser.parse_args()

    package_wreel_video(
        video_path=args.video,
        reference_portrait_path=args.portrait,
        output_path=args.output,
        title=args.title,
        yunet_path=args.yunet,
        sface_path=args.sface,
        identity_seed=args.identity_seed,
    )
