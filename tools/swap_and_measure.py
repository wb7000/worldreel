"""Swap a reference portrait onto a video and report the REAL identity change.

This is the honest measurement the Kaggle job was missing: it reported `sim=1.0` whenever a
face was still detectable after the swap, never the actual embedding similarity.

Usage:
    python tools/swap_and_measure.py --video in.mp4 --portrait ref.png [--out out.mp4]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_swap import FaceSwapper

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"


def load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return np.asarray(img, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--portrait", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    swapper = FaceSwapper(YUNET, SFACE)
    ref = load(args.portrait)
    ref_emb = swapper.compute_embedding(ref)
    if ref_emb is None:
        print("FAIL: no face in the reference portrait")
        return 2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"FAIL: cannot open {args.video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    pre_sims: list[float] = []
    post_sims: list[float] = []
    rows: list[dict[str, float | int | bool]] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_img = np.asarray(frame, dtype=np.uint8)
        swapped = swapper.swap_face(frame_img, ref)

        pre_emb = swapper.compute_embedding(frame_img)
        post_emb = swapper.compute_embedding(swapped)
        pre = swapper.similarity(pre_emb, ref_emb) if pre_emb is not None else float("nan")
        post = swapper.similarity(post_emb, ref_emb) if post_emb is not None else float("nan")
        if np.isfinite(pre):
            pre_sims.append(pre)
        if np.isfinite(post):
            post_sims.append(post)
        rows.append(
            {
                "frame": idx,
                "pre_sim": round(pre, 4),
                "post_sim": round(post, 4),
                "delta": round(post - pre, 4),
                "face_found": post_emb is not None,
            }
        )
        if writer is not None:
            writer.write(swapped)
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()

    if not post_sims:
        print("FAIL: no faces detected in the video")
        return 2

    pre_arr = np.array(pre_sims, dtype=np.float64)
    post_arr = np.array(post_sims, dtype=np.float64)
    print(f"frames: {idx}, faces detected: {len(post_sims)}")
    print(f"PRE  swap similarity to reference: mean={pre_arr.mean():.4f} min={pre_arr.min():.4f}")
    print(f"POST swap similarity to reference: mean={post_arr.mean():.4f} min={post_arr.min():.4f}")
    print(f"improvement: {post_arr.mean() - pre_arr.mean():+.4f}")
    print(f"frames below 0.85 after swap: {int(np.sum(post_arr < 0.85))}/{len(post_arr)}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(
                {
                    "video": args.video,
                    "portrait": args.portrait,
                    "frames": idx,
                    "pre_mean": float(pre_arr.mean()),
                    "post_mean": float(post_arr.mean()),
                    "post_min": float(post_arr.min()),
                    "below_threshold": int(np.sum(post_arr < 0.85)),
                    "per_frame": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
