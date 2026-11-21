"""Build LoRA training targets from a rendered video (D-9.26).

Why this exists
---------------
The D-9.24 adapter was trained against a *still portrait* plus +-6 px micro-jitter.
Flow matching to those near-static latents taught the transformer the statistics of
one photograph, so the render came back as a still image that shakes (D-9.25:
88% of consecutive frames differ by < 0.5 grey levels, face Laplacian 66 -> 15).
The fix is not a better reward -- it is a better **target**: the target has to
contain the character's real motion and real texture, otherwise the adapter is
taught to erase both.

This module turns a rendered video (the D-9.25 reference render, `worldreel-data-v12render`)
into training variants:

    variant = [seed portrait] + [take frames]

where *seed portrait* is byte-identical to the conditioning seed frame, so the
target's first frame and the conditioning's first frame agree exactly.

The takes can be selected by measured identity (the best-matching takes become the
targets) and the portrait can be blended into the take frames to raise the target's
identity while keeping the take's motion.

CLI (local diagnostics):

    python -m tools.lora_targets VIDEO PORTRAIT [--take-len 48] [--takes 5]
        [--top 3] [--swap] [--dump DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_swap import FaceSwapper

U8 = NDArray[np.uint8]

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"


def read_video(path: str) -> list[U8]:
    """Every frame of a video as BGR uint8."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    frames: list[U8] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(np.asarray(frame, dtype=np.uint8))
    cap.release()
    return frames


def fit_seed(portrait_path: str, width: int, height: int) -> U8:
    """The conditioning seed frame: the portrait fitted to the render canvas.

    Must match the render job exactly (PIL LANCZOS fit, centering (0.5, 0.42)),
    because the target's first frame has to equal the conditioned frame.
    """
    img = Image.open(portrait_path).convert("RGB")
    fitted = ImageOps.fit(
        img, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.42)
    )
    return np.asarray(fitted, dtype=np.uint8)


def motion_diffs(frames: list[U8]) -> list[float]:
    """Mean absolute frame-to-frame difference (grey levels) for each step."""
    out: list[float] = []
    for i in range(1, len(frames)):
        a = frames[i - 1].astype(np.float32)
        b = frames[i].astype(np.float32)
        out.append(float(np.abs(a - b).mean()))
    return out


def frame_identities(
    swapper: FaceSwapper, frames: list[U8], ref_emb: NDArray[np.float32]
) -> list[float]:
    """SFace cosine of every frame against the reference portrait (-1 when no face)."""
    out: list[float] = []
    for f in frames:
        emb = swapper.compute_embedding(f)
        out.append(float(emb @ ref_emb) if emb is not None else -1.0)
    return out


def frame_sharpness(swapper: FaceSwapper, frames: list[U8]) -> list[float]:
    """Laplacian variance inside the detected face box (-1 when no face)."""
    out: list[float] = []
    for f in frames:
        det = swapper.detect_face(f)
        if det is None:
            out.append(-1.0)
            continue
        x, y, w, h = (int(v) for v in det[0][:4])
        roi = f[max(0, y) : y + h, max(0, x) : x + w]
        if roi.size == 0:
            out.append(-1.0)
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        out.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    return out


@dataclass
class Take:
    """One contiguous take of a rendered video."""

    index: int
    start: int
    frames: list[U8]
    identity_mean: float = -1.0
    identity_min: float = -1.0
    sharpness_mean: float = -1.0
    motion_mean: float = 0.0
    notes: list[str] = field(default_factory=list)


def split_takes(frames: list[U8], take_len: int, take_count: int) -> list[Take]:
    """Split a render into `take_count` takes of `take_len` frames."""
    takes: list[Take] = []
    for i in range(take_count):
        start = i * take_len
        chunk = frames[start : start + take_len]
        if len(chunk) < take_len:
            break
        takes.append(Take(index=i, start=start, frames=chunk))
    return takes


def score_takes(
    takes: list[Take],
    swapper: FaceSwapper,
    ref_emb: NDArray[np.float32],
) -> list[Take]:
    """Measure identity / sharpness / motion of every take, in place."""
    for take in takes:
        ids = frame_identities(swapper, take.frames, ref_emb)
        sharp = frame_sharpness(swapper, take.frames)
        diffs = motion_diffs(take.frames)
        valid_ids = [v for v in ids if v >= 0.0]
        valid_sharp = [v for v in sharp if v >= 0.0]
        take.identity_mean = float(np.mean(valid_ids)) if valid_ids else -1.0
        take.identity_min = float(np.min(valid_ids)) if valid_ids else -1.0
        take.sharpness_mean = float(np.mean(valid_sharp)) if valid_sharp else -1.0
        take.motion_mean = float(np.mean(diffs)) if diffs else 0.0
    return takes


def blend_portrait(
    frames: list[U8],
    portrait_bgr: U8,
    swapper: FaceSwapper,
    tone: float = 1.0,
) -> list[U8]:
    """Blend the portrait's face into every frame, keeping the take's motion.

    The face is warped into each frame's own geometry (position, scale, rotation),
    so the head still moves -- only the identity inside the face oval changes.
    """
    out: list[U8] = []
    for f in frames:
        out.append(
            np.asarray(swapper.swap_face(f, portrait_bgr, clone=False, tone=tone), dtype=np.uint8)
        )
    return out


def build_variants(
    takes: list[Take],
    seed_rgb: U8,
    swapper: FaceSwapper | None = None,
    portrait_bgr: U8 | None = None,
    swap: bool = False,
) -> list[list[U8]]:
    """Training variants: [seed] + take frames, as RGB uint8 lists."""
    variants: list[list[U8]] = []
    for take in takes:
        body = take.frames
        if swap:
            if swapper is None or portrait_bgr is None:
                raise ValueError("swap=True needs a swapper and a portrait")
            body = blend_portrait(body, portrait_bgr, swapper)
        variant = [seed_rgb] + [
            np.asarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), dtype=np.uint8) for f in body
        ]
        variants.append(variant)
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build/inspect LoRA training targets from a render"
    )
    parser.add_argument("video", help="rendered video to take the targets from")
    parser.add_argument("portrait", help="reference portrait (the character)")
    parser.add_argument("--take-len", type=int, default=48, help="frames per take")
    parser.add_argument("--takes", type=int, default=5, help="number of takes to inspect")
    parser.add_argument(
        "--top", type=int, default=3, help="how many best takes to keep as variants"
    )
    parser.add_argument("--swap", action="store_true", help="blend the portrait face into the take")
    parser.add_argument("--dump", default=None, help="write a JSON report here")
    args = parser.parse_args()

    swapper = FaceSwapper(YUNET, SFACE)
    portrait_bgr = cv2.imread(args.portrait)
    if portrait_bgr is None:
        print(f"FAIL: cannot read portrait {args.portrait}")
        return 2
    ref_emb = swapper.compute_embedding(np.asarray(portrait_bgr, dtype=np.uint8))
    if ref_emb is None:
        print("FAIL: no face in the reference portrait")
        return 2

    frames = read_video(args.video)
    print(f"{args.video}: {len(frames)} frames")
    takes = score_takes(split_takes(frames, args.take_len, args.takes), swapper, ref_emb)
    for take in takes:
        print(
            f"  take {take.index} [{take.start}:{take.start + args.take_len}] "
            f"identity mean {take.identity_mean:.4f} min {take.identity_min:.4f} | "
            f"sharpness {take.sharpness_mean:.1f} | motion {take.motion_mean:.3f}"
        )

    ranked = sorted(takes, key=lambda t: t.identity_mean, reverse=True)[: args.top]
    print("selected takes: " + ", ".join(str(t.index) for t in ranked))

    seed_rgb = fit_seed(args.portrait, frames[0].shape[1], frames[0].shape[0])
    variants = build_variants(
        ranked, seed_rgb, swapper, np.asarray(portrait_bgr, dtype=np.uint8), args.swap
    )
    for take, variant in zip(ranked, variants, strict=True):
        bgr = [np.asarray(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), dtype=np.uint8) for f in variant]
        ids = frame_identities(swapper, bgr, ref_emb)
        sharp = frame_sharpness(swapper, bgr)
        diffs = motion_diffs(bgr)
        valid = [v for v in ids if v >= 0.0]
        vs = [v for v in sharp if v >= 0.0]
        print(
            f"  variant from take {take.index}: identity mean {np.mean(valid):.4f} "
            f"min {np.min(valid):.4f} | sharpness {np.mean(vs):.1f} | "
            f"motion {np.mean(diffs):.3f} | frames {len(variant)}"
        )

    if args.dump:
        report = {
            "video": args.video,
            "portrait": args.portrait,
            "take_len": args.take_len,
            "takes": [
                {
                    "index": t.index,
                    "start": t.start,
                    "identity_mean": round(t.identity_mean, 4),
                    "identity_min": round(t.identity_min, 4),
                    "sharpness_mean": round(t.sharpness_mean, 2),
                    "motion_mean": round(t.motion_mean, 4),
                }
                for t in takes
            ],
            "selected": [t.index for t in ranked],
            "swap": bool(args.swap),
        }
        Path(args.dump).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written to {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
