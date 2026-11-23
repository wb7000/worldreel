"""Normalise the reference portrait set for VACE conditioning (D-9.29).

Why this exists
---------------
VACE's `reference_images` are the model's only source of lighting, background and
framing: whatever spread exists in the reference set reappears in the output. Ours
(`C01_0..C01_5`) was a mess -- 512x512 squares whose backgrounds run from light grey
to near-black, whose lighting runs from flat frontal to a hard side key, whose face
scale varies by ~30% -- and the delivered segments reproduced exactly that spread.

The second problem is mechanical. diffusers' `WanVACEPipeline.preprocess_conditions`
scales every reference to *fit* the output size and pastes it centred onto
`canvas = torch.ones(3, H, W)` -- a **white** canvas. A square 512x512 reference
therefore reaches the model as a 480x480 image with 176 px of white bars on each side
of an 832x480 frame: half the conditioning width is white padding and the effective
conditioning resolution is 480x480 rather than 832x480. Feeding references that are
already at the output size and aspect removes the bars and raises the effective face
resolution by ~20%.

What this module does, per reference:

1. detect the face (YuNet) and crop a window with the **output's aspect ratio** in
   which the face occupies `face_frac` of the height and sits at `face_cy_frac` --
   so all six views present the head at the same scale and position;
2. replicate the border where the square source cannot fill the wide window (the
   studio wall simply continues; never white, never a letterbox bar);
3. resize to the output size with Lanczos and unsharp-mask the resample;
4. tone-match every normalised view to the primary reference (Reinhard per-channel,
   `stages.generate.face_align.tone_match`) so the set has ONE lighting.

The output feeds both the render's `reference_images` and the seed frame, so the
conditioning frame is no longer a 512x512 upscaling.

Usage:
    python tools/prepare_refs.py INPUT_DIR --out OUT_DIR [--contact SHEET.png]
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.generate.face_align import tone_match

YUNET = "training/kaggle/jobs/face_detection_yunet_2023mar.onnx"
SFACE = "training/kaggle/jobs/face_recognition_sface_2021dec.onnx"

# (width, height) -- VACE-1.3B's trained 480p resolution, and the render's.
OUT_SIZE = (832, 480)
# The face fills this share of the normalised height; the output video frames sit
# around 0.40-0.45, so the references are framed like the thing they condition.
FACE_FRAC = 0.42
# Where the face centre sits vertically inside the window (0.38 ≈ the seed frame's
# `ImageOps.fit(..., centering=(0.5, 0.42))`: head above the middle, shoulders below).
FACE_CY_FRAC = 0.38
# Unsharp amount after the Lanczos resample (0 = none).
SHARPEN = 0.6


def window_for_face(
    face_bbox: tuple[float, float, float, float],
    img_shape: tuple[int, ...],
    out_size: tuple[int, int] = OUT_SIZE,
    face_frac: float = FACE_FRAC,
    face_cy_frac: float = FACE_CY_FRAC,
) -> tuple[float, float, float, float]:
    """Crop window `(x0, y0, w, h)` in source coordinates for one face bbox.

    Pure geometry: the window has the output's aspect ratio, the face occupies
    `face_frac` of its height and its centre lands at `face_cy_frac` of its height,
    horizontally centred on the face. The window is clamped inside the source when
    the source is big enough; only a window that cannot fit is border-replicated by
    the caller (never a white bar).
    """
    img_w = img_shape[1]
    fx, fy, fw, fh = face_bbox
    out_w, out_h = out_size
    win_h = fh / face_frac
    win_w = win_h * (out_w / out_h)
    # Never wider than ~1.6x the source: an absurd window means a bad detection.
    win_w = min(win_w, img_w * 1.6)
    win_h = win_w * (out_h / out_w)
    cx = fx + fw / 2.0
    cy = fy + fh / 2.0
    x0 = cx - win_w / 2.0
    y0 = cy - face_cy_frac * win_h
    # Fit inside the source whenever the source is big enough: replicating the
    # bottom rows smears the sweater into streaks, and a window that fits needs
    # no padding at all. Only a window larger than the source is replicated.
    if win_h <= img_shape[0]:
        y0 = min(max(y0, 0.0), img_shape[0] - win_h)
    if win_w <= img_w:
        x0 = min(max(x0, 0.0), img_w - win_w)
    return (x0, y0, win_w, win_h)


def crop_resize(
    img: NDArray[np.uint8],
    window: tuple[float, float, float, float],
    out_size: tuple[int, int] = OUT_SIZE,
    sharpen: float = SHARPEN,
) -> NDArray[np.uint8]:
    """Crop `window` out of `img`, border-replicating what the source does not have,
    then Lanczos-resize to `out_size` and unsharp-mask the resample."""
    out_w, out_h = out_size
    x0, y0, win_w, win_h = window
    pad_l = max(0, int(np.ceil(-x0)))
    pad_t = max(0, int(np.ceil(-y0)))
    pad_r = max(0, int(np.ceil(x0 + win_w - img.shape[1])))
    pad_b = max(0, int(np.ceil(y0 + win_h - img.shape[0])))
    if pad_l or pad_t or pad_r or pad_b:
        # BORDER_REPLICATE: the studio wall continues instead of a white bar.
        img = np.asarray(
            cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE),
            dtype=np.uint8,
        )
        x0 += pad_l
        y0 += pad_t
    ix0, iy0 = round(x0), round(y0)
    ix1, iy1 = ix0 + round(win_w), iy0 + round(win_h)
    crop = img[max(0, iy0) : max(0, iy1), max(0, ix0) : max(0, ix1)]
    if crop.size == 0:
        crop = img
    resized = np.asarray(
        cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4), dtype=np.uint8
    )
    if sharpen <= 0:
        return resized
    blurred = cv2.GaussianBlur(resized, (0, 0), 1.2)
    return np.asarray(
        cv2.addWeighted(resized, 1.0 + sharpen, blurred, -sharpen, 0), dtype=np.uint8
    )


def normalize_reference(
    img_bgr: NDArray[np.uint8],
    face_bbox: tuple[float, float, float, float],
    out_size: tuple[int, int] = OUT_SIZE,
    face_frac: float = FACE_FRAC,
    face_cy_frac: float = FACE_CY_FRAC,
    sharpen: float = SHARPEN,
) -> NDArray[np.uint8]:
    """One reference, geometry-normalised (detection stays with the caller so the
    geometry is testable without the ONNX models)."""
    window = window_for_face(face_bbox, img_bgr.shape, out_size, face_frac, face_cy_frac)
    return crop_resize(img_bgr, window, out_size, sharpen)


def _fallback_bbox(img_shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    """A centred bbox when YuNet finds no face -- better than dropping the view."""
    img_h, img_w = img_shape[0], img_shape[1]
    fh = img_h * 0.35
    fw = fh * 0.75
    return ((img_w - fw) / 2.0, (img_h - fh) * 0.4, fw, fh)


def normalize_references(
    paths: Sequence[str | Path],
    out_dir: str | Path,
    yunet_path: str = YUNET,
    sface_path: str = SFACE,
    target: int = 0,
    out_size: tuple[int, int] = OUT_SIZE,
    face_frac: float = FACE_FRAC,
    tone_strength: float = 1.0,
    sharpen: float = SHARPEN,
) -> list[Path]:
    """Normalise every reference into `out_dir` and return the written paths.

    All views are geometrically normalised first, then tone-matched to
    `paths[target]` so the whole set carries ONE lighting (the render's seed frame
    comes from the primary, so matching to it keeps the conditioning frame in the
    same tone as the references).
    """
    from stages.generate.face_swap import FaceSwapper  # deferred: needs the ONNX files

    out_directory = Path(out_dir)
    out_directory.mkdir(parents=True, exist_ok=True)
    swapper = FaceSwapper(yunet_path, sface_path)

    normalised: list[NDArray[np.uint8]] = []
    names: list[str] = []
    for p in paths:
        path = Path(p)
        raw = cv2.imread(str(path))
        if raw is None:
            raise SystemExit(f"cannot read reference {path}")
        img = np.asarray(raw, dtype=np.uint8)
        det = swapper.detect_face(img)
        if det is not None:
            d = det[0]
            bbox = (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
        else:
            print(f"prepare_refs: no face in {path.name}, using a centred window")
            bbox = _fallback_bbox(img.shape)
        normalised.append(
            normalize_reference(img, bbox, out_size, face_frac, FACE_CY_FRAC, sharpen)
        )
        names.append(path.stem)

    if not normalised:
        raise SystemExit("no references to normalise")
    target_idx = min(max(0, target), len(normalised) - 1)
    ref = normalised[target_idx]
    ref_mask = np.full(ref.shape[:2], 255, dtype=np.uint8)
    for i, img in enumerate(normalised):
        if i != target_idx and tone_strength > 0:
            img = tone_match(img, ref_mask, ref, ref_mask, strength=tone_strength)
        out_path = out_directory / f"{names[i]}_norm.png"
        if not cv2.imwrite(str(out_path), img):
            raise SystemExit(f"cannot write {out_path}")
        print(f"prepare_refs: {Path(paths[i]).name} -> {out_path.name}")
    return [out_directory / f"{n}_norm.png" for n in names]


def contact_sheet(paths: Sequence[str | Path], out_path: str | Path, cols: int = 3) -> Path:
    """Tile the normalised references for visual inspection (no video tooling)."""
    imgs: list[NDArray[np.uint8]] = []
    for p in paths:
        raw = cv2.imread(str(p))
        if raw is None:
            raise SystemExit(f"contact sheet: cannot read {p}")
        imgs.append(np.asarray(raw, dtype=np.uint8))
    h, w = imgs[0].shape[:2]
    imgs = [np.asarray(cv2.resize(im, (w, h)), dtype=np.uint8) for im in imgs]
    rows: list[NDArray[np.uint8]] = []
    for r in range(0, len(imgs), cols):
        row = imgs[r : r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(imgs[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), sheet):
        raise SystemExit(f"cannot write {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="normalise VACE reference portraits")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contact", type=Path, default=None)
    parser.add_argument("--yunet", default=YUNET)
    parser.add_argument("--sface", default=SFACE)
    parser.add_argument("--target", type=int, default=0, help="index of the tone target")
    parser.add_argument("--face-frac", type=float, default=FACE_FRAC)
    parser.add_argument("--tone", type=float, default=1.0, help="0 disables tone matching")
    parser.add_argument("--sharpen", type=float, default=SHARPEN)
    parser.add_argument("--pattern", default="*", help="filename pattern, e.g. C01_*")
    args = parser.parse_args(argv)

    paths: list[Path] = []
    for ext in ("png", "jpg", "jpeg"):
        paths = sorted(args.input_dir.glob(f"{args.pattern}.{ext}"))
        if paths:
            break
    if not paths:
        print(f"no images in {args.input_dir}")
        return 2

    written = normalize_references(
        paths,
        args.out,
        yunet_path=args.yunet,
        sface_path=args.sface,
        target=args.target,
        face_frac=args.face_frac,
        tone_strength=args.tone,
        sharpen=args.sharpen,
    )
    sheet = args.contact or (args.out / "contact.png")
    contact_sheet(written, sheet)
    print(f"contact sheet: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
