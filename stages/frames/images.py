"""`frames` from an image sequence: the item's path is a folder of PNG/JPEG frames in name order.
Used for rendered scenes (benchmarks) and for photo sequences. Produces the same outputs as the
ffmpeg `frames` stage, minus audio and scene cuts."""

from __future__ import annotations

from typing import Any, ClassVar

import cv2
import numpy as np

from pipeline.io import FRAMES_DIR, FRAMES_JSON, PROXY_NPY, save_npy, write_json
from pipeline.stage import SOURCE, Impl, StageContext, StageError

DEFAULTS: dict[str, Any] = {"fps": 12.0, "proxy_width": 64, "step": 1}
_EXT = {".png", ".jpg", ".jpeg"}


class ImageSequenceFramesStage:
    name: ClassVar[str] = "frames"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "images"
    depends_on: ClassVar[tuple[str, ...]] = (SOURCE,)

    def run(self, ctx: StageContext) -> None:
        cfg = {**DEFAULTS, **ctx.config}
        src = ctx.item.path
        if not src.is_dir():
            raise StageError("the image-sequence frames stage needs a folder of images")
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in _EXT)[:: int(cfg["step"])]
        if not files:
            raise StageError(f"no images in {src}")
        out = ctx.out / FRAMES_DIR
        out.mkdir()
        proxies = []
        h = w = 0
        for i, path in enumerate(files, start=1):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise StageError(f"cannot read {path.name}")
            if i == 1:
                h, w = img.shape[:2]
            elif img.shape[:2] != (h, w):
                raise StageError(f"{path.name} has a different size")
            cv2.imwrite(str(out / f"{i:06d}.png"), img)
            pw = int(cfg["proxy_width"])
            ph = max(2, round(h * pw / w / 2) * 2)
            proxies.append(
                cv2.cvtColor(
                    cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
                )
            )
        fps = float(cfg["fps"]) / int(cfg["step"])
        save_npy(ctx.out / PROXY_NPY, np.stack(proxies).astype(np.uint8))
        write_json(
            ctx.out / FRAMES_JSON,
            {
                "width": w,
                "height": h,
                "source_fps": float(cfg["fps"]),
                "duration": len(files) / fps,
                "codec": "image-sequence",
                "has_audio": False,
                "fps": fps,
                "count": len(files),
                "frame_width": w,
                "frame_height": h,
                "proxy_count": len(files),
                "proxy_width": proxies[0].shape[1],
                "proxy_height": proxies[0].shape[0],
                "cuts": [],
                "audio": None,
            },
        )
