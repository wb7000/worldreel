"""Export one camera path of a `.wreel` as a flat MP4 or a `.wedit` edit bundle (brief Phase 6).

    python -m tools.export_video world.wreel --out clip.mp4
    python -m tools.export_video world.wreel --out clip.wedit --wedit [--zip]

`--camera` picks the camera path (default: the first one in the world). The `.wedit` bundle opens
in Shotcut via `timeline/main.mlt`; `timeline/main.otio` is the OpenTimelineIO version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from formats.wreel import read_wreel
from stages.export.flat import export_mp4
from stages.export.wedit_export import export_wedit, sha256_file


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.export_video")
    p.add_argument("world", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera", help="camera path id (default: the first)")
    p.add_argument("--wedit", action="store_true", help="write a .wedit bundle instead of an MP4")
    p.add_argument("--zip", action="store_true", help="with --wedit: one zip file, not a folder")
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--width", type=int, default=854)
    p.add_argument("--height", type=int, default=480)
    a = p.parse_args(argv)

    world = read_wreel(a.world)
    cameras = [c["id"] for c in world.manifest["cameras"]]
    camera = a.camera or cameras[0]
    if camera not in cameras:
        p.error(f"unknown camera {camera!r}; the world has {cameras}")
    size = {"fps": a.fps, "width": a.width, "height": a.height}
    if a.wedit:
        info = export_wedit(world, sha256_file(a.world), a.out, camera, as_zip=a.zip, **size)
    else:
        info = export_mp4(world, a.out, camera, **size)
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
