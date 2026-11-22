"""Join rendered clips into one timeline WITHOUT re-encoding them.

The delivery render is split over several Kaggle jobs so no job is near the
6 h kernel timeout (D-9.28): three segments in one job, the remaining two in
another. Each job exports its own clip, and the two clips have to become one
240-frame timeline before the local chain runs.

Two ways to join, and the difference matters:

* **stream copy** (`-c copy`): the H.264 bits of both clips are copied into one
  container. Nothing is decoded, nothing is re-encoded, so the joined file is
  bit-identical to the renders -- the only generation loss is the one the
  render itself already had. This is the default when the clips share their
  codec parameters.
* **lossless re-encode** (FFV1 in Matroska): the clips are decoded and written
  with a lossless codec, so the joined file is pixel-identical to the decoded
  renders. Used when the parameters differ (a different resolution, a
  different frame rate, a different pixel format) and a stream copy would
  produce a broken file. A lossy re-encode is never used here: D-9.22/23
  forbid passing a render through a lossy intermediate.

Usage:
    python tools/join_clips.py OUT.mp4 CLIP_A.mp4 CLIP_B.mp4 [...]
    python tools/join_clips.py --list OUT.mp4 CLIP_A.mp4 CLIP_B.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.toolpaths import require_tool


def probe(path: Path) -> dict[str, str]:
    """Stream parameters of a video, as ffprobe reports them."""
    ffprobe = require_tool("ffprobe")
    out = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,time_base",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    streams = json.loads(out).get("streams", [])
    if not streams:
        raise SystemExit(f"{path}: no video stream")
    return {k: str(v) for k, v in streams[0].items()}


def count_frames(path: Path) -> int:
    ffprobe = require_tool("ffprobe")
    out = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(out or 0)


STREAM_KEYS = ("codec_name", "width", "height", "pix_fmt", "r_frame_rate", "time_base")


def same_parameters(params: list[dict[str, str]]) -> bool:
    """True when every clip can be joined with a pure stream copy.

    A stream copy concatenates the encoded bits, so it is only valid when the
    clips agree on the codec and on every parameter the decoder uses to
    interpret the stream. If they disagree, the join falls back to a lossless
    re-encode instead of producing a file that decodes badly.
    """
    if not params:
        return False
    return all(all(p.get(key) == params[0].get(key) for key in STREAM_KEYS) for p in params)


def join(clips: list[Path], output: Path, force_lossless: bool) -> str:
    ffmpeg = require_tool("ffmpeg")
    params = [probe(c) for c in clips]
    same = same_parameters(params)

    output.parent.mkdir(parents=True, exist_ok=True)
    if same and not force_lossless:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            for clip in clips:
                escaped = str(clip.resolve()).replace("'", "'\\''")
                fh.write(f"file '{escaped}'\n")
            listing = Path(fh.name)
        cmd = [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(output),
        ]
        mode = "stream copy (no re-encode)"
    else:
        cmd = [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
        ]
        for clip in clips:
            cmd += ["-i", str(clip)]
        chain = "".join(f"[{i}:v]" for i in range(len(clips)))
        cmd += [
            "-filter_complex",
            f"{chain}concat=n={len(clips)}:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            str(output),
        ]
        mode = "lossless FFV1 re-encode (stream parameters differ)"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"ffmpeg failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
        )
    if "listing" in locals():
        listing.unlink(missing_ok=True)
    return mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join rendered clips without re-encoding")
    parser.add_argument("output", type=Path, help="Output video path")
    parser.add_argument("clips", type=Path, nargs="+", help="Clips to join, in order")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only print the parameters and frame counts, do not join",
    )
    parser.add_argument(
        "--lossless",
        action="store_true",
        help="Force the FFV1 re-encode even when a stream copy is possible",
    )
    args = parser.parse_args(argv)

    clips = [c for c in args.clips]
    for clip in clips:
        if not clip.exists():
            raise SystemExit(f"missing clip: {clip}")
        params = probe(clip)
        print(
            f"{clip.name}: {params['codec_name']} {params['width']}x{params['height']} "
            f"{params['pix_fmt']} {params['r_frame_rate']} tb={params['time_base']} "
            f"frames={count_frames(clip)}"
        )
    if args.list:
        return 0

    mode = join(clips, args.output, args.lossless)
    print(f"joined {len(clips)} clips -> {args.output} ({mode})")
    print(f"frames: {count_frames(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
