# Job: deflicker and stabilize generated video across segment boundaries
# Fixes light flicker and character structure inconsistency at segment joins
#   python -m tools.kaggle_runner run training/kaggle/jobs/video_deflicker.py --scene data:vacecast --wait
import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

from formats.wscript import read_wscript
from stages.generate.bible import expand
from stages.generate.portraits import appearance_words

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
py = Path("/tmp/deflicker/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/deflicker"], check=True)
r = subprocess.run(
    [*UV, "pip", "install", "--python", str(py), "opencv-python-headless", "numpy"],
    capture_output=True, text=True)
log(f"deflicker env install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

# Find input video
INPUT_VIDEO = Path("/kaggle/input/datasets/wb3none/worldreel-data-vacecast")
video_path = None
for p in INPUT_VIDEO.rglob("consistency_long.mp4"):
    video_path = str(p)
    break
if video_path is None:
    # Fallback: look for any mp4
    for p in INPUT_VIDEO.rglob("*.mp4"):
        video_path = str(p)
        break

log(f"input video: {video_path}")

args = {
    "video_path": video_path,
    "out": str(OUT),
    "problem_ranges": [[8, 9], [13, 14], [17, 18]],
}
step = WORK / "step_deflicker.py"
step.write_text(
    textwrap.dedent("""
    import cv2, numpy as np, json, sys
    from pathlib import Path

    a = json.load(open(sys.argv[1]))
    log = print
    video_path = a["video_path"]
    out_dir = Path(a["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"Input: {width}x{height}, {fps} fps, {total_frames} frames, {total_frames/fps:.2f}s")

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    log(f"Read {len(frames)} frames")

    # Detect segment boundaries (where frame-to-frame diff is high)
    diffs = []
    for i in range(1, len(frames)):
        diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
        diffs.append(diff)

    # Find boundaries: diffs > 3.0 are likely segment joins
    boundaries = []
    for i, d in enumerate(diffs):
        if d > 3.0:
            boundaries.append((i, d))
            log(f"Boundary detected at frame {i} (t={i/fps:.2f}s): diff={d:.2f}")

    # Apply temporal smoothing: blend across boundaries
    smoothed = frames.copy()
    for b_idx, (b, diff) in enumerate(boundaries):
        # Blend frames before and after boundary
        blend_range = 5  # frames to blend on each side
        for offset in range(-blend_range, blend_range + 1):
            idx1 = b + offset
            idx2 = b + offset + 1
            if 0 <= idx1 < len(frames) and 0 <= idx2 < len(frames):
                # Weight: closer to boundary = more weight from frame after boundary
                w = (blend_range + 1 - abs(offset)) / (2 * blend_range + 1)
                w = max(0, min(1, w))
                smoothed[idx2] = (smoothed[idx2].astype(float) * w +
                                   smoothed[idx1].astype(float) * (1 - w)).astype(np.uint8)

    # Apply histogram matching across segments to maintain color consistency
    # Use first frame as reference
    ref_frame = frames[0]
    ref_hsv = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2HSV)

    for i in range(1, len(smoothed)):
        curr_hsv = cv2.cvtColor(smoothed[i], cv2.COLOR_BGR2HSV)
        # Match histogram of V channel (brightness) to reference
        for ch in [1, 2]:  # S and V channels
            ref_ch = ref_hsv[:, :, ch]
            curr_ch = curr_hsv[:, :, ch]
            # Simple gain matching
            ref_mean = np.mean(ref_ch)
            curr_mean = np.mean(curr_ch)
            if curr_mean > 0:
                gain = ref_mean / curr_mean
                gain = np.clip(gain, 0.8, 1.2)  # Limit correction
                curr_hsv[:, :, ch] = np.clip(curr_ch * gain, 0, 255).astype(np.uint8)
        smoothed[i] = cv2.cvtColor(curr_hsv, cv2.COLOR_HSV2BGR)

    # Write output
    out_path = out_dir / "consistency_deflickered.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    for frame in smoothed:
        writer.write(frame)
    writer.release()
    log(f"Output written to {out_path}")

    # Verify: compute frame-to-frame diffs after smoothing
    new_diffs = []
    for i in range(1, len(smoothed)):
        diff = np.mean(np.abs(smoothed[i].astype(float) - smoothed[i-1].astype(float)))
        new_diffs.append(diff)

    # Check boundary diffs after smoothing
    log("\\nAfter smoothing, boundary diffs:")
    for b, old_diff in boundaries:
        new_diff = new_diffs[b-1] if b-1 < len(new_diffs) else 0
        improvement = old_diff - new_diff
        log(f"  Frame {b}: {old_diff:.2f} -> {new_diff:.2f} (improvement: {improvement:.2f})")

    avg_before = np.mean(diffs)
    avg_after = np.mean(new_diffs)
    log(f"\\nAverage frame-to-frame diff: {avg_before:.2f} -> {avg_after:.2f}")
    log(f"Improvement: {((avg_before - avg_after) / avg_before * 100):.1f}%")
    """),
    encoding="utf-8",
)
(WORK / "deflicker_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "deflicker_args.json")],
                   capture_output=True, text=True, timeout=6 * 3600, env=env)
(OUT / "deflicker.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-8000:] if r.returncode else r.stdout[-20000:], encoding="utf-8")
log(f"deflicker rc={r.returncode}")
log("done")