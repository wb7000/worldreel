# Job: deflicker generated video with per-frame gain correction (no blur, no accumulation)
# Fixes segment boundary flicker by adjusting each frame's brightness independently to match
# neighbors, without blending frames.
#   python -m tools.kaggle_runner run training/kaggle/jobs/video_deflicker_v3.py --scene data:consistent-video --wait
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
py = Path("/tmp/deflicker3/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/deflicker3"], check=True)
r = subprocess.run(
    [*UV, "pip", "install", "--python", str(py), "opencv-python-headless", "numpy"],
    capture_output=True, text=True)
log(f"deflicker v3 env install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

# Find input video in all mounted Kaggle datasets
INPUT = Path("/kaggle/input")
video_path = None
for p in INPUT.rglob("consistency_long.mp4"):
    video_path = str(p)
    break
if video_path is None:
    for p in INPUT.rglob("*.mp4"):
        video_path = str(p)
        break

log(f"input video: {video_path}")

args = {
    "video_path": video_path,
    "out": str(OUT),
}
step = WORK / "step_deflicker_v3.py"
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

    # Compute per-frame mean brightness (V channel in HSV)
    frame_brightness = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        frame_brightness.append(float(np.mean(hsv[:, :, 2])))

    # Detect segment boundaries where brightness jumps significantly
    brightness_diffs = []
    for i in range(1, len(frame_brightness)):
        diff = abs(frame_brightness[i] - frame_brightness[i - 1])
        brightness_diffs.append(diff)

    boundary_indices = []
    for i, d in enumerate(brightness_diffs):
        if d > 15.0:  # brightness jump threshold
            boundary_indices.append(i)  # frame i is where jump happens
            log(f"Brightness jump at frame {i} (t={i / fps:.2f}s): diff={d:.2f}")

    # Compute target brightness for each frame
    # Frames not near boundaries keep their own brightness
    # Frames near boundaries smoothly transition between the two sides
    target_brightness = np.array(frame_brightness, dtype=np.float32).copy()
    blend_range = 6  # frames on each side of boundary to blend

    for b in boundary_indices:
        # b is the frame where jump happens (i.e., diff between b-1 and b)
        brightness_before = frame_brightness[b - 1]
        brightness_after = frame_brightness[b]

        # Smooth transition from brightness_before to brightness_after over blend_range frames
        for offset in range(-blend_range, blend_range + 1):
            idx = b + offset
            if 0 <= idx < len(frames):
                # Weight: 0 = keep brightness_before, 1 = use brightness_after
                w = (offset + blend_range) / (2 * blend_range)
                w = max(0.0, min(1.0, w))
                target = brightness_before * (1 - w) + brightness_after * w
                target_brightness[idx] = target

    # Apply per-frame gain: each frame scaled independently to match target
    smoothed = []
    for i, f in enumerate(frames):
        current = frame_brightness[i]
        target = target_brightness[i]
        if current > 1.0:
            gain = target / current
            gain = np.clip(gain, 0.6, 1.6)  # limit correction to avoid over-correction
            adjusted = np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        else:
            adjusted = f
        smoothed.append(adjusted)

    # Write output
    out_path = out_dir / "consistency_deflickered_v3.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    for frame in smoothed:
        writer.write(frame)
    writer.release()
    log(f"Output written to {out_path}")

    # Verify: compute frame-to-frame diffs after smoothing
    new_diffs = []
    for i in range(1, len(smoothed)):
        diff = np.mean(np.abs(smoothed[i].astype(float) - smoothed[i - 1].astype(float)))
        new_diffs.append(diff)

    log("\\nBoundary diffs after deflicker v3:")
    for b in boundary_indices:
        old_diff = brightness_diffs[b - 1]
        new_diff = new_diffs[b - 1] if b - 1 < len(new_diffs) else 0
        improvement = old_diff - new_diff
        log(f"  Frame {b}: {old_diff:.2f} -> {new_diff:.2f} (improvement: {improvement:.2f})")

    avg_before = np.mean(brightness_diffs)
    avg_after = np.mean(new_diffs)
    log(f"\\nAverage frame-to-frame diff: {avg_before:.2f} -> {avg_after:.2f}")
    log(f"Improvement: {((avg_before - avg_after) / avg_before * 100):.1f}%")
    """),
    encoding="utf-8",
)
(WORK / "deflicker_v3_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "deflicker_v3_args.json")],
                   capture_output=True, text=True, timeout=6 * 3600, env=env)
(OUT / "deflicker_v3.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-8000:] if r.returncode else r.stdout[-20000:], encoding="utf-8")
log(f"deflicker v3 rc={r.returncode}")
log("done")