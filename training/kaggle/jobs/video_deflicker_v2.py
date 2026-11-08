# Job: deflicker generated video with per-frame color correction (NO blur)
# Fixes segment boundary flicker by adjusting each frame's color to match neighbors,
# without blending frames (which causes blur/ghosting).
#   python -m tools.kaggle_runner run training/kaggle/jobs/video_deflicker_v2.py --scene data:consistent-video --wait
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
py = Path("/tmp/deflicker2/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/deflicker2"], check=True)
r = subprocess.run(
    [*UV, "pip", "install", "--python", str(py), "opencv-python-headless", "numpy"],
    capture_output=True, text=True)
log(f"deflicker v2 env install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

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
    "problem_ranges": [[8, 9], [13, 14], [17, 18]],
}
step = WORK / "step_deflicker_v2.py"
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

    # Compute per-frame mean brightness in HSV V channel
    frame_means = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        frame_means.append(float(np.mean(hsv[:, :, 2])))

    # Detect segment boundaries where brightness jumps
    diffs = []
    for i in range(1, len(frame_means)):
        diffs.append(abs(frame_means[i] - frame_means[i-1]))

    boundaries = []
    for i, d in enumerate(diffs):
        if d > 15.0:  # brightness jump threshold
            boundaries.append((i, d))
            log(f"Brightness jump at frame {i} (t={i/fps:.2f}s): diff={d:.2f}")

    # Build smooth gain curve: 1.0 everywhere, transitions across boundaries
    gains = np.ones(len(frames), dtype=np.float32)

    for b_idx, (b, diff) in enumerate(boundaries):
        # Blend gain across boundary: frames before use frame b's level,
        # frames after use frame b+1's level, transition over blend_range frames
        blend_range = 8  # frames on each side
        target_before = frame_means[b]
        target_after = frame_means[b + 1] if b + 1 < len(frame_means) else frame_means[b]

        for offset in range(-blend_range, blend_range + 1):
            idx1 = b + offset      # frame before/at boundary
            idx2 = b + offset + 1  # frame after boundary
            if 0 <= idx1 < len(frames) and 0 <= idx2 < len(frames):
                # Weight: closer to boundary = more weight from target_after
                w = (blend_range + 1 - abs(offset)) / (2 * blend_range + 2)
                w = max(0.0, min(1.0, w))
                target = target_before * (1 - w) + target_after * w
                current = frame_means[idx2]
                if current > 0:
                    gain = target / current
                    gain = np.clip(gain, 0.7, 1.4)  # limit correction
                    gains[idx2] *= gain

    # Apply gains: multiply each channel by gain (color-preserving brightness adjustment)
    smoothed = []
    for i, f in enumerate(frames):
        gain = gains[i]
        adjusted = np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        smoothed.append(adjusted)

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

    log("\\nBoundary diffs after deflicker:")
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
(WORK / "deflicker_v2_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "deflicker_v2_args.json")],
                   capture_output=True, text=True, timeout=6 * 3600, env=env)
(OUT / "deflicker_v2.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-8000:] if r.returncode else r.stdout[-20000:], encoding="utf-8")
log(f"deflicker v2 rc={r.returncode}")
log("done")