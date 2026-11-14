# Job: Kaggle smoke test for output-path plumbing and dataset mounts.
#
# Why this exists: the face-consistency job reported success (rc=0) for several runs while
# every artefact landed in /tmp/kaggle_out instead of /kaggle/working/out. Kaggle archives
# ONLY /kaggle/working/out, so the MP4 was silently lost and the downloaded folder looked
# empty (D-9.10). This job is CPU-only and finishes in under a minute, so it can guard the
# plumbing without consuming the 30 h/week GPU quota.
#
# Usage: python -m tools.kaggle_runner run training/kaggle/jobs/output_paths_test.py \
#            --scene data:facemodels --scene data:vacecast --cpu
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- Resolve the output directory exactly the way production jobs do ---------------
OUT_DIR = Path(os.environ.get("OUT", "/kaggle/working/out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The bootstrap must export OUT; if it does not, a job would silently fall back to /tmp.
BOOTSTRAP_EXPORTED = os.environ.get("OUT")

checks: list[dict[str, object]] = []


def check(name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    log(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")


def find(pattern: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob(pattern)):
            return p
    return None


INPUT_ROOT = Path("/kaggle/input")

# 1. The bootstrap must export OUT and it must be the archived directory.
EXPECTED_OUT = "/kaggle/working/out"
check(
    "bootstrap exports OUT",
    BOOTSTRAP_EXPORTED is not None,
    f"OUT={BOOTSTRAP_EXPORTED!r}",
)
check(
    "OUT is inside /kaggle/working/out",
    str(OUT_DIR) == EXPECTED_OUT,
    f"resolved OUT_DIR={OUT_DIR}",
)
check(
    "no stray /tmp/kaggle_out in use",
    str(OUT_DIR) != "/tmp/kaggle_out",
    f"OUT_DIR={OUT_DIR}",
)

# 2. The dataset-mounted face models must be present (never fetch from GitHub).
yunet = find("face_detection_yunet_2023mar.onnx", [INPUT_ROOT])
sface = find("face_recognition_sface_2021dec.onnx", [INPUT_ROOT])
check("YuNet model mounted", yunet is not None, str(yunet))
check("SFace model mounted", sface is not None, str(sface))

# 3. The portrait dataset must be present.
portrait = find("C01_0.png", [INPUT_ROOT])
check("portrait mounted", portrait is not None, str(portrait))

# 4. A subprocess launched with OUT must write into OUT_DIR (the exact regression).
marker = OUT_DIR / "output_paths_test_marker.txt"
child = (
    "import os,sys;"
    "from pathlib import Path;"
    "p=Path(os.environ['OUT'])/'output_paths_test_child.txt';"
    "p.write_text('child wrote here');"
    "print(p)"
)
proc = subprocess.run(
    [sys.executable, "-c", child],
    capture_output=True,
    text=True,
    env={**os.environ, "OUT": str(OUT_DIR)},
)
child_file = OUT_DIR / "output_paths_test_child.txt"
check(
    "child process writes into OUT_DIR",
    proc.returncode == 0 and child_file.exists(),
    f"rc={proc.returncode} stdout={proc.stdout.strip()!r}",
)

# 5. Real artefacts must be readable back after being written.
marker.write_text("worldreel output path test\n", encoding="utf-8")
check(
    "marker readable after write",
    marker.exists() and marker.read_text(encoding="utf-8").startswith("worldreel"),
    f"{marker} ({marker.stat().st_size} bytes)" if marker.exists() else "missing",
)

# 6. A real MP4 must be producible and land in OUT_DIR (best effort: needs an encoder).
mp4 = OUT_DIR / "output_paths_test.mp4"
mp4_note = "no encoder available — skipped"
try:
    import cv2
    import numpy as np

    # OpenCV 5 renamed VideoWriter_fourcc -> VideoWriter.fourcc; support both.
    fourcc = getattr(cv2, "VideoWriter_fourcc", None)
    if fourcc is None:
        fourcc = cv2.VideoWriter.fourcc

    writer = cv2.VideoWriter(str(mp4), fourcc(*"mp4v"), 16.0, (64, 64))
    if writer.isOpened():
        for i in range(16):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :] = (40 + i * 5, 80, 120)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        size = mp4.stat().st_size if mp4.exists() else 0
        mp4_note = f"cv2 mp4v {size} bytes"
    else:
        mp4_note = "cv2 VideoWriter refused mp4v — skipped"
except Exception as exc:  # pragma: no cover - depends on the Kaggle image
    mp4_note = f"encoder unavailable ({exc}) — skipped"
check("MP4 produced in OUT_DIR", mp4.exists() and mp4.stat().st_size > 0, mp4_note)

# 7. Nothing important may live outside OUT_DIR.
strays = [p for p in Path("/tmp/kaggle_out").glob("*")] if Path("/tmp/kaggle_out").exists() else []
check("no artefacts stranded in /tmp/kaggle_out", not strays, f"{len(strays)} stray file(s)")

passed = sum(1 for c in checks if c["passed"])
report = {
    "out_dir": str(OUT_DIR),
    "bootstrap_exported_out": BOOTSTRAP_EXPORTED,
    "checks": checks,
    "passed": passed,
    "total": len(checks),
    "ok": passed == len(checks),
}
(OUT_DIR / "output_paths_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log(f"output path test: {passed}/{len(checks)} checks passed")

# Also drop a copy in /kaggle/working so it is visible even if OUT moves again.
with contextlib.suppress(OSError):
    shutil.copy2(OUT_DIR / "output_paths_test.json", Path("/kaggle/working/output_paths_test.json"))

if not report["ok"]:
    failed = [c["name"] for c in checks if not c["passed"]]
    log(f"FATAL: failing checks: {failed}")
    raise SystemExit(1)

log("OUTPUT PATH TEST PASSED")
