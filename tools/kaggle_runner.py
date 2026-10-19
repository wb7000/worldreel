"""Run WorldReel jobs on Kaggle GPUs from the command line (brief §2: training moves machines
through configuration, not code changes).

    python -m tools.kaggle_runner push-code                  # code → private dataset
    python -m tools.kaggle_runner push-scene <scene dir>     # a synthetic scene → private dataset
    python -m tools.kaggle_runner push-video <id> --name N --config <toml>  # an ingested video
    python -m tools.kaggle_runner push-data <folder> --name N  # any prepared data folder
    python -m tools.kaggle_runner run <job.py> --scene data:N  # attach it to a job
    python -m tools.kaggle_runner run <job.py> [--scene NAME] [--wait]
    python -m tools.kaggle_runner status <job.py>
    python -m tools.kaggle_runner fetch <job.py>            # outputs → runs/kaggle/<job>/

What gets uploaded:
- code: only files tracked by git under the Python packages, `spec/` and `pyproject.toml`
  (never `data/`, never anything git-ignored, never credentials), as one zip;
- scenes: the folder you name, as one zip.
Everything is uploaded as PRIVATE Kaggle datasets/kernels of the logged-in account.

Each job is a Python script in `training/kaggle/jobs/`. It runs on Kaggle as a script kernel
with a T4 GPU and internet (for `pip install`), after `training/kaggle/bootstrap.py` has
unpacked the code. Outputs go to `/kaggle/working/out/`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from tools.toolpaths import REPO_ROOT

KAGGLE = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
CODE_DATASET = "worldreel-code"
CODE_ZIP = "worldreel_code.zip"
CODE_DIRS = ("formats", "pipeline", "stages", "tools", "eval", "training", "spec")
JOBS_DIR = REPO_ROOT / "training" / "kaggle" / "jobs"
BOOTSTRAP = REPO_ROOT / "training" / "kaggle" / "bootstrap.py"
RUNS_DIR = REPO_ROOT / "runs" / "kaggle"
MACHINE_SHAPE = "NvidiaTeslaT4"


def kaggle(*args: str, capture: bool = True) -> str:
    cmd = [str(KAGGLE), *args]
    proc = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout or ""


def username() -> str:
    for line in kaggle("config", "view").splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("no Kaggle username configured (see ~/.kaggle)")


def tracked_code_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", *CODE_DIRS, "pyproject.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if line.strip()]


def _upload(folder: Path, slug: str, title: str, message: str) -> None:
    """Create the private dataset, or add a new version if it already exists."""
    user = username()
    meta = {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    try:
        kaggle("datasets", "status", f"{user}/{slug}")
        exists = True
    except RuntimeError:
        exists = False
    if exists:
        kaggle("datasets", "version", "-p", str(folder), "-m", message)
    else:
        kaggle("datasets", "create", "-p", str(folder))
    print(f"uploaded {user}/{slug} ({'new version' if exists else 'created, private'})")
    _wait_ready(f"{user}/{slug}")


def _wait_ready(ref: str, timeout_s: float = 600.0) -> None:
    """Wait until Kaggle has processed the upload: a job started earlier can mount the previous
    version (happened in D-8.2). Assumption: `datasets status` reports the newest version."""
    deadline = time.time() + timeout_s
    time.sleep(10)
    while time.time() < deadline:
        state = kaggle("datasets", "status", ref).strip().lower()
        if "ready" in state:
            time.sleep(30)  # margin for the version to reach the kernel mounts
            print(f"{ref}: ready")
            return
        time.sleep(15)
    print(f"{ref}: not ready after {timeout_s:.0f}s (status: {state!r})")


def push_code(message: str) -> None:
    files = tracked_code_files()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / CODE_ZIP
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, f.relative_to(REPO_ROOT).as_posix())
        print(f"code archive: {len(files)} tracked files, {archive.stat().st_size / 1e6:.1f} MB")
        _upload(Path(tmp), CODE_DATASET, "WorldReel code", message)


def push_scene(scene: Path, name: str) -> None:
    if not (scene / "gt.json").exists():
        raise SystemExit(f"{scene} is not a synthetic scene (no gt.json)")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "scene.zip"
        shutil.make_archive(str(archive.with_suffix("")), "zip", scene)
        print(f"scene archive: {archive.stat().st_size / 1e6:.1f} MB")
        _upload(Path(tmp), f"worldreel-scene-{name}", f"WorldReel scene {name}", "scene upload")


def push_data(folder: Path, name: str) -> None:
    """A prepared data folder (e.g. a training set built from data/) as one private dataset."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "data.zip"
        shutil.make_archive(str(archive.with_suffix("")), "zip", folder)
        print(f"data archive: {archive.stat().st_size / 1e6:.1f} MB")
        _upload(Path(tmp), f"worldreel-data-{name}", f"WorldReel data {name}", "data upload")


def push_video(store_root: Path, file_id: str, name: str, config: Path) -> None:
    """Upload one ingested video (by data id) and the pipeline config to run on it, as a
    PRIVATE dataset. Only the video file, its manifest row and the config are uploaded."""
    from stages.ingest.store import DataStore

    store = DataStore(store_root)
    row = store.get(file_id)
    if row.status != "active" or row.media_type != "video":
        raise SystemExit(f"{file_id} is not an active video")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        shutil.copy(store.root / row.path, folder / f"video{Path(row.path).suffix}")
        shutil.copy(config, folder / "pipeline.toml")
        (folder / "item.json").write_text(
            json.dumps(
                {
                    "id": row.id,
                    "sha256": row.hash,
                    "source": row.source,
                    "license": row.license,
                    "upload_rights": row.upload_rights,
                }
            ),
            encoding="utf-8",
        )
        _upload(folder, f"worldreel-video-{name}", f"WorldReel video {name}", "video upload")


def kernel_slug(job: Path) -> str:
    return f"worldreel-{job.stem.replace('_', '-')}"


def run(
    job: Path, scenes: list[str], wait: bool, any_gpu: bool = False, requeue_after: float = 0
) -> None:
    """Push the job. With `requeue_after` minutes > 0, a job still QUEUED after that long is
    deleted and pushed again (observed on Kaggle: a fresh push often starts at once while the
    queued one waits for hours; DECISIONS D-3b.3)."""
    if not job.exists():
        job = JOBS_DIR / job.name
    _push(job, scenes, any_gpu)
    if requeue_after > 0:
        started = time.time()
        while "queued" in status(job).lower():
            if time.time() - started > requeue_after * 60:
                print(
                    f"still queued after {requeue_after:g} min: deleting and pushing again",
                    flush=True,
                )
                kaggle("kernels", "delete", f"{username()}/{kernel_slug(job)}", "-y")
                _push(job, scenes, any_gpu)
                started = time.time()
            time.sleep(60)
    if wait:
        wait_for(job)


def _push(job: Path, scenes: list[str], any_gpu: bool) -> None:
    user = username()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        # One self-contained script: bootstrap (unpack code, set paths) + the job body.
        script = (
            BOOTSTRAP.read_text(encoding="utf-8")
            + "\n\n# ---- job: "
            + job.name
            + "\n"
            + job.read_text(encoding="utf-8")
        )
        (folder / "job.py").write_text(script, encoding="utf-8")
        meta = {
            "id": f"{user}/{kernel_slug(job)}",
            "title": kernel_slug(job),
            "code_file": "job.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            **({} if any_gpu else {"machine_shape": MACHINE_SHAPE}),
            "dataset_sources": [f"{user}/{CODE_DATASET}"]
            + [
                f"{user}/worldreel-video-{s[6:]}"
                if s.startswith("video:")
                else f"{user}/worldreel-data-{s[5:]}"
                if s.startswith("data:")
                else f"{user}/worldreel-scene-{s}"
                for s in scenes
            ],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(kaggle("kernels", "push", "-p", str(folder)).strip())


def status(job: Path) -> str:
    out = kaggle("kernels", "status", f"{username()}/{kernel_slug(job)}").strip()
    return out


def wait_for(job: Path, poll_s: int = 60, max_hours: float = 12.0) -> None:
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        s = status(job)
        print(time.strftime("%H:%M:%S"), s, flush=True)
        if any(word in s.lower() for word in ("complete", "error", "cancel")):
            return
        time.sleep(poll_s)


def fetch(job: Path) -> Path:
    target = RUNS_DIR / job.stem
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    kaggle("kernels", "output", f"{username()}/{kernel_slug(job)}", "-p", str(target))
    print(f"outputs in {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.kaggle_runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("push-code")
    c.add_argument("-m", "--message", default="code update")
    d = sub.add_parser("push-data")
    d.add_argument("folder", type=Path)
    d.add_argument("--name", required=True)
    s = sub.add_parser("push-scene")
    s.add_argument("scene", type=Path)
    s.add_argument("--name", required=True)
    v = sub.add_parser("push-video")
    v.add_argument("id")
    v.add_argument("--name", required=True)
    v.add_argument("--config", type=Path, required=True)
    v.add_argument("--root", type=Path, default=REPO_ROOT / "data")
    r = sub.add_parser("run")
    r.add_argument("job", type=Path)
    r.add_argument("--scene", action="append", default=[])
    r.add_argument("--video", action="append", default=[])
    r.add_argument("--wait", action="store_true")
    r.add_argument(
        "--any-gpu", action="store_true", help="let Kaggle pick the GPU (T4 x2 queues can be long)"
    )
    r.add_argument(
        "--requeue-after",
        type=float,
        default=20.0,
        help="minutes queued before re-pushing (0 = never)",
    )
    for name in ("status", "fetch", "wait"):
        sub.add_parser(name).add_argument("job", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "push-code":
        push_code(a.message)
    elif a.cmd == "push-data":
        push_data(a.folder, a.name)
    elif a.cmd == "push-scene":
        push_scene(a.scene, a.name)
    elif a.cmd == "push-video":
        push_video(a.root, a.id, a.name, a.config)
    elif a.cmd == "run":
        datasets = a.scene + [f"video:{n}" for n in a.video]
        run(a.job, datasets, a.wait, a.any_gpu, a.requeue_after)
    elif a.cmd == "status":
        print(status(a.job))
    elif a.cmd == "wait":
        wait_for(a.job)
    else:
        fetch(a.job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
