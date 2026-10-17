# WorldReel Kaggle bootstrap (prepended to every job by tools/kaggle_runner.py).
# Locates the uploaded code under /kaggle/input (Kaggle may keep our zip or unpack it), copies
# it to /kaggle/working/worldreel, puts it on sys.path, and prepares /kaggle/working/out.
import json as _json
import os as _os
import shutil as _shutil
import sys as _sys
import time as _time
import zipfile as _zipfile
from pathlib import Path as _Path

_T0 = _time.time()
INPUT = _Path("/kaggle/input")
WORK = _Path("/kaggle/working")
CODE = WORK / "worldreel"
OUT = WORK / "out"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{_time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    with (OUT / "log.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def save_json(name: str, data: object) -> None:
    (OUT / name).write_text(_json.dumps(data, indent=2, default=str), encoding="utf-8")


log("input tree: " + ", ".join(sorted(str(p.relative_to(INPUT)) for p in INPUT.glob("*/*"))[:40]))

if not CODE.exists():
    zips = sorted(INPUT.rglob("worldreel_code.zip"))
    if zips:
        with _zipfile.ZipFile(zips[0]) as _zf:
            _zf.extractall(CODE)
    else:
        marker = sorted(INPUT.rglob("formats/wreel.py"))
        if not marker:
            raise FileNotFoundError("WorldReel code not found under /kaggle/input")
        _shutil.copytree(marker[0].parent.parent, CODE)
_sys.path.insert(0, str(CODE))
_os.chdir(CODE)
log(f"code ready at {CODE}")


def find_scenes() -> list[_Path]:
    """Every attached synthetic scene (zipped or already unpacked by Kaggle), as folders
    named after their dataset (e.g. worldreel-scene-seed12)."""
    scenes: list[_Path] = []
    for z in sorted(INPUT.rglob("scene.zip")):
        dest = WORK / "scenes" / z.parent.name
        if not dest.exists():
            with _zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
        scenes.append(dest)
    for gt in sorted(INPUT.rglob("gt.json")):
        # Name each scene after its dataset slug (Kaggle mounts datasets at varying depths, e.g.
        # /kaggle/input/datasets/<user>/<slug>/...), falling back to the folder holding gt.json.
        name = next(
            (p.name for p in gt.parents if p.name.startswith("worldreel-scene")), gt.parent.name
        )
        dest = WORK / "scenes" / name
        if not dest.exists():
            _shutil.copytree(gt.parent, dest)
        if dest not in scenes:
            scenes.append(dest)
    log(f"scenes: {[s.name for s in scenes]}")
    return scenes
