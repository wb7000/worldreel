"""D-8.0 face marks on the MPFB humans (A-14 v2, D-8.11).

    python -m eval.human_faces runs/humans --choice runs/kaggle/portraits_run3_choice.json \
        --portraits runs/kaggle/portraits_run3_merged

Reads runs/humans/<cid>/views/d{d}_y{yaw}.png (from `tools/blender/views.py`); the same marks
as `eval.face_metrics`: detection on eligible frames (head >= 40 px, yaw <= 30), identity by
nearest portrait embedding, distinctness = cosine of mean rendered embeddings < 0.363.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from eval.face_metrics import SAME_PERSON, Faces, imread, portrait_face

HEAD_M = 0.23  # MPFB adult head height, chin to crown (approximate)
VIEW = re.compile(r"d(\d+)_y(\d+)\.png")


def measure(humans: Path, portraits: dict[str, Path]) -> dict[str, Any]:
    faces = Faces()
    refs = {c: portrait_face(faces, p)[2] for c, p in portraits.items()}
    f_px = 240 / math.tan(math.radians(25))
    rows: list[dict[str, Any]] = []
    rendered: dict[str, list[np.ndarray]] = {}
    for cid in sorted(portraits):
        rendered[cid] = []
        for p in sorted((humans / cid / "views").glob("*.png")):
            m = VIEW.fullmatch(p.name)
            if not m:
                continue
            d, yaw = float(m.group(1)), int(m.group(2))
            img = imread(p)
            face = faces.find(img)
            if face is not None:
                rendered[cid].append(faces.embed(img, face))
            head_px = HEAD_M * f_px / d
            rows.append({"cid": cid, "d": d, "yaw": yaw, "found": face is not None,
                         "eligible": head_px >= 40 and yaw <= 30})  # fmt: skip
    ok = total = 0
    for cid, embs in rendered.items():
        for e in embs:
            sims = {c: float(e @ r) for c, r in refs.items()}
            ok += max(sims, key=lambda c: sims[c]) == cid
            total += 1
    elig = [r for r in rows if r["eligible"]]
    det = sum(bool(r["found"]) for r in elig) / max(1, len(elig))
    means = {c: np.mean(e, 0) / np.linalg.norm(np.mean(e, 0)) for c, e in rendered.items() if e}
    pairs = {f"{a}-{b}": round(float(means[a] @ means[b]), 3)
             for a, b in combinations(sorted(means), 2)}  # fmt: skip
    own = {c: round(float(means[c] @ refs[c]), 3) for c in means}
    return {
        "detection_rate": round(det, 3),
        "eligible_frames": len(elig),
        "identity_accuracy": round(ok / max(1, total), 3),
        "faces_identified": total,
        "own_portrait_cosine": own,
        "distinct_pairs": pairs,
        "pass": {
            "detection": det >= 0.5,
            "identity": ok / max(1, total) >= 0.9,
            "distinctness": bool(pairs) and max(pairs.values()) < SAME_PERSON,
        },
        "frames": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.human_faces")
    ap.add_argument("humans", type=Path)
    ap.add_argument("--choice", type=Path, required=True)
    ap.add_argument("--portraits", type=Path, required=True)
    a = ap.parse_args(argv)
    choice = json.loads(a.choice.read_text())
    res = measure(a.humans, {c: a.portraits / f for c, f in choice.items()})
    (a.humans / "human_faces.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
