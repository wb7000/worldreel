"""Choose one portrait per character from the candidates (A-14): only candidates with exactly
one clear face (YuNet score >= 0.9) qualify; among them the combination whose SFace embeddings
are most distinct (smallest largest cosine between characters) is kept — the same rule as the
designed voices (`best_reference_set`, D-8.2b).

    python -m eval.choose_portraits runs/kaggle/portraits/out/portraits --out choice.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from eval.face_metrics import Faces
from stages.generate.voices import best_reference_set


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.choose_portraits")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    faces = Faces()
    report: dict[str, dict[str, object]] = {}
    cands: dict[str, list[tuple[str, np.ndarray]]] = {}
    for p in sorted(a.folder.glob("C*_*.png")):
        cid = p.stem.split("_")[0]
        img = cv2.imread(str(p))
        if img is None:
            continue
        faces.det.setInputSize((img.shape[1], img.shape[0]))
        _, found = faces.det.detect(img)
        n = 0 if found is None else int((found[:, -1] >= 0.9).sum())
        score = 0.0 if found is None else float(found[:, -1].max())
        report[p.name] = {"faces_at_0.9": n, "best_score": round(score, 3)}
        if n == 1 and found is not None:
            best = found[int(np.argmax(found[:, -1]))]
            cands.setdefault(cid, []).append((p.name, faces.embed(img, best)))
    usable = {c: v for c, v in cands.items() if v}
    choice_idx, worst = best_reference_set(
        {c: np.stack([e for _, e in v]) for c, v in usable.items()}
    )
    choice = {c: usable[c][i][0] for c, i in choice_idx.items()}
    a.out.write_text(json.dumps(choice, indent=2))
    summary = {"candidates": report, "choice": choice, "max_cosine_between_characters": worst}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
