"""Choose one portrait per character from the candidates (A-14): only candidates with exactly
one clear face (YuNet score >= 0.9) qualify; among them the combination whose SFace embeddings
are most distinct (smallest largest cosine between characters) is kept — the same rule as the
designed voices (`best_reference_set`, D-8.2b).

With `--rendered` each qualifying candidate is first built into its avatar and rendered
(`eval.face_metrics.views`); the mean embedding of the rendered faces replaces the portrait's, so
the set is chosen by how distinct the characters look in the world, not on paper (D-8.10: the
portrait set at 0.346 still rendered Ade and Tomas at 0.516).

    python -m eval.choose_portraits runs/kaggle/portraits/out/portraits --out c.json [--rendered]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from eval.face_metrics import Faces, box_of, views
from formats.wscript import read_wscript
from stages.generate.avatars import build_avatar
from stages.generate.bible import expand
from stages.generate.voices import best_reference_set
from tools.make_sample_wscript import OUT


def rendered_embedding(
    faces: Faces, cid: str, img: np.ndarray, face: np.ndarray
) -> np.ndarray | None:
    """Mean SFace embedding of the candidate's avatar over the eval views; None if no face found."""
    c = read_wscript(OUT).header["dictionary"]["characters"][cid]
    ident = expand(int(c["identity"]["seed"]), str(c["identity"]["requested"]))
    rgb: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    av = build_avatar(ident, rgb, box_of(face))
    embs = []
    for bgr, _, _ in views(av):
        f = faces.find(bgr)
        if f is not None:
            embs.append(faces.embed(bgr, f))
    if not embs:
        return None
    m = np.mean(embs, 0)
    out: np.ndarray = m / np.linalg.norm(m)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.choose_portraits")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rendered", action="store_true", help="choose by rendered-avatar faces")
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
            emb = (
                rendered_embedding(faces, cid, img, best) if a.rendered else faces.embed(img, best)
            )
            if emb is not None:
                cands.setdefault(cid, []).append((p.name, emb))
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
