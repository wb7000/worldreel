"""Face identity and distinctness on rendered avatars (D-8.0 thresholds, A-14).

    python -m eval.face_metrics --portraits DIR --choice choice.json --out DIR

Every character's avatar (bible + chosen portrait) is rendered alone with the offline renderer at
distances 1.0-3.5 m and yaw 0-40 degrees (854 x 480, fov 50, no fog — generated worlds, D-8.10).
YuNet finds faces, SFace embeds them. Reported against the D-8.0 marks:
- detection: faces found in >= 50 % of frames where the head is >= 40 px tall and yaw <= 30;
- identity: nearest character (by the four portraits' embeddings) correct for >= 90 % of faces;
- distinctness: cosine between characters' mean rendered-face embeddings < 0.363 for all pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from formats.wscript import read_wscript
from stages.export.render import CameraPose, Scene, render
from stages.generate.avatars import POINT_SIZE, Avatar, build_avatar
from stages.generate.bible import expand
from tools.make_sample_wscript import OUT

YUNET = "data/models/face_detection_yunet_2023mar.onnx"
SFACE = "data/models/face_recognition_sface_2021dec.onnx"
DISTANCES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
YAWS = (0, 10, 20, 30, 40)
W, H, FOV = 854, 480, 50.0
SAME_PERSON = 0.363  # OpenCV's SFace cosine threshold


class Faces:
    def __init__(self) -> None:
        self.det = cv2.FaceDetectorYN.create(YUNET, "", (W, H), 0.6)
        self.rec = cv2.FaceRecognizerSF.create(SFACE, "")

    def find(self, bgr: np.ndarray) -> np.ndarray | None:
        self.det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        best: np.ndarray = faces[int(np.argmax(faces[:, -1]))]
        return best

    def embed(self, bgr: np.ndarray, face: np.ndarray) -> np.ndarray:
        f = self.rec.feature(self.rec.alignCrop(bgr, face)).ravel()
        out: np.ndarray = f / np.linalg.norm(f)
        return out


def portrait_face(
    faces: Faces, path: Path
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]:
    """(RGB image, face box, embedding) of a portrait."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    bgr: np.ndarray = img
    face = faces.find(bgr)
    if face is None:
        raise ValueError(f"no face in {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), box_of(face), faces.embed(bgr, face)


def box_of(face: np.ndarray) -> tuple[float, float, float, float]:
    """YuNet row -> face box (x, y, w, h) in image pixels."""
    return (float(face[0]), float(face[1]), float(face[2]), float(face[3]))


def _scene(av: Avatar) -> Scene:
    empty = np.zeros((0, 3))
    return Scene(
        empty, np.zeros((0, 3, 3)), empty, np.zeros(0), np.zeros(0), empty, empty, np.zeros(0),
        np.zeros(0), [("a1", av.cloud, POINT_SIZE)],
        {"a1": {"times": [0.0, 1.0], "positions": [[0, 0, 0], [0, 0, 0]],
                "rotations": [[0, 0, 0, 1], [0, 0, 0, 1]]}},
        np.array([0.55, 0.55, 0.57]),
    )  # fmt: skip


def views(av: Avatar) -> list[tuple[np.ndarray, float, int]]:
    """(BGR frame, head height in px, yaw) for every distance and yaw; camera looks at the head."""
    scene = _scene(av)
    head = np.array(av.head_center)
    f = (H / 2) / math.tan(math.radians(FOV) / 2)
    out = []
    for d in DISTANCES:
        for yaw in YAWS:
            th = math.radians(yaw)
            pos = head + d * np.array([math.sin(th), 0.0, -math.cos(th)])
            psi = math.pi - th  # yaw so the camera's -Z looks at the head
            rot = np.array([0.0, math.sin(psi / 2), 0.0, math.cos(psi / 2)])
            fr = render(scene, CameraPose(pos, rot, FOV), 0.5, W, H, fog_strength=0.0)
            out.append((cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR), 2 * av.head_radii[1] * f / d, yaw))
    return out


def measure(portraits: dict[str, Path], out: Path) -> dict[str, Any]:
    faces = Faces()
    chars = read_wscript(OUT).header["dictionary"]["characters"]
    refs: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    rendered: dict[str, list[np.ndarray]] = {}
    out.mkdir(parents=True, exist_ok=True)
    for cid, path in sorted(portraits.items()):
        rgb, box, emb = portrait_face(faces, path)
        refs[cid] = emb
        ident = expand(
            int(chars[cid]["identity"]["seed"]), str(chars[cid]["identity"]["requested"])
        )
        av = build_avatar(ident, rgb, box)
        rendered[cid] = []
        for k, (bgr, head_px, yaw) in enumerate(views(av)):
            face = faces.find(bgr)
            emb_r = faces.embed(bgr, face) if face is not None else None
            if emb_r is not None:
                rendered[cid].append(emb_r)
            rows.append(
                {"cid": cid, "head_px": round(head_px, 1), "yaw": yaw,
                 "eligible": head_px >= 40 and yaw <= 30, "found": face is not None}
            )  # fmt: skip
            if k in (0, 12):
                cv2.imwrite(str(out / f"{cid}_view{k}.png"), bgr)
    # identity needs all four references: decided after every portrait is known
    ident_ok = total = 0
    for cid, embs in rendered.items():
        for e in embs:
            sims = {c: float(e @ refs[c]) for c in refs}
            total += 1
            ident_ok += max(sims, key=lambda c: sims[c]) == cid
    elig = [r for r in rows if r["eligible"]]
    means = {c: (np.mean(e, 0) / np.linalg.norm(np.mean(e, 0))) for c, e in rendered.items() if e}
    pairs = {
        f"{a}-{b}": round(float(means[a] @ means[b]), 3) for a, b in combinations(sorted(means), 2)
    }
    return {
        "detection_rate": round(sum(r["found"] for r in elig) / max(1, len(elig)), 3),
        "eligible_frames": len(elig),
        "identity_accuracy": round(ident_ok / max(1, total), 3),
        "faces_identified": total,
        "distinct_pairs": pairs,
        "max_pair": max(pairs.values()) if pairs else None,
        "pass": {
            "detection": sum(r["found"] for r in elig) / max(1, len(elig)) >= 0.5,
            "identity": ident_ok / max(1, total) >= 0.9,
            "distinctness": bool(pairs) and max(pairs.values()) < SAME_PERSON,
        },
        "frames": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.face_metrics")
    ap.add_argument("--portraits", type=Path, required=True)
    ap.add_argument("--choice", type=Path, required=True, help='{"C01": "C01_2.png", ...}')
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    choice = json.loads(a.choice.read_text())
    res = measure({c: a.portraits / f for c, f in choice.items()}, a.out)
    (a.out / "face_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
