"""Build a character as a real MPFB2 human with its portrait face (A-14 v2, D-8.11).

    python -m tools.make_human C02 --portrait runs/kaggle/portraits_run3_merged/C02_3.png \
        --out runs/humans/C02

Writes the preset, runs Blender pass `build` (human + face camera render), aligns the portrait
to that render (`stages.generate.face_projection`), then Blender pass `portrait` (projection +
look-test renders). Blender comes from worldreel.local.toml (4.2+ with MPFB installed).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from eval.face_metrics import Faces, imread
from formats.wscript import read_wscript
from stages.generate.bible import expand
from stages.generate.face_landmarks import dense
from stages.generate.face_projection import (
    BEARD,
    MOUSTACHE,
    STUBBLE,
    align,
    body_colour,
    curliness,
    eye_texture,
    facial_hair_kind,
    hair_colour,
    landmarks,
    skin_colour,
)
from stages.generate.human_preset import cast_presets
from tools.make_sample_wscript import OUT
from tools.toolpaths import find_tool

SCRIPT = Path(__file__).with_name("blender") / "human.py"
SKINS = Path("portable/extensions/.user/user_default/mpfb/data/skins")
SKIN_STRENGTH = 0.6  # body skin texture pulled this far toward the shared skin albedo


def _blender(args: list[str]) -> None:
    blender = find_tool("blender")
    if blender is None:
        raise RuntimeError("Blender not found (worldreel.local.toml [tools] blender)")
    r = subprocess.run([str(blender), *args], capture_output=True, text=True)
    if r.returncode or "Traceback" in r.stderr + r.stdout:
        raise RuntimeError((r.stdout + r.stderr)[-3000:])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.make_human")
    ap.add_argument("cid")
    ap.add_argument("--portrait", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    chars = read_wscript(OUT).header["dictionary"]["characters"]
    cast = {cid: expand(int(c["identity"]["seed"]), str(c["identity"]["requested"]))
            for cid, c in chars.items()}  # fmt: skip
    preset = cast_presets(cast)[a.cid]  # hair / outfit differ across the cast
    faces = Faces()
    pimg = imread(a.portrait)
    p_face = faces.find(pimg)
    if p_face is None:
        raise RuntimeError(f"no face in {a.portrait}")
    if preset["hair"]:  # the hair colour the portrait shows, not the bible's word for it
        # the bible's colour, darkened to the portrait's where it is darker (hair_colour)
        wanted = preset["worldreel"]["hair_rgb"]
        hair = hair_colour(pimg, landmarks(p_face), (wanted[0], wanted[1], wanted[2]))
        preset["worldreel"]["hair_rgb"] = [round(v, 4) for v in hair]
        preset["worldreel"]["curl"] = round(curliness(pimg, landmarks(p_face)), 3)
    (out / "preset.json").write_text(json.dumps(preset, indent=1))
    _blender(["-b", "--factory-startup", "-P", str(SCRIPT), "--", "build",
              str(out / "preset.json"), str(out)])  # fmt: skip
    render = imread(out / "face.png")
    r_face = faces.find(render)
    if r_face is None:
        raise RuntimeError("no face in the face-camera render")
    skin_dir = (
        Path(str(find_tool("blender"))).parent / SKINS / preset["skin_mhmat"].removesuffix(".mhmat")
    )
    tex_img = cv2.imread(str(next(skin_dir.glob("*diffuse*.png"))), cv2.IMREAD_UNCHANGED)
    if tex_img is None:
        raise FileNotFoundError(skin_dir)
    tex: np.ndarray = tex_img
    opaque = tex[..., 3] > 128 if tex.shape[2] == 4 else tex[..., :3].sum(2) > 30
    tb = np.median(tex[..., :3][opaque].astype(np.float64), 0) / 255
    tex_rgb = (float(tb[2]), float(tb[1]), float(tb[0]))
    port_rgb = skin_colour(pimg, landmarks(p_face))
    body_rgb = body_colour(tex_rgb, port_rgb)
    p_dense, r_dense = dense(pimg), dense(render)
    kind = facial_hair_kind(cast[a.cid].appearance)
    al = align(pimg, p_face, render, r_face, body_rgb, p_dense, r_dense, kind)
    beard_args: list[str] = []
    rgb = ",".join(f"{v:.4f}" for v in preset["worldreel"]["hair_rgb"])  # same as head hair
    for piece, mask in al.facial_hair:
        mask_path = out / f"facial_{piece}.png"
        cv2.imwrite(str(mask_path), mask)
        length, density = {"stubble": STUBBLE, "moustache": MOUSTACHE, "beard": BEARD}[piece]
        piece_rgb = rgb
        if piece == "stubble":  # short hairs over skin read partly as skin colour
            hair = preset["worldreel"]["hair_rgb"]
            piece_rgb = ",".join(
                f"{0.6 * h + 0.4 * b:.4f}" for h, b in zip(hair, body_rgb, strict=True)
            )
        beard_args += ["--beard", f"{mask_path}@{piece_rgb}@{length}@{density}@{piece}"]
    cv2.imwrite(str(out / "aligned.png"), al.rgba)
    eye = eye_texture(pimg, landmarks(p_face), cast[a.cid].appearance["skin_tone"])
    _blender(["-b", str(out / "human.blend"), "-P", str(SCRIPT), "--", "portrait",
              str(out / "aligned.png"), str(out), "--eye", eye, *beard_args,
              "--skin", ",".join(f"{v:.4f}" for v in (*body_rgb, SKIN_STRENGTH))])  # fmt: skip
    info = {"cid": a.cid, "portrait": str(a.portrait),
            "similarity_residual_px": round(al.fit_error_px, 1),
            "eye_texture": eye,
            "dense_landmarks": p_dense is not None and r_dense is not None,
            "facial_hair": kind,
            "skin_texture_rgb": [round(v, 3) for v in tex_rgb],
            "portrait_skin_rgb": [round(v, 3) for v in port_rgb],
            "body_rgb": [round(v, 3) for v in body_rgb]}  # fmt: skip
    (out / "make_human.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
