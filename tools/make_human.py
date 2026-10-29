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

from eval.face_metrics import Faces
from formats.wscript import read_wscript
from stages.generate.bible import expand
from stages.generate.face_projection import align, body_colour, landmarks, skin_colour
from stages.generate.human_preset import human_preset
from tools.make_sample_wscript import OUT
from tools.toolpaths import find_tool

SCRIPT = Path(__file__).with_name("blender_human.py").resolve()
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
    c = read_wscript(OUT).header["dictionary"]["characters"][a.cid]
    preset = human_preset(expand(int(c["identity"]["seed"]), str(c["identity"]["requested"])))
    (out / "preset.json").write_text(json.dumps(preset, indent=1))
    _blender(["-b", "--factory-startup", "-P", str(SCRIPT), "--", "build",
              str(out / "preset.json"), str(out)])  # fmt: skip
    faces = Faces()
    render = cv2.imread(str(out / "face.png"))
    pimg = cv2.imread(str(a.portrait))
    r_face, p_face = faces.find(render), faces.find(pimg)
    if r_face is None or p_face is None:
        raise RuntimeError(
            f"no face found (render {r_face is not None}, portrait {p_face is not None})"
        )
    skin_dir = (
        Path(str(find_tool("blender"))).parent / SKINS / preset["skin_mhmat"].removesuffix(".mhmat")
    )
    tex = cv2.imread(str(next(skin_dir.glob("*diffuse*.png"))), cv2.IMREAD_UNCHANGED)
    opaque = tex[..., 3] > 128 if tex.shape[2] == 4 else tex[..., :3].sum(2) > 30
    tb = np.median(tex[..., :3][opaque].astype(np.float64), 0) / 255
    tex_rgb = (float(tb[2]), float(tb[1]), float(tb[0]))
    port_rgb = skin_colour(pimg, landmarks(p_face))
    body_rgb = body_colour(tex_rgb, port_rgb)
    al = align(pimg, p_face, render, r_face, body_rgb)
    cv2.imwrite(str(out / "aligned.png"), al.rgba)
    iris = ",".join(f"{v:.4f}" for v in al.iris_rgb)
    _blender(["-b", str(out / "human.blend"), "-P", str(SCRIPT), "--", "portrait",
              str(out / "aligned.png"), str(out), "--iris", iris,
              "--skin", ",".join(f"{v:.4f}" for v in (*body_rgb, SKIN_STRENGTH))])  # fmt: skip
    info = {"cid": a.cid, "portrait": str(a.portrait),
            "similarity_residual_px": round(al.fit_error_px, 1),
            "iris_rgb": [round(v, 3) for v in al.iris_rgb],
            "skin_texture_rgb": [round(v, 3) for v in tex_rgb],
            "portrait_skin_rgb": [round(v, 3) for v in port_rgb],
            "body_rgb": [round(v, 3) for v in body_rgb]}  # fmt: skip
    (out / "make_human.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
