"""Build a signed Gaussian-splat `.wreel` from a Kaggle splat export (runs/kaggle/.../<scene>/).

    python -m tools.splat_wreel runs/kaggle/splat_export/out/seed11 --out world_splats.wreel

Inputs from the export: `splats.npz` (trained parameters, OpenCV axes), `template.wreel` (the
classical reconstruction, signed on Kaggle with a throwaway key) and `meta.json`. The result is
validated and signed with the owner's local key (formats.common.load_or_create_signing_key).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.geometry import quat_to_matrix
from formats.common import load_or_create_signing_key
from formats.wreel import read_wreel, write_wreel
from stages.package.gaussians import carve_people, splats_to_gaussians, wreel_with_gaussians


def build(export: Path, out: Path) -> Path:
    meta = json.loads((export / "meta.json").read_text(encoding="utf-8"))
    with np.load(export / "splats.npz") as npz:
        params = {k: npz[k] for k in npz.files}
    template = read_wreel(export / "template.wreel")
    if template.report.errors and any(
        i.code != "signature_invalid" for i in template.report.errors
    ):
        raise SystemExit(f"template is invalid: {template.report.errors}")
    # Background sphere sits at background_radius x extent; anything beyond half of that is
    # background (spec: lower confidence, DECISIONS D-3b.7).
    beyond = 0.5 * meta["background_radius"] * meta["extent"]
    gaussians = splats_to_gaussians(
        params, centre_cv=np.asarray(meta["centre_cv"]), background_beyond=beyond
    )
    people = []
    for actor in template.manifest.get("actors", []):
        tr = template.track(actor["track"]["path"])
        ups = np.stack([quat_to_matrix(q) @ np.array([0.0, 1.0, 0.0]) for q in tr["rotations"]])
        people.append((np.asarray(tr["positions"], np.float64), ups, actor["eye_height"] / 0.925))
    if people:
        gaussians, removed = carve_people(gaussians, people)
        print(f"carved {removed:,} gaussians inside {len(people)} people's body volumes")
    manifest, entries = wreel_with_gaussians(
        template,
        gaussians,
        stage_record={"name": "reconstruct.splat", "version": "0.1.0"},
    )
    manifest["generation"]["seeds"]["splat"] = int(meta["seed"])
    manifest["title"] = manifest["title"] + " (Gaussian splats)"
    write_wreel(out, manifest, entries, load_or_create_signing_key())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.splat_wreel")
    p.add_argument("export", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    path = build(a.export, a.out)
    doc = read_wreel(path)
    layer = doc.manifest["scene"]["layers"][0]
    print(
        f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB): "
        f"{layer['lods'][-1]['count']:,} gaussians, valid={doc.report.ok}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
