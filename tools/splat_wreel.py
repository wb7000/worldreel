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
from stages.fill.planes import drop_seen_through, progressive_fill
from stages.package.confidence import OBSERVED, relabel_gaussians, views_from_camera_json
from stages.package.follow import follow_cameras
from stages.package.gaussians import (
    BACKGROUND_CONFIDENCE,
    add_fill_layer,
    carve_people,
    fill_gaussians,
    splats_to_gaussians,
    wreel_with_gaussians,
)


def build(export: Path, out: Path, camera: Path | None = None, masks: Path | None = None) -> Path:
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
    if camera is not None:
        # Honest confidence map (algorithm A-1, D-4.1): relabel every non-background Gaussian
        # from how many registered views saw it, from how wide an angle, and whether the
        # classical multi-view stereo found a surface near it.
        cam = json.loads(camera.read_text(encoding="utf-8"))
        labels = None
        if masks is not None:
            with np.load(masks) as npz:
                labels = npz["labels"]
        views = views_from_camera_json(cam, labels)
        layer = template.manifest["scene"]["layers"][0]
        mvs = template.points(layer["lods"][-1]["path"]).positions.astype(np.float64)
        fg = gaussians.confidence != np.float32(BACKGROUND_CONFIDENCE)
        cls, conf, report = relabel_gaussians(
            gaussians.positions.astype(np.float64),
            gaussians.opacity,
            fg,
            views,
            mvs,
            float(meta["extent"]),
        )
        gaussians.cls[fg] = cls
        gaussians.confidence[fg] = conf
        print(f"confidence map: {report}")
    else:
        print("confidence map: no camera.json given, keeping constant labels (baseline)")
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
    if camera is not None and template.manifest.get("actors"):
        # Follow views planned in seen space (algorithm A-2, D-4.2): needs the A-1 confidences.
        keep = np.nonzero(gaussians.opacity >= 0.2)[0]
        if len(keep) > 60000:
            keep = np.sort(np.random.default_rng(0).choice(keep, 60000, replace=False))
        tracks = {a["id"]: template.track(a["track"]["path"]) for a in template.manifest["actors"]}
        follow = follow_cameras(
            manifest,
            tracks,
            gaussians.positions[keep].astype(np.float64),
            gaussians.confidence[keep],
        )
        print(f"follow paths planned (mean seen share): {follow}")
    if camera is not None:
        # Progressive plane-completion fill (algorithm A-3, D-4.3): built from observed Gaussians,
        # quick tier within a 5 s budget (fill LOD 0), refined tier after (fill LOD 1).
        obs = (gaussians.cls == OBSERVED) & (
            gaussians.confidence != np.float32(BACKGROUND_CONFIDENCE)
        )
        planes, quick, refined = progressive_fill(
            gaussians.positions[obs].astype(np.float64),
            gaussians.colors[obs],
            float(meta["extent"]),
            budget_s=5.0,
            refine_budget_s=60.0,
            weights=gaussians.opacity[obs].astype(np.float64),
        )
        obs_pos = gaussians.positions[obs].astype(np.float64)
        seen_through = [drop_seen_through(t, views, obs_pos) for t in (quick, refined)]
        print(f"fill cells dropped as seen-through open space: {seen_through}")
        tiers = [
            fill_gaussians(
                t.positions, t.normals, t.colors, t.confidence.astype(np.float64), t.cell
            )
            for t in (quick, refined)
        ]
        add_fill_layer(manifest, entries, tiers[0], tiers[1])
        if len(refined.positions):
            lo, hi = refined.positions.min(0), refined.positions.max(0)
            manifest.setdefault("regions", []).append(
                {
                    "id": "r_fill_planes",
                    "bounds": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
                    "class": "inferred",
                    "confidence": round(float(refined.confidence.mean()), 3),
                    "sources": [
                        {"kind": "video"},
                        {
                            "kind": "model",
                            "model": "worldreel A-3 plane-completion fill",
                            "version": "0.1.0",
                        },
                    ],
                }
            )
        print(
            f"fill: {len(planes)} planes; quick {len(quick.positions)} in {quick.seconds:.1f} s "
            f"(complete={quick.complete}); "
            f"refined {len(refined.positions)} in {refined.seconds:.1f} s"
        )
    manifest["generation"]["seeds"]["splat"] = int(meta["seed"])
    manifest["title"] = manifest["title"] + " (Gaussian splats)"
    write_wreel(out, manifest, entries, load_or_create_signing_key())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.splat_wreel")
    p.add_argument("export", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera", type=Path, help="camera.json of the run (enables the A-1 map)")
    p.add_argument("--masks", type=Path, help="actors masks.npz (people pixels are not evidence)")
    a = p.parse_args(argv)
    path = build(a.export, a.out, a.camera, a.masks)
    doc = read_wreel(path)
    layer = doc.manifest["scene"]["layers"][0]
    print(
        f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB): "
        f"{layer['lods'][-1]['count']:,} gaussians, valid={doc.report.ok}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
