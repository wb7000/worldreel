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
from formats.wreel import PROVENANCE_CLASSES, read_wreel, write_wreel
from pipeline.io import read_rgb
from stages.fill.generative import GENERATED_CONFIDENCE, generative_colours, lama_inpainter
from stages.fill.planes import drop_seen_through, progressive_fill
from stages.fill.retrieve import RETRIEVED_CONFIDENCE, FrameView, apply_references, user_candidates
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

RETRIEVED = PROVENANCE_CLASSES.index("retrieved")
GENERATED = PROVENANCE_CLASSES.index("generated")


def build(
    export: Path,
    out: Path,
    camera: Path | None = None,
    masks: Path | None = None,
    fill: bool = False,
    references: Path | None = None,
    generative: bool = False,
    frames_dir: Path | None = None,
) -> Path:
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
    if references is not None:
        # Owner-provided reference photos (algorithm A-4, D-4.4): verified geometrically against
        # the video frames before any use; recolour low-confidence Gaussians they can see.
        if camera is None or frames_dir is None:
            raise SystemExit("--references needs --camera and --frames")
        cam_ref = json.loads(camera.read_text(encoding="utf-8"))
        frame_files = sorted(frames_dir.glob("*.png"))
        reg = [i for i, pose in enumerate(cam_ref["poses"]) if pose.get("registered", True)]
        views_ref = views_from_camera_json(cam_ref)
        frame_views = [
            FrameView(read_rgb(frame_files[i]), v.r, v.t, v.k)
            for i, v in zip(reg, views_ref, strict=True)
            if i < len(frame_files)
        ]
        fg_mask = gaussians.confidence != np.float32(BACKGROUND_CONFIDENCE)
        obs_mask = fg_mask & (gaussians.cls == OBSERVED)
        target = np.nonzero(fg_mask & (gaussians.cls != OBSERVED))[0]
        size = float(np.median(gaussians.scales[target].max(axis=1))) * 2 if len(target) else 0.0
        mask, cols, ref_regions, ref_credits = apply_references(
            user_candidates(references),
            frame_views,
            gaussians.positions[obs_mask].astype(np.float64),
            gaussians.colors[obs_mask],
            gaussians.positions[target].astype(np.float64),
            size,
        )
        hit = target[mask]
        gaussians.colors[hit] = cols[mask]
        gaussians.cls[hit] = RETRIEVED
        gaussians.confidence[hit] = RETRIEVED_CONFIDENCE
        print(f"references: {len(ref_regions)} verified, {len(hit)} Gaussians recoloured")
    else:
        ref_regions, ref_credits = [], []
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
    if camera is not None and fill:
        # Progressive plane-completion fill (algorithm A-3), EXPERIMENTAL and off by default: on
        # exact synthetic geometry it is only as accurate as the reconstruction under it (D-4.3).
        # Built from observed Gaussians; quick tier within 5 s (fill LOD 0), refined (LOD 1).
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
        if generative and len(refined.positions):
            # Generated texture for the refined tier (algorithm A-5, D-4.5): realistic but
            # invented, so class `generated` with low confidence; the quick tier stays plane-mean.
            gen, gen_s = generative_colours(
                planes,
                gaussians.positions[obs].astype(np.float64),
                gaussians.colors[obs],
                gaussians.opacity[obs].astype(np.float64),
                refined,
                lama_inpainter(),
            )
            tiers[1].colors[:] = gen
            tiers[1].cls[:] = GENERATED
            tiers[1].confidence[:] = GENERATED_CONFIDENCE
            lo, hi = refined.positions.min(0), refined.positions.max(0)
            manifest.setdefault("regions", []).append(
                {
                    "id": "r_fill_generated",
                    "bounds": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
                    "class": "generated",
                    "confidence": GENERATED_CONFIDENCE,
                    "sources": [
                        {
                            "kind": "model",
                            "model": "LaMa big-lama (Carve/LaMa-ONNX lama_fp32)",
                            "version": "sha256:1faef530",
                        }
                    ],
                }
            )
            print(f"generated texture for {len(refined.positions)} fill elements in {gen_s:.1f} s")
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
    if ref_regions:
        manifest.setdefault("regions", []).extend(ref_regions)
        manifest.setdefault("extensions", {})["org.worldreel.credits"] = ref_credits
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
    p.add_argument("--fill", action="store_true", help="experimental plane-completion fill (A-3)")
    p.add_argument("--references", type=Path, help="owner reference photos + .info.json (A-4)")
    p.add_argument("--frames", type=Path, help="video frames (PNG) matching camera.json poses")
    p.add_argument("--generative", action="store_true", help="LaMa texture for --fill (A-5)")
    a = p.parse_args(argv)
    if a.generative and not a.fill:
        p.error("--generative needs --fill")
    path = build(
        a.export,
        a.out,
        camera=a.camera,
        masks=a.masks,
        fill=a.fill,
        references=a.references,
        frames_dir=a.frames,
        generative=a.generative,
    )
    doc = read_wreel(path)
    layer = doc.manifest["scene"]["layers"][0]
    print(
        f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB): "
        f"{layer['lods'][-1]['count']:,} gaussians, valid={doc.report.ok}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
