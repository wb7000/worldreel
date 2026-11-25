# Job: Render the character as a continuation chain (Route B).
#
# Instead of regenerating every segment from the portrait alone (which creates
# cut boundaries that need a dissolve fix), each segment after the first is
# conditioned on the previous segment's tail frames.  VACE keeps those tail
# frames as-is (black mask) and generates new frames continuing from them
# (white mask), so the timeline has no cuts at all -- no dissolve needed.
#
# Segments are dependent (N+1 needs N's tail), so they render sequentially in
# one or two Kaggle jobs.  Each job saves its last segment's tail as PNGs so
# the next job can pick up exactly where it left off.
#
# D-9.17/D-9.18 lesson: conditioning on [portrait, tail] together poisons the
# handoff (the model bridges the whole window instead of continuing from the
# last frame).  Route B conditions on the TAIL ONLY for segments 2+, with the
# six reference portraits still passed as reference_images for identity
# anchoring.  This avoids the portrait/tail conflict while keeping identity.
#
# Usage (job 1 -- segments 1-3):
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_render_continuation.py
#       --scene data:vacecast --scene data:facemodels --wait --requeue-after 30
#
# Usage (job 2 -- segments 4-5, conditioned on job 1's tail):
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_render_continuation.py
#       --scene data:vacecast --scene data:facemodels --scene data:cont-tail --wait --requeue-after 30
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ================================================================
# STEP 1: Install required packages
# ================================================================
r = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "diffusers",
        "peft",
        "opencv-python-headless",
        "scipy",
        "ftfy",
        "Pillow",
    ],
    capture_output=True,
    text=True,
)
log(f"pip install rc={r.returncode}")
if r.returncode != 0:
    log(f"pip stderr: {r.stderr[-500:]}")

u = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    capture_output=True,
    text=True,
)
log(f"torchao uninstall rc={u.returncode}")

# ================================================================
# STEP 2: Find face models and references
# ================================================================
MODEL_DIR = Path("/kaggle/input/datasets/wb3none/worldreel-data-facemodels")
OUT_DIR = Path(os.environ.get("OUT", "/kaggle/working/out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

yunet_path = None
sface_path = None
if MODEL_DIR.exists():
    for p in MODEL_DIR.rglob("face_detection_yunet_2023mar.onnx"):
        yunet_path = str(p)
        log(f"Found YuNet model: {yunet_path}")
        break
    for p in MODEL_DIR.rglob("face_recognition_sface_2021dec.onnx"):
        sface_path = str(p)
        log(f"Found SFace model: {sface_path}")
        break
if yunet_path is None or sface_path is None:
    input_root = Path("/kaggle/input")
    if input_root.exists():
        if yunet_path is None:
            for p in input_root.rglob("face_detection_yunet_2023mar.onnx"):
                yunet_path = str(p)
                log(f"Found YuNet model (broad search): {yunet_path}")
                break
        if sface_path is None:
            for p in input_root.rglob("face_recognition_sface_2021dec.onnx"):
                sface_path = str(p)
                log(f"Found SFace model (broad search): {sface_path}")
                break
if yunet_path is None:
    raise SystemExit("FATAL: face_detection_yunet_2023mar.onnx not found.")
if sface_path is None:
    raise SystemExit("FATAL: face_recognition_sface_2021dec.onnx not found.")

# ================================================================
# STEP 3: Find reference portraits and optional tail source
# ================================================================
WORK_DIR = Path(os.environ.get("WORK", "/tmp/kaggle_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR = Path(os.environ.get("CODE", "."))

input_root = Path("/kaggle/input")
portrait_path = None
reference_paths: list[str] = []
if input_root.exists():
    for ext in ("png", "jpg", "jpeg"):
        found = sorted(input_root.rglob(f"C01_*.{ext}"))
        if found:
            reference_paths = [str(p) for p in found]
            portrait_path = reference_paths[0]
            log(f"Found {len(reference_paths)} reference portraits; primary={portrait_path}")
            break
if not portrait_path:
    raise SystemExit("FATAL: no C01_* reference portraits found in any mounted dataset.")

# Optional: tail frames from a previous continuation job (mounted as data:cont-tail).
tail_source_dir: Path | None = None
tail_source_paths: list[str] = []
if input_root.exists():
    for p in input_root.rglob("tail_*.png"):
        tail_source_paths.append(str(p))
    if tail_source_paths:
        tail_source_dir = Path(tail_source_paths[0]).parent
        log(f"Found {len(tail_source_paths)} tail frames from {tail_source_dir}")

# ================================================================
# STEP 4: Write the step script
# ================================================================
step = WORK_DIR / "step_continuation.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, shutil, sys, time, torch, cv2, numpy as np
    from pathlib import Path
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    log = print

    args_path = sys.argv[1]
    with open(args_path) as f:
        a = json.load(f)

    W, H = a["width"], a["height"]
    model_id = a.get("model_id", "Wan-AI/Wan2.1-VACE-1.3B-diffusers")
    continuation = bool(a.get("continuation", True))
    tail_size = int(a.get("tail_frames", 4))
    tail_source = a.get("tail_source")
    segment_seeds = list(a.get("segment_seeds") or [])
    num_segments = int(a.get("segments", len(segment_seeds)))
    segment_seeds = segment_seeds[:num_segments]
    steps = int(a.get("steps", 50))
    log(f"Configuration: {W}x{H}, continuation={continuation}, tail_size={tail_size}")
    log(f"Segments: {num_segments}, seeds: {segment_seeds}")
    log(f"Model: {model_id}")
    log(f"Portrait: {a['portrait']}")
    if tail_source:
        log(f"Tail source: {tail_source}")
    log(f"YuNet: {a['yunet']}")
    log(f"SFace: {a['sface']}")

    # ---- Face detection + recognition ----
    from stages.generate.face_swap import FaceSwapper

    swapper = FaceSwapper(a["yunet"], a["sface"])
    detect_face = swapper.detect_face

    # ---- Load reference portraits (normalised in-step) ----
    ref_paths = a.get("references") or [a["portrait"]]
    from tools.prepare_refs import normalize_references

    norm_paths = normalize_references(
        ref_paths, Path(a["out"]) / "refs_norm",
        yunet_path=a["yunet"], sface_path=a["sface"], target=0,
        tone_strength=float(a.get("tone", 1.0)),
    )
    norm_paths = norm_paths[: int(a.get("num_references", 6))]
    ref_imgs = []
    for rp in norm_paths:
        try:
            ref_imgs.append(Image.open(rp).convert("RGB"))
        except Exception as exc:
            log(f"skipping unreadable reference {rp}: {exc}")
    if not ref_imgs:
        raise SystemExit("no readable reference portraits")
    log("normalised references: %s" % [Path(p).name for p in norm_paths])
    for p in norm_paths:
        shutil.copy2(p, Path(a["out"]) / ("ref_" + Path(p).name))

    metric_img = Image.open(a["portrait"]).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(metric_img), cv2.COLOR_RGB2BGR)
    ref_face = detect_face(ref_bgr)
    if ref_face is None:
        raise SystemExit("no face found in reference portrait")
    ref_emb = swapper.compute_embedding(ref_bgr)
    if ref_emb is None:
        raise SystemExit("could not compute reference face embedding")
    log(f"Loaded {len(ref_imgs)} reference portraits; embedding norm={float(np.linalg.norm(ref_emb)):.4f}")

    # ---- Load Wan2.1-VACE pipeline ----
    log("Loading Wan2.1-VACE pipeline...")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    log("Base model (no adapter): identity from reference views + tail continuation")

    pipe.text_encoder.to("cuda")
    with torch.no_grad():
        pos, neg = pipe.encode_prompt(prompt=a["prompt"], negative_prompt=a["negative"],
                                      do_classifier_free_guidance=True, device=torch.device("cuda"))
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log("attention: xformers memory-efficient enabled")
    except Exception as exc:
        log(f"attention: xformers unavailable ({type(exc).__name__}); using SDPA")
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        log("attention: math SDPA disabled (memory-efficient only)")
    except Exception as exc:
        log(f"attention: could not restrict SDPA backends ({type(exc).__name__})")

    grey = Image.new("RGB", (W, H), (128, 128, 128))
    mask_black = Image.new("L", (W, H), 0)     # black = keep / condition on this frame
    mask_white = Image.new("L", (W, H), 255)   # white = generate this frame

    refs = ref_imgs[: int(a.get("num_references", 6))]
    log("Using %d reference views: %s" % (len(refs), [Path(p).name for p in ref_paths[: len(refs)]]))

    from PIL import ImageOps
    seed_img = ImageOps.fit(ref_imgs[0], (W, H), method=Image.LANCZOS, centering=(0.5, 0.42))

    # ---- Load tail frames from previous job, if provided ----
    tail_frames: list[Image.Image] | None = None
    if tail_source:
        import re as _re

        tp = Path(tail_source)
        tail_files = sorted(tp.glob("tail_*.png"))
        if tail_files:
            # A mounted dataset carries every previous segment's tail group
            # (tail_seg<N>_*.png); only the HIGHEST segment continues the chain.
            def _seg_no(p) -> int:
                m = _re.search(r"tail_seg(\\d+)_", p.name)
                return int(m.group(1)) if m else -1

            last_seg = max(_seg_no(p) for p in tail_files)
            tail_files = sorted(p for p in tail_files if _seg_no(p) == last_seg)
            tail_frames = [Image.open(str(p)).convert("RGB") for p in tail_files]
            log(f"Loaded {len(tail_frames)} tail frames from seg {last_seg} in {tp}")
            # Resize to output size if needed
            tail_frames = [im.resize((W, H), Image.LANCZOS) for im in tail_frames]
        else:
            log(f"WARNING: tail source {tp} exists but no tail_*.png found; starting from portrait")

    # ---- Build conditioning for one segment ----
    def build_segment(nf: int, tail: list | None) -> tuple[list, list, int]:
        # Return (video, mask, keep) for one segment.
        # If tail is None: first segment, condition on portrait only.
        # If tail is provided: continuation, condition on tail frames only.
        if tail is None:
            # First segment: portrait (black mask) + grey canvas (white mask).
            video = [seed_img, *[grey] * (nf - 1)]
            mask = [mask_black, *[mask_white] * (nf - 1)]
            keep = 1
        else:
            # Continuation: tail frames (black mask) + grey canvas (white mask).
            # Total must be 4k+1 for Wan's causal VAE.
            k = len(tail)
            if k >= nf:
                raise SystemExit(f"tail_size {k} >= n_frames {nf}; reduce tail_frames")
            video = list(tail) + [grey] * (nf - k)
            mask = [mask_black] * k + [mask_white] * (nf - k)
            keep = k
        return video, mask, keep

    # ---- Generate one segment ----
    def generate_segment(nf: int, tail: list[Image.Image] | None, seed_value: int) -> list[Image.Image]:
        video, mask, keep = build_segment(nf, tail)
        log(f"Generating {nf} frames, keep={keep}, seed={seed_value}, "
            f"{'portrait' if tail is None else 'tail-continuation'}")
        out = None
        floor = max(keep + 1, 9)  # at least keep frames + some generated
        while True:
            try:
                out = pipe(video=video, mask=mask, reference_images=refs,
                           prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                           num_frames=nf, num_inference_steps=steps, guidance_scale=5.0,
                           conditioning_scale=1.0, output_type="pil",
                           generator=torch.Generator("cpu").manual_seed(seed_value)).frames[0]
                break
            except Exception as exc:
                msg = str(exc).lower()
                if "out of memory" not in msg:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                smaller = max(floor, ((nf // 2) | 1))
                if smaller >= nf:
                    raise RuntimeError(f"CUDA OOM at floor of {nf} frames") from exc
                nf = smaller
                log(f"CUDA OOM -> retrying with {nf} frames")
                video, mask, keep = build_segment(nf, tail)
        if out is None:
            raise RuntimeError("generation failed")
        return list(out), keep

    # ---- Save tail frames as PNGs for the next job ----
    def save_tail(frames: list[Image.Image], tag: str) -> list[str]:
        saved = []
        tail_dir = Path(a["out"]) / "tail_frames"
        tail_dir.mkdir(parents=True, exist_ok=True)
        k = min(tail_size, len(frames))
        for i, fr in enumerate(frames[-k:]):
            p = tail_dir / f"tail_{tag}_{i:02d}.png"
            fr.save(p)
            saved.append(str(p))
        log(f"Saved {len(saved)} tail frames ({tag}) to {tail_dir}")
        return saved

    # ---- Run all segments ----
    all_frames: list[Image.Image] = []
    report = {"segments": [], "reference_portrait": a["portrait"],
              "reference_images": len(refs), "resolution": f"{W}x{H}",
              "steps": steps, "continuation": continuation,
              "tail_size": tail_size, "total_seconds": 0.0,
              "segment_seeds": list(segment_seeds), "identity_sim": []}
    n_frames = 49  # Wan causal VAE wants 4k+1

    for s in range(num_segments):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        seed_value = int(segment_seeds[s])

        # Determine tail for this segment.
        if s == 0 and tail_frames is None:
            # First segment of the whole chain: portrait conditioning.
            seg_tail = None
        else:
            # Continuation: use tail from previous segment (or loaded from previous job).
            seg_tail = tail_frames if tail_frames is not None else None
            if seg_tail is None:
                raise SystemExit(f"Segment {s}: no tail frames available for continuation")

        out, keep = generate_segment(n_frames, seg_tail, seed_value)

        # Save tail frames from this segment's OUTPUT (last tail_size generated frames).
        # These are real generated frames, not the conditioned input.
        tail_tag = f"seg{s}_seed{seed_value}"
        save_tail(out, tail_tag)

        # The next segment's tail comes from this segment's output.
        tail_frames = out[-tail_size:]

        # Drop the `keep` conditioned frames (portrait or tail) from output.
        new_frames = out[keep:]
        all_frames += new_frames

        # Identity measurement (every generated frame vs reference portrait).
        seg_sims = []
        seg_embs = []
        seg_sharps = []
        faces_found = 0
        for fr in new_frames:
            frame_bgr = cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR)
            det = detect_face(frame_bgr)
            if det is None:
                continue
            faces_found += 1
            emb = swapper.compute_embedding(frame_bgr)
            if emb is not None:
                seg_sims.append(float(emb @ ref_emb))
                seg_embs.append(np.asarray(emb, dtype=np.float64))
            bx, by, bw, bh = (int(v) for v in det[0][:4])
            roi = frame_bgr[max(0, by):by + bh, max(0, bx):bx + bw]
            if roi.size:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                seg_sharps.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        self_min = -1.0
        if seg_embs:
            matrix = np.stack(seg_embs)
            median = np.median(matrix, axis=0)
            norm = float(np.linalg.norm(median))
            if norm > 0:
                self_min = float(np.min(matrix @ (median / norm)))
        drift = ((seg_sims[-1] - seg_sims[0]) / max(1, len(seg_sims) - 1)
                 if len(seg_sims) > 1 else 0.0)
        seg_identity = float(np.mean(seg_sims)) if seg_sims else -1.0
        report["identity_sim"].append(round(seg_identity, 4))

        seg_seconds = len(new_frames) / 16.0
        report["total_seconds"] += seg_seconds
        report["segments"].append({
            "segment": s,
            "seed": seed_value,
            "seconds": round(time.time() - t0, 1),
            "video_seconds": round(seg_seconds, 2),
            "frames": len(new_frames),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "faces_found": faces_found,
            "identity_mean": round(seg_identity, 4),
            "identity_min": round(float(np.min(seg_sims)), 4) if seg_sims else -1.0,
            "identity_first": round(float(seg_sims[0]), 4) if seg_sims else -1.0,
            "identity_last": round(float(seg_sims[-1]), 4) if seg_sims else -1.0,
            "identity_drift_per_frame": round(float(drift), 5),
            "self_consistency_min": round(self_min, 4),
            "sharpness_mean": round(float(np.mean(seg_sharps)), 2) if seg_sharps else -1.0,
            "continuation": seg_tail is not None,
        })
        log("Segment %d (seed %d): faces %d/%d identity %.4f (min %.4f, drift %+.5f/frame) "
            "self_min %.4f sharpness %.1f peak_gb %.2f continuation=%s"
            % (s, seed_value, faces_found, len(new_frames), seg_identity,
               float(np.min(seg_sims)) if seg_sims else -1.0, drift, self_min,
               float(np.mean(seg_sharps)) if seg_sharps else -1.0,
               torch.cuda.max_memory_allocated() / 1e9, seg_tail is not None))
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Export video (continuous, no cuts, no dissolve needed) ----
    out_dir = Path(a["out"])
    seed_tag = f"{segment_seeds[0]}-{segment_seeds[-1]}" if segment_seeds else "none"
    cont_tag = "cont" if continuation else "independent"
    stem = f"refs{len(refs)}_segs{seed_tag}_{cont_tag}"
    out_mp4 = str(out_dir / f"{stem}.mp4")
    export_to_video(all_frames, out_mp4, fps=16, quality=10)
    log(f"Video exported: {out_mp4} ({len(all_frames)} frames, {len(all_frames)/16.0:.2f}s)")

    # ---- Motion stats ----
    diffs = []
    try:
        for i in range(1, len(all_frames)):
            pa = np.asarray(all_frames[i - 1], dtype=np.float32)
            cu = np.asarray(all_frames[i], dtype=np.float32)
            diffs.append(float(np.abs(pa - cu).mean()))
    except Exception as exc:
        log(f"motion measurement failed: {type(exc).__name__}: {exc}")
    report["motion_mean"] = round(float(np.mean(diffs)), 4) if diffs else 0.0
    report["motion_median"] = round(float(np.median(diffs)), 4) if diffs else 0.0
    report["motion_max"] = round(float(np.max(diffs)), 4) if diffs else 0.0
    report["still_fraction"] = round(float(np.mean([1.0 if d < 0.5 else 0.0 for d in diffs])), 4) if diffs else 0.0
    report["motion_pairs"] = len(diffs)
    report["num_frames"] = len(all_frames)
    report["duration_seconds"] = round(len(all_frames) / 16.0, 2)
    log(f"MOTION mean={report['motion_mean']} median={report['motion_median']} "
        f"max={report['motion_max']} still={report['still_fraction']} frames={len(all_frames)}")

    with open(str(out_dir / f"{stem}.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"Report: {json.dumps(report, indent=2)}")
    log(f"TOTAL VIDEO SECONDS: {report['total_seconds']:.2f}")
    log("DONE - SUCCESS")
    """),
    encoding="utf-8",
)

# ================================================================
# STEP 5: Write args and run
# ================================================================
args = {
    "model_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "portrait": portrait_path,
    "references": reference_paths,
    "yunet": str(yunet_path),
    "sface": str(sface_path),
    "out": str(OUT_DIR),
    "num_references": 6,
    # Job 2 config: continue the chain from job 1's tail (mounted data:cont-tail).
    # Job 1 ran with segment_seeds [43,44,45] and segments 3 (git history: 533618a).
    "segment_seeds": [46, 47],
    "segments": 2,
    "tail_frames": 4,
    "continuation": True,
    "tail_source": str(tail_source_dir) if tail_source_dir else None,
    "steps": 50,
    "width": 832,
    "height": 480,
    "prompt": (
        "real video footage of the same woman shown in the reference photographs, "
        "standing still in a studio, breathing gently, blinking naturally, mouth closed, "
        "calm neutral expression, eyes open looking directly at the camera, "
        "subtle natural head movement, locked-off tripod camera, no camera movement, "
        "no zoom, consistent head and shoulders framing throughout, "
        "plain grey studio wall behind, soft even natural light, consistent lighting, "
        "detailed natural skin with visible pores, natural hair, sharp focus, crisp detail, "
        "shot on 35mm film, photorealistic"
    ),
    "negative": (
        "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
        "blurry, soft focus, out of focus, deformed face, extra eyes, second face, "
        "overlapping face, double face, morphing face, changing face, inconsistent face, "
        "warped face, flickering, jitter, strobing, pulsing brightness, "
        "text, watermark, overexposed, dark blotches on skin, static, still picture, "
        "talking, open mouth, camera movement, camera zoom, dolly, pull back, push in, "
        "handheld shake"
    ),
    "tone": 1.0,
}

args_path = WORK_DIR / "continuation_args.json"
args_path.write_text(json.dumps(args, indent=2))

env = {
    **os.environ,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "OUT": str(OUT_DIR),
    "WORK": str(WORK_DIR),
    "CODE": str(CODE_DIR),
    "PYTHONPATH": str(CODE_DIR),
}

r = subprocess.run(
    [str(sys.executable), str(step), str(args_path)],
    capture_output=True,
    text=True,
    timeout=6 * 3600,
    env=env,
)

log_path = OUT_DIR / "continuation_render.log"
stdout_part = r.stdout[-20000:] if r.stdout else ""
stderr_part = r.stderr[-8000:] if r.stderr else ""
log_path.write_text(
    (stdout_part + "\\n" + stderr_part).strip() if r.returncode else stdout_part.strip(),
    encoding="utf-8",
)

log(f"Continuation job rc={r.returncode}")
if r.returncode != 0:
    log(f"STDERR: {r.stderr[-2000:]}")
else:
    log("Job completed successfully!")

# ================================================================
# STEP 6: Recover + verify artefacts
# ================================================================
for stray in (Path("/tmp/kaggle_out"), WORK_DIR / "out"):
    if not stray.exists():
        continue
    for p in sorted(stray.iterdir()):
        if p.is_file() and not (OUT_DIR / p.name).exists():
            shutil.copy2(p, OUT_DIR / p.name)
            log(f"Recovered stray output -> {p.name}")

# Also recover the tail_frames directory
tail_src = WORK_DIR / "out" / "tail_frames"
if tail_src.exists():
    tail_dst = OUT_DIR / "tail_frames"
    tail_dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(tail_src.iterdir()):
        if p.is_file() and not (tail_dst / p.name).exists():
            shutil.copy2(p, tail_dst / p.name)
            log(f"Recovered tail frame -> {p.name}")

_used_seeds = args["segment_seeds"][: args["segments"]]
expected_stem = f"refs{args['num_references']}_segs{_used_seeds[0]}-{_used_seeds[-1]}_cont"
expected = [f"{expected_stem}.mp4", f"{expected_stem}.json", "continuation_render.log"]
for name in expected:
    p = OUT_DIR / name
    size = p.stat().st_size if p.exists() else -1
    log(f"artefact {name}: {'MISSING' if size < 0 else f'{size} bytes'}")

if not (OUT_DIR / f"{expected_stem}.mp4").exists():
    log("FATAL: continuation video was not produced inside the archived output directory")
    raise SystemExit(2)

log("done")
