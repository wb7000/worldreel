# Job: VACE seam-bridge probe (route A of D-9.32 follow-up).
#
# Replaces the 6-frame pixel-dissolve at each cut of the assembled, stabilized
# timeline with a short VACE-generated bridge: context frames from both takes
# (black mask = kept as conditioning) plus newly generated in-between frames
# (white mask = generated), guided by the six C01 reference portraits so the
# face identity holds through the transition.  The bridge invents real
# in-between motion instead of spreading a pose jump over 6 blended frames.
#
# Input: runs/analysis/refsnorm_asm_stab.mkv, uploaded as the private dataset
#   worldreel-data-seam-timeline and mounted via --scene data:seam-timeline.
# References: six C01 portraits (mounted via --scene data:vacecast or
#   --scene data:facemodels, found by globbing C01_* in /kaggle/input).
# Face models: YuNet + SFace from --scene data:facemodels (for identity
#   measurement against the reference portrait).
#
# Output: refs_bridge_seams.mp4 (the full timeline with bridges at all 4 cuts)
#         refs_bridge_seams.json  (per-cut identity + sharpness report)
#
# Usage:
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_render_seam_bridge.py
#       --scene data:vacecast --scene data:facemodels --scene data:seam-timeline --wait
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

# peft's LoRA dispatcher raises on an importable-but-old torchao (Kaggle ships 0.10.0,
# peft wants >=0.16). Uninstall it so the plain nn.Linear path is used.
u = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    capture_output=True,
    text=True,
)
log(f"torchao uninstall rc={u.returncode}")

# ================================================================
# STEP 2: Find face detection/recognition models in mounted Kaggle dataset
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
# STEP 3: Find the stabilized timeline and the reference portraits
# ================================================================
WORK_DIR = Path(os.environ.get("WORK", "/tmp/kaggle_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR = Path(os.environ.get("CODE", "."))

# The assembled, stabilized timeline (uploaded as worldreel-data-seam-timeline).
timeline_path = None
input_root = Path("/kaggle/input")
if input_root.exists():
    for p in input_root.rglob("refsnorm_asm_stab.mkv"):
        timeline_path = str(p)
        log(f"Found stabilized timeline: {timeline_path}")
        break
if timeline_path is None:
    raise SystemExit("FATAL: refsnorm_asm_stab.mkv not found. Mount --scene data:seam-timeline.")

# Six C01 reference portraits of the same identity.
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

# ================================================================
# STEP 4: Write the step script
# ================================================================
step = WORK_DIR / "step_seam_bridge.py"
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
    log(f"Configuration: {W}x{H}, seam-bridge probe")
    log(f"Model: {model_id}")
    log(f"Timeline: {a['timeline']}")
    log(f"Portrait: {a['portrait']}")
    log(f"References: {a.get('references', [a['portrait']])}")
    log(f"YuNet: {a['yunet']}")
    log(f"SFace: {a['sface']}")

    # ---- Face detection + recognition (repo implementation) ----
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
    ref_bbox, ref_lm = ref_face
    ref_emb = swapper.compute_embedding(ref_bgr)
    if ref_emb is None:
        raise SystemExit("could not compute reference face embedding")
    log(f"Loaded {len(ref_imgs)} reference portraits; embedding norm={float(np.linalg.norm(ref_emb)):.4f}")

    # ---- Load Wan2.1-VACE pipeline ----
    log("Loading Wan2.1-VACE pipeline...")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    log("Base model (no adapter): identity from reference views")

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

    # ---- Read the stabilized timeline ----
    cap = cv2.VideoCapture(a["timeline"])
    timeline_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timeline_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    N = len(timeline_frames)
    log(f"Timeline: {N} frames at {W}x{H}")

    # Cut bounds in the stabilized timeline (from the D-9.32 assembly step).
    cuts = [int(c) for c in a.get("cuts", [46, 84, 125, 168])]
    # Context frames kept as-is on each side of the cut.
    ctx = int(a.get("context_frames", 4))      # frames kept from each take
    bridge = int(a.get("bridge_frames", 9))    # frames generated between them
    steps = int(a.get("steps", 50))

    # Sanity: ctx*2 + bridge must be 4k+1 for Wan's causal VAE.
    region = 2 * ctx + bridge
    if (region - 1) % 4 != 0:
        raise SystemExit(f"region size {region} is not 4k+1 (ctx={ctx}, bridge={bridge})")

    grey = Image.new("RGB", (W, H), (128, 128, 128))
    mask_black = Image.new("L", (W, H), 0)     # black = keep / condition on this frame
    mask_white = Image.new("L", (W, H), 255)   # white = generate this frame

    refs = ref_imgs[: int(a.get("num_references", 6))]
    log("Using %d reference views: %s" % (len(refs), [Path(p).name for p in ref_paths[: len(refs)]]))

    from PIL import ImageOps
    seed_img = ImageOps.fit(ref_imgs[0], (W, H), method=Image.LANCZOS, centering=(0.5, 0.42))

    # ---- Generate a bridge at one cut ----
    def build_bridge(cut: int) -> list:
        # Return [A_tail(ctx, kept) + bridge(bridge, generated) + B_head(ctx, kept)]
        # and the corresponding mask list.
        lo = max(0, cut - ctx)
        hi = min(N, cut + ctx)
        a_tail = [Image.fromarray(timeline_frames[i]) for i in range(lo, cut)]
        b_head = [Image.fromarray(timeline_frames[i]) for i in range(cut, hi)]
        # Pad context to exactly ctx frames each side (repeat edge frames if needed).
        while len(a_tail) < ctx:
            a_tail.insert(0, a_tail[0])
        while len(b_head) < ctx:
            b_head.append(b_head[-1])
        video = a_tail + [grey] * bridge + b_head
        mask = [mask_black] * ctx + [mask_white] * bridge + [mask_black] * ctx
        nf = len(video)
        assert nf == region and (nf - 1) % 4 == 0, f"region {nf} not 4k+1"
        out = pipe(video=video, mask=mask, reference_images=refs,
                   prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                   num_frames=nf, num_inference_steps=steps, guidance_scale=5.0,
                   conditioning_scale=1.0, output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(1000 + cut)).frames[0]
        gc.collect()
        torch.cuda.empty_cache()
        return list(out)

    # ---- Stitch the full bridged timeline ----
    all_frames: list = []
    report = {"cuts": [], "resolution": f"{W}x{H}", "steps": steps,
              "context_frames": ctx, "bridge_frames": bridge,
              "reference_portrait": a["portrait"], "reference_images": len(refs)}
    prev = 0
    for ci, cut in enumerate(cuts):
        # Raw frames from previous cut end up to this cut's tail start.
        tail_start = max(prev, cut - ctx)
        all_frames.extend([Image.fromarray(timeline_frames[i]) for i in range(prev, tail_start)])

        log(f"Cut {ci} at {cut}: generating bridge ({ctx} tail + {bridge} + {ctx} head)")
        bridge_out = build_bridge(cut)
        # Drop the first ctx frames (they duplicate the raw tail already emitted)
        # and the last ctx frames (they will be emitted as the next raw segment).
        # Only the generated bridge frames are new content.
        all_frames.extend(bridge_out[ctx : ctx + bridge])

        # Measure identity + sharpness on the bridge frames.
        seg_sims = []
        seg_sharps = []
        for fr in bridge_out[ctx : ctx + bridge]:
            frame_bgr = cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR)
            det = detect_face(frame_bgr)
            if det is None:
                continue
            emb = swapper.compute_embedding(frame_bgr)
            if emb is not None:
                seg_sims.append(float(emb @ ref_emb))
            bx, by, bw, bh = (int(v) for v in det[0][:4])
            roi = frame_bgr[max(0, by):by + bh, max(0, bx):bx + bw]
            if roi.size:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                seg_sharps.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        report["cuts"].append({
            "cut": cut,
            "bridge_frames": bridge,
            "identity_mean": round(float(np.mean(seg_sims)), 4) if seg_sims else -1.0,
            "identity_min": round(float(np.min(seg_sims)), 4) if seg_sims else -1.0,
            "sharpness_mean": round(float(np.mean(seg_sharps)), 2) if seg_sharps else -1.0,
        })
        log(f"Cut {ci}: identity {report['cuts'][-1]['identity_mean']} (min {report['cuts'][-1]['identity_min']}) "
            f"sharpness {report['cuts'][-1]['sharpness_mean']}")

        # Next raw segment starts after the context head frames.
        prev = cut + ctx

    # Raw tail after the last cut's head context.
    all_frames.extend([Image.fromarray(timeline_frames[i]) for i in range(prev, N)])

    log(f"Bridged timeline: {len(all_frames)} frames (was {N})")

    # ---- Export video ----
    out_dir = Path(a["out"])
    out_mp4 = str(out_dir / "refs_bridge_seams.mp4")
    export_to_video(all_frames, out_mp4, fps=16, quality=10)
    log(f"Video exported: {out_mp4}")

    # Motion stats.
    diffs = []
    try:
        for i in range(1, len(all_frames)):
            prev_a = np.asarray(all_frames[i - 1], dtype=np.float32)
            cur = np.asarray(all_frames[i], dtype=np.float32)
            diffs.append(float(np.abs(prev_a - cur).mean()))
    except Exception as exc:
        log(f"motion measurement failed: {type(exc).__name__}: {exc}")
    report["motion_mean"] = round(float(np.mean(diffs)), 4) if diffs else 0.0
    report["motion_median"] = round(float(np.median(diffs)), 4) if diffs else 0.0
    report["still_fraction"] = round(float(np.mean([1.0 if d < 0.5 else 0.0 for d in diffs])), 4) if diffs else 0.0
    log(f"MOTION mean={report['motion_mean']} median={report['motion_median']} still={report['still_fraction']}")

    with open(str(out_dir / "refs_bridge_seams.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"Report: {json.dumps(report, indent=2)}")
    log("DONE - SUCCESS")
    """),
    encoding="utf-8",
)

# ================================================================
# STEP 5: Write args and run
# ================================================================
args = {
    "model_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "timeline": timeline_path,
    "portrait": portrait_path,
    "references": reference_paths,
    "yunet": str(yunet_path),
    "sface": str(sface_path),
    "out": str(OUT_DIR),
    "num_references": 6,
    "cuts": [46, 84, 125, 168],
    "context_frames": 4,
    "bridge_frames": 5,
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

args_path = WORK_DIR / "seam_bridge_args.json"
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

log_path = OUT_DIR / "seam_bridge_render.log"
stdout_part = r.stdout[-20000:] if r.stdout else ""
stderr_part = r.stderr[-8000:] if r.stderr else ""
log_path.write_text(
    (stdout_part + "\\n" + stderr_part).strip() if r.returncode else stdout_part.strip(),
    encoding="utf-8",
)

log(f"Seam bridge job rc={r.returncode}")
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

expected = ["refs_bridge_seams.mp4", "refs_bridge_seams.json", "seam_bridge_render.log"]
for name in expected:
    p = OUT_DIR / name
    size = p.stat().st_size if p.exists() else -1
    log(f"artefact {name}: {'MISSING' if size < 0 else f'{size} bytes'}")

if not (OUT_DIR / "refs_bridge_seams.mp4").exists():
    log("FATAL: refs_bridge_seams.mp4 was not produced inside the archived output directory")
    raise SystemExit(2)

log("done")
