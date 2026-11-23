# Job: D-9.29 probe -- normalised references + a low-denoise sharpening pass.
#
# The user rejected S017_refs6 ("there is still the blur and inconsistency"), and
# the free measurements (D-9.29) localised four visible defects the metric table
# could not see: softness that comes and goes, a head-scale walk inside each
# segment, per-segment lighting jumps, and gaze variation. Two of the four are
# caused by the conditioning set itself -- the six references are 512x512 squares
# with light-grey-to-black backgrounds and mixed lighting, and diffusers' VACE
# preprocessing letterboxes every reference onto a WHITE canvas
# (pipeline_wan_vace.py:508), so the model sees 480x480 images with 176 px white
# bars. The third -- softness -- is what a second pass at low denoise fixes
# (the community-standard sharpening move, proven on this T4 in D-8.12's
# realism_pass.py).
#
# Four arms in one job, so one GPU run separates the two levers:
#
#   arm A  probe_refs6_seed43.mp4            control (the D-9.28 probe's 6-ref
#                                             seed-43 clip, pushed as data)
#   arm B  render(seed 43, normalised refs,  new conditioning, no pass
#          camera-lock + gaze + flicker negatives)
#   arm D  arm B repainted at strength 0.25  the sharpen pass, gentle dose
#   arm G  arm B repainted at strength 0.35  the sharpen pass, stronger dose
#   arm C  arm A repainted at strength 0.25  "can the EXISTING delivery be
#                                             fixed without a re-render?"
#
# B-A isolates the conditioning change; D-B and G-B isolate the pass (dose
# response); C-A is the cheapest possible visible win. Every arm is scored with
# the probe job's own metrics (identity vs the ORIGINAL C01_0, self-consistency
# vs the clip's own median face, face Laplacian, motion, still frames) so the
# numbers line up with the D-9.28 probe table.
#
# The sharpen pass reuses the render's prompt embeddings: the V2V pipeline is
# loaded with text_encoder=None / tokenizer=None (diffusers explicitly supports
# a null component, pipeline_utils.py:972), which skips the ~22 GB T5 download
# and keeps the job at one big model on disk.
#
# Usage:
#   python -m tools.kaggle_runner push-data runs/kaggle/character_render_lora_probe/out \
#       --name probebase
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_probe_sharp.py \
#       --scene data:vacecast --scene data:facemodels --scene data:probebase \
#       --requeue-after 30 --wait
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# ================================================================
# STEP 1: Install required packages
# ================================================================
# No peft: D-9.28/D-9.27 closed the adapter direction, this probe runs the base
# model only.
r = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "diffusers",
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

# ================================================================
# STEP 2: Face detection/recognition models from the mounted dataset
# ================================================================
MODEL_DIR = Path("/kaggle/input/datasets/wb3none/worldreel-data-facemodels")

# Kaggle only archives /kaggle/working/out; every artefact must land there (D-9.10).
OUT_DIR = Path(os.environ.get("OUT", "/kaggle/working/out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

yunet_path = None
sface_path = None
if MODEL_DIR.exists():
    for p in MODEL_DIR.rglob("face_detection_yunet_2023mar.onnx"):
        yunet_path = str(p)
        break
    for p in MODEL_DIR.rglob("face_recognition_sface_2021dec.onnx"):
        sface_path = str(p)
        break

input_root = Path("/kaggle/input")
if (yunet_path is None or sface_path is None) and input_root.exists():
    if yunet_path is None:
        for p in input_root.rglob("face_detection_yunet_2023mar.onnx"):
            yunet_path = str(p)
            break
    if sface_path is None:
        for p in input_root.rglob("face_recognition_sface_2021dec.onnx"):
            sface_path = str(p)
            break

if yunet_path is None or sface_path is None:
    raise SystemExit(
        "FATAL: face models not found -- mount them with --scene data:facemodels"
    )
log(f"YuNet={yunet_path}")
log(f"SFace={sface_path}")

# ================================================================
# STEP 3: References (originals -- normalised inside the step script) + baseline
# ================================================================
portrait_path = None
reference_paths: list[str] = []
if input_root.exists():
    for ext in ("png", "jpg", "jpeg"):
        found = sorted(input_root.rglob(f"C01_*.{ext}"))
        if found:
            reference_paths = [str(p) for p in found]
            portrait_path = reference_paths[0]
            break
if not portrait_path:
    raise SystemExit("FATAL: C01_* references not found -- mount data:vacecast")
log(f"{len(reference_paths)} references; metric reference (original)={portrait_path}")

# The control clip: the D-9.28 probe's 6-ref / seed-43 arm, pushed as a dataset.
baseline_path = None
for p in sorted(input_root.rglob("probe_refs6_seed43.mp4")):
    baseline_path = str(p)
    break
if baseline_path is None:
    raise SystemExit("FATAL: probe_refs6_seed43.mp4 not found -- mount data:probebase")
log(f"baseline control clip={baseline_path}")

# ================================================================
# STEP 4: Write the step script
# ================================================================
WORK_DIR = Path(os.environ.get("WORK", "/tmp/kaggle_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR = Path(os.environ.get("CODE", "."))

step = WORK_DIR / "step_probe_sharp.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, sys, time, torch, cv2, numpy as np
    from pathlib import Path
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline, WanVideoToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    a = json.load(open(sys.argv[1]))
    log = print
    W, H = a["width"], a["height"]
    N = int(a["frames_per_segment"])
    steps = int(a["steps"])
    out_dir = Path(a["out"])
    t_start = time.time()

    from stages.generate.face_swap import FaceSwapper

    swapper = FaceSwapper(a["yunet"], a["sface"])
    detect_face = swapper.detect_face
    ref_bgr = cv2.imread(a["portrait"])
    if ref_bgr is None:
        raise SystemExit("cannot read the metric reference portrait")
    ref_emb = swapper.compute_embedding(np.asarray(ref_bgr, dtype=np.uint8))
    if ref_emb is None:
        raise SystemExit("no face embedding in the metric reference")

    def measure(frames, name):
        # Identical definitions to character_render_lora_probe.py so the numbers
        # line up with the D-9.28 probe table.
        sims, sharps, embeds = [], [], []
        faces = 0
        for fr in frames:
            bgr = cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR)
            det = detect_face(bgr)
            if det is None:
                continue
            faces += 1
            emb = swapper.compute_embedding(bgr)
            if emb is not None:
                sims.append(float(emb @ ref_emb))
                embeds.append(np.asarray(emb, dtype=np.float64))
            bx, by, bw, bh = (int(v) for v in det[0][:4])
            roi = bgr[max(0, by):by + bh, max(0, bx):bx + bw]
            if roi.size:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                sharps.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        self_mean = self_min = -1.0
        if embeds:
            matrix = np.stack(embeds)
            median = np.median(matrix, axis=0)
            norm = float(np.linalg.norm(median))
            if norm > 0:
                median = median / norm
                to_median = matrix @ median
                self_mean = float(np.mean(to_median))
                self_min = float(np.min(to_median))
        drift = (sims[-1] - sims[0]) / max(1, len(sims) - 1) if len(sims) > 1 else 0.0
        diffs = [
            float(np.abs(np.asarray(frames[i - 1], dtype=np.float32)
                         - np.asarray(frames[i], dtype=np.float32)).mean())
            for i in range(1, len(frames))
        ]
        return {
            "name": name,
            "identity_mean": round(float(np.mean(sims)), 4) if sims else -1.0,
            "identity_min": round(float(np.min(sims)), 4) if sims else -1.0,
            "identity_first": round(float(sims[0]), 4) if sims else -1.0,
            "identity_last": round(float(sims[-1]), 4) if sims else -1.0,
            "identity_drift_per_frame": round(float(drift), 5),
            "self_consistency_mean": round(self_mean, 4),
            "self_consistency_min": round(self_min, 4),
            "identity_sims": [round(float(s), 4) for s in sims],
            "sharpness_mean": round(float(np.mean(sharps)), 2) if sharps else -1.0,
            "sharpness_min": round(float(np.min(sharps)), 2) if sharps else -1.0,
            "faces_found": faces,
            "frames": len(frames),
            "motion_mean": round(float(np.mean(diffs)), 3) if diffs else 0.0,
            "motion_median": round(float(np.median(diffs)), 3) if diffs else 0.0,
            "still_frames": sum(1 for d in diffs if d < 0.5),
            "still_total": len(diffs),
        }

    def save(results, extra=None):
        payload = {"results": results}
        if extra:
            payload.update(extra)
        json.dump(payload, open(str(out_dir / "probe_sharp.json"), "w"), indent=2)

    results = []

    # ---- Arm A: the pushed control clip, measured before any GPU work ----
    cap = cv2.VideoCapture(a["baseline"])
    frames_a = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames_a.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if len(frames_a) < 5:
        raise SystemExit("baseline clip unreadable or too short")
    entry = measure(frames_a, "A_baseline")
    entry["seconds"] = round(time.time() - t_start, 1)
    results.append(entry)
    log("RESULT A identity %.4f (min %.4f) self %.4f/%.4f sharp %.1f motion %.3f"
        % (entry["identity_mean"], entry["identity_min"],
           entry["self_consistency_mean"], entry["self_consistency_min"],
           entry["sharpness_mean"], entry["motion_mean"]))
    save(results)

    # ---- D-9.29 lever 1: normalise the references ----
    # Same module the local contact sheet used (tools/prepare_refs.py): output
    # aspect (no white letterbox bars from preprocess_conditions), one face scale,
    # one lighting. Written into the archived output for inspection.
    from tools.prepare_refs import normalize_references

    norm_paths = normalize_references(
        a["references"], out_dir / "refs_norm",
        yunet_path=a["yunet"], sface_path=a["sface"], target=0,
        tone_strength=float(a.get("tone", 1.0)),
    )
    norm_paths = norm_paths[: int(a.get("num_references", 6))]
    ref_imgs = [Image.open(p).convert("RGB") for p in norm_paths]
    log("normalised references: %s" % [Path(p).name for p in norm_paths])

    seed = ref_imgs[0]  # already W x H: the conditioning frame is no longer an upscale
    grey = Image.new("RGB", (W, H), (128, 128, 128))
    mask_black = Image.new("L", (W, H), 0)
    mask_white = Image.new("L", (W, H), 255)

    # ---- Arm B: one render with the new conditioning ----
    model_id = a["model_id"]
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae",
                                            torch_dtype=torch.float16)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config,
                                                         flow_shift=3.0)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log("attention: xformers memory-efficient enabled")
    except Exception as exc:
        log("attention: xformers unavailable (%s)" % type(exc).__name__)
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        log("attention: math SDPA disabled")
    except Exception as exc:
        log("attention: could not restrict SDPA backends (%s)" % type(exc).__name__)

    pipe.text_encoder.to("cuda")
    with torch.no_grad():
        pos, neg = pipe.encode_prompt(prompt=a["prompt"], negative_prompt=a["negative"],
                                      do_classifier_free_guidance=True,
                                      device=torch.device("cuda"))
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    out = pipe(video=[seed, *[grey] * (N - 1)], mask=[mask_black, *[mask_white] * (N - 1)],
               reference_images=ref_imgs, prompt_embeds=pos, negative_prompt_embeds=neg,
               height=H, width=W, num_frames=N, num_inference_steps=steps,
               guidance_scale=float(a.get("guidance_scale", 5.0)),
               conditioning_scale=1.0, output_type="pil",
               generator=torch.Generator("cpu").manual_seed(int(a["seed"]))).frames[0]
    frames_b = out[1:]  # the conditioned portrait frame is dropped, as in every render
    export_to_video(frames_b, str(out_dir / "probeB.mp4"), fps=16)
    entry = measure(frames_b, "B_newrefs")
    entry["seconds"] = round(time.time() - t0, 1)
    entry["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    results.append(entry)
    log("RESULT B identity %.4f (min %.4f) self %.4f/%.4f sharp %.1f motion %.3f %.0fs"
        % (entry["identity_mean"], entry["identity_min"],
           entry["self_consistency_mean"], entry["self_consistency_min"],
           entry["sharpness_mean"], entry["motion_mean"], entry["seconds"]))
    save(results, {"peak_vace_gb": entry["peak_gb"]})
    del pipe, vae, out
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Arms D/G/C: the low-denoise second pass (D-8.12 pattern) ----
    # The prompt embeds from the render are reused: the V2V pipeline loads with
    # text_encoder=None, so no second T5 download and no second encode.
    t2v = a["model_t2v"]
    vae2 = AutoencoderKLWan.from_pretrained(t2v, subfolder="vae",
                                             torch_dtype=torch.float32)
    rpipe = WanVideoToVideoPipeline.from_pretrained(
        t2v, vae=vae2, text_encoder=None, tokenizer=None, torch_dtype=torch.float16)
    rpipe.scheduler = UniPCMultistepScheduler.from_config(rpipe.scheduler.config,
                                                          flow_shift=3.0)
    try:
        rpipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        log("refine attention: xformers unavailable (%s)" % type(exc).__name__)
    rpipe.enable_model_cpu_offload()
    rpipe.vae.enable_tiling()  # fp32 VAE OOMs on 33+ frames without tiling (D-8.12)

    arms = [("probeB_ref25", frames_b, 0.25), ("probeB_ref35", frames_b, 0.35),
            ("probeA_ref25", frames_a, 0.25)]

    def pad_for_vae(frames):
        # The Wan VAE encodes 1 + 4k frames and SILENTLY IGNORES the rest
        # (autoencoder_kl_wan._encode reads x[:, :, 13:17] for i=4 and never
        # touches frame 17+), so a 20-frame clip would come back as 17 frames
        # with its tail chopped. Pad by repeating the last frame to the next
        # 4k+1, then trim the result back to the real length.
        n = len(frames)
        need = (4 - ((n - 1) % 4)) % 4
        return (frames + [frames[-1]] * need) if need else frames, n

    for tag, src, strength in arms:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        padded, real_n = pad_for_vae(src)
        refined = rpipe(video=padded, prompt_embeds=pos, negative_prompt_embeds=neg,
                        height=H, width=W, guidance_scale=5.0, strength=strength,
                        num_inference_steps=int(a.get("refine_steps", 30)),
                        output_type="pil",
                        generator=torch.Generator("cpu").manual_seed(7)).frames[0]
        refined = refined[:real_n]
        export_to_video(refined, str(out_dir / ("%s.mp4" % tag)), fps=16)
        entry = measure(refined, tag)
        entry["seconds"] = round(time.time() - t0, 1)
        entry["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        entry["strength"] = strength
        results.append(entry)
        log("RESULT %s identity %.4f (min %.4f) self %.4f/%.4f sharp %.1f motion %.3f %.0fs"
            % (tag, entry["identity_mean"], entry["identity_min"],
               entry["self_consistency_mean"], entry["self_consistency_min"],
               entry["sharpness_mean"], entry["motion_mean"], entry["seconds"]))
        save(results, {"peak_v2v_gb": entry["peak_gb"]})
        del refined
        gc.collect()
        torch.cuda.empty_cache()

    log("PROBE SHARP " + json.dumps(results))
    log("DONE - SUCCESS")
    """),
    encoding="utf-8",
)

# ================================================================
# STEP 5: Write args and run
# ================================================================
POSITIVE = (
    "real video footage of the same woman shown in the reference photographs, "
    "standing still in a studio, breathing gently, blinking naturally, mouth closed, "
    "calm neutral expression, eyes open looking directly at the camera, "
    "subtle natural head movement, locked-off tripod camera, no camera movement, "
    "no zoom, consistent head and shoulders framing throughout, "
    "plain grey studio wall behind, soft even natural light, consistent lighting, "
    "detailed natural skin with visible pores, natural hair, sharp focus, crisp detail, "
    "shot on 35mm film, photorealistic"
)
# D-9.29: the flicker/morphing terms are the community's standard guard against
# texture shimmer and face morphing; the camera-lock terms attack the head-scale
# walk the contact sheets exposed.
NEGATIVE = (
    "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
    "blurry, soft focus, out of focus, deformed face, extra eyes, second face, "
    "overlapping face, double face, morphing face, changing face, inconsistent face, "
    "warped face, flickering, jitter, strobing, pulsing brightness, "
    "text, watermark, overexposed, dark blotches on skin, static, still picture, "
    "talking, open mouth, camera movement, camera zoom, dolly, pull back, push in, "
    "handheld shake"
)

args = {
    "model_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    # Loaded WITHOUT text_encoder/tokenizer: the refine arms reuse the render's
    # prompt embeds (see the header), which skips the ~22 GB T5 download.
    "model_t2v": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    # The ORIGINAL primary portrait stays the metric reference so every number is
    # comparable with the D-9.28 probe table; the references are normalised in-step.
    "portrait": portrait_path,
    "references": reference_paths,
    "yunet": str(yunet_path),
    "sface": str(sface_path),
    "out": str(OUT_DIR),
    "baseline": baseline_path,
    "num_references": 6,
    # Same clip, same seed and steps as the D-9.28 probe's best arm (6/43):
    # 21 frames generated, the conditioned frame dropped, 20 frames exported.
    "seed": 43,
    "width": 832,
    "height": 480,
    "frames_per_segment": 21,
    "steps": 50,
    "refine_steps": 30,
    "guidance_scale": 5.0,
    "tone": 1.0,
    "prompt": POSITIVE,
    "negative": NEGATIVE,
}

args_path = WORK_DIR / "probe_sharp_args.json"
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
    timeout=5 * 3600,
    env=env,
)

log_path = OUT_DIR / "probe_sharp.log"
stdout_part = r.stdout[-20000:] if r.stdout else ""
stderr_part = r.stderr[-8000:] if r.stderr else ""
log_path.write_text(
    (stdout_part + "\n" + stderr_part).strip() if r.returncode else stdout_part.strip(),
    encoding="utf-8",
)
log(f"probe sharp rc={r.returncode}")
if r.returncode != 0:
    log(f"STDERR: {r.stderr[-2000:]}")
else:
    log("Job completed successfully!")

# ================================================================
# STEP 6: Recover + verify artefacts (D-9.10)
# ================================================================
for stray in (Path("/tmp/kaggle_out"), WORK_DIR / "out"):
    if not stray.exists():
        continue
    for p in sorted(stray.iterdir()):
        if p.is_file() and not (OUT_DIR / p.name).exists():
            import shutil

            shutil.copy2(p, OUT_DIR / p.name)
            log(f"Recovered stray output -> {p.name}")

expected = [
    "probeB.mp4",
    "probeB_ref25.mp4",
    "probeB_ref35.mp4",
    "probeA_ref25.mp4",
    "probe_sharp.json",
    "probe_sharp.log",
]
for name in expected:
    p = OUT_DIR / name
    size = p.stat().st_size if p.exists() else -1
    log(f"artefact {name}: {'MISSING' if size < 0 else f'{size} bytes'}")

if not (OUT_DIR / "probe_sharp.json").exists():
    log("FATAL: probe_sharp.json was not produced inside the archived output directory")
    raise SystemExit(2)

log("done")
