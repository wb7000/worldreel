# Job: Probe adapter configurations on ONE short clip at deployment quality.
#
# Why this job exists (D-9.27):
#   * v5 training (character_lora_v5) produced NO usable adapter. Its reward loop
#     measured identity on a 10-step validation clip, but the deployment render
#     runs 50 steps: at 50 steps the motion-rich target is reachable, at 10 steps
#     the same adapter is off-distribution. Every trained checkpoint was scored
#     below the base model, so the "best" checkpoint stayed the untrained
#     initialisation (all 360 LoRA B matrices zero -> a no-op adapter).
#   * The v4 adapter is the only artifact that ever produced identity
#     (0.9256 mean) -- but at full strength it freezes the pose: the delivery
#     scored motion 0.484 with 180/215 still frames ("an image that shakes").
#
# This job renders the SAME 21-frame / 50-step clip (seed 42, portrait-seeded)
# once per configuration, so the only variable is the adapter. Two knobs:
#
#   strength  -- the LoRA delta scale (0 = base model, 1 = trained adapter).
#   schedule  -- WHEN the adapter acts during the denoising trajectory:
#                 "static": the same strength for every step.
#                 "late":   0 until the trajectory has progressed `start` of the
#                           way, then a linear ramp to the full strength at the
#                           last step. The early (high-noise) steps decide the
#                           pose and the trajectory -- i.e. the motion the user
#                           rejected the v4 render for -- and the late steps only
#                           refine what is already there.
#
# Every configuration is measured on the rendered clip: identity (SFace vs the
# reference portrait), face sharpness (Laplacian variance in the face box),
# motion (mean frame-to-frame difference) and the share of frames that barely
# change (< 0.5 grey levels -- the "shaking image" metric).
#
# Artefacts: out/probe_<name>.mp4, out/probe.json, out/probe.log.
#
# Usage:
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_render_lora_probe.py
#       --scene data:vacecast --scene data:facemodels --scene data:lorav4 --scene data:lorav5 --wait
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

# peft's LoRA dispatcher raises on an importable-but-old torchao (Kaggle ships
# 0.10.0, peft wants >=0.16) -- this killed character_lora_v4's add_adapter on
# run 1, and loading an adapter injects modules through the same dispatcher.
u = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    capture_output=True,
    text=True,
)
log(f"torchao uninstall rc={u.returncode} {u.stdout.strip()[-200:]}")

# ================================================================
# STEP 2: Find face detection/recognition models in mounted Kaggle dataset
# ================================================================
MODEL_DIR = Path("/kaggle/input/datasets/wb3none/worldreel-data-facemodels")

# Kaggle only archives /kaggle/working/out. Every artefact MUST be written there
# or it is silently lost (D-9.10).
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
    raise SystemExit(
        "FATAL: face_detection_yunet_2023mar.onnx not found in any Kaggle dataset. "
        "Make sure --scene data:facemodels is passed and the dataset is mounted."
    )
if sface_path is None:
    raise SystemExit(
        "FATAL: face_recognition_sface_2021dec.onnx not found in any Kaggle dataset. "
        "Make sure --scene data:facemodels is passed and the dataset is mounted."
    )

log(f"Using YuNet: {yunet_path}")
log(f"Using SFace: {sface_path}")

# ================================================================
# STEP 2b: Resolve the adapter directories of every configuration
# ================================================================
# Each config names "<dataset slug>/<subdir>". The datasets are mounted at
# /kaggle/input/datasets/wb3none/<slug>. A missing adapter must fail loudly:
# a silently unloaded adapter renders the base model and looks like a
# successful measurement of the wrong thing (D-9.24).
CONFIGS = [
    # 2x2 design: {4 references, 6 references} x {seed 42, seed 43}.
    #
    # WHY references: the base model's identity decays INSIDE every segment
    # (seg 0 0.9176 -> 0.8435 over 48 frames, seg 3 down to 0.7870) because the
    # reference-image conditioning weakens with the temporal distance from the
    # conditioning frame. The vacecast dataset holds SIX C01 portraits and the
    # render job caps the reference list at FOUR, so two views of the same
    # person are thrown away. More references = a stronger identity anchor, and
    # unlike the LoRA it cannot freeze the pose.
    #
    # WHY two seeds: the segment-to-segment decay rate varies a lot
    # (-0.00048 to -0.00285/frame), so one clip cannot separate "the change
    # worked" from "this seed drifted less". The second seed is the noise floor.
    {
        "name": "refs4_seed42",
        "dataset": "",
        "subdir": "",
        "strength": 0.0,
        "schedule": "static",
        "start": 0.0,
        "seed": 42,
        "num_references": 4,
        "guidance_scale": 5.0,
    },
    {
        "name": "refs6_seed42",
        "dataset": "",
        "subdir": "",
        "strength": 0.0,
        "schedule": "static",
        "start": 0.0,
        "seed": 42,
        "num_references": 6,
        "guidance_scale": 5.0,
    },
    {
        "name": "refs4_seed43",
        "dataset": "",
        "subdir": "",
        "strength": 0.0,
        "schedule": "static",
        "start": 0.0,
        "seed": 43,
        "num_references": 4,
        "guidance_scale": 5.0,
    },
    {
        "name": "refs6_seed43",
        "dataset": "",
        "subdir": "",
        "strength": 0.0,
        "schedule": "static",
        "start": 0.0,
        "seed": 43,
        "num_references": 6,
        "guidance_scale": 5.0,
    },
]

input_root = Path("/kaggle/input")


def resolve_adapter(dataset: str, subdir: str) -> str | None:
    """Find the adapter folder that holds pytorch_lora_weights.safetensors.

    Only the NAMED dataset is searched: a broad search would happily resolve
    "lorav4/lora_adapter_best" to the lorav5 folder, whose best checkpoint is
    the untrained no-op -- and the probe would then measure the base model
    while believing it measured the v4 adapter.
    """
    if not dataset:
        return None
    root = Path("/kaggle/input/datasets/wb3none") / dataset
    if not root.exists():
        return None
    direct = root / subdir
    if direct.exists() and next(direct.rglob("pytorch_lora_weights.safetensors"), None):
        return str(direct)
    for hit in root.rglob(subdir + "/pytorch_lora_weights.safetensors"):
        return str(hit.parent)
    return None


for cfg in CONFIGS:
    cfg["adapter_dir"] = resolve_adapter(cfg["dataset"], cfg["subdir"])
    if cfg["dataset"] and cfg["adapter_dir"] is None:
        short = cfg["dataset"].replace("worldreel-data-", "")
        raise SystemExit(
            f"FATAL: adapter {cfg['dataset']}/{cfg['subdir']} not found. Push the dataset first: "
            f"python -m tools.kaggle_runner push-data --name {short} <out folder> "
            f"and pass --scene data:{short}."
        )
    log(
        f"config {cfg['name']}: adapter={cfg['adapter_dir']} strength={cfg['strength']} "
        f"schedule={cfg['schedule']} seed={cfg['seed']} refs={cfg['num_references']} "
        f"guidance={cfg['guidance_scale']}"
    )

# The training report of the v5 run, if the dataset is mounted: it explains why
# the v5 "best" checkpoint is a no-op.
for dataset in ("worldreel-data-lorav5", "worldreel-data-lorav4"):
    report = Path("/kaggle/input/datasets/wb3none") / dataset
    if not report.exists():
        continue
    hit = next(report.rglob("lora_v*_report.json"), None)
    if hit is None:
        continue
    try:
        rep = json.loads(hit.read_text(encoding="utf-8"))
        log(
            f"training report {dataset}: baseline={rep.get('baseline')} "
            f"best={rep.get('best')} improvement={rep.get('improvement')} "
            f"roundtrip_ok={rep.get('roundtrip_ok')}"
        )
    except Exception as exc:
        log(f"could not read training report {hit}: {exc}")

# ================================================================
# STEP 3: Find the reference portrait
# ================================================================
PORTRAIT_DIR = Path("/tmp/worldreel_portraits")
PORTRAIT_DIR.mkdir(exist_ok=True)

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
    raise SystemExit(
        "FATAL: no C01_*.png reference portrait in the mounted datasets. "
        "Pass --scene data:vacecast."
    )

# ================================================================
# STEP 4: Write the step script
# ================================================================
# NOTE: do NOT redefine OUT_DIR here. Kaggle only archives /kaggle/working/out,
# so a stray "/tmp/kaggle_out" default silently discards every artefact (D-9.10).
WORK_DIR = Path(os.environ.get("WORK", "/tmp/kaggle_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

CODE_DIR = Path(os.environ.get("CODE", "."))

step = WORK_DIR / "step_lora_probe.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, math, sys, time, torch, cv2, numpy as np
    from pathlib import Path
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    log = print

    a = json.load(open(sys.argv[1]))

    W, H = a["width"], a["height"]
    N = a["frames_per_segment"]
    steps = int(a.get("steps", 50))
    model_id = a.get("model_id", "Wan-AI/WACE-1.3B-diffusers")
    out_dir = Path(a["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Configuration: %dx%d, %d frames, %d steps" % (W, H, N, steps))
    log("Portrait: %s" % a["portrait"])

    # ---- Face detection + recognition (the repo implementation, so the kernel
    # and the local QA use exactly one algorithm) ----
    from stages.generate.face_swap import FaceSwapper

    swapper = FaceSwapper(a["yunet"], a["sface"])
    detect_face = swapper.detect_face

    ref_paths = a.get("references") or [a["portrait"]]
    ref_imgs = []
    for rp in ref_paths:
        try:
            ref_imgs.append(Image.open(rp).convert("RGB"))
        except Exception as exc:
            log("skipping unreadable reference %s: %s" % (rp, exc))
    if not ref_imgs:
        raise SystemExit("no readable reference portraits")
    ref_img = ref_imgs[0]
    ref_bgr = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
    ref_face = detect_face(ref_bgr)
    if ref_face is None:
        raise SystemExit("no face found in reference portrait")
    ref_emb = swapper.compute_embedding(ref_bgr)
    if ref_emb is None:
        raise SystemExit("could not compute reference face embedding")
    log("Reference face embedding norm: %.4f" % float(np.linalg.norm(ref_emb)))

    # ---- Pipeline ----
    log("Loading Wan2.1-VACE pipeline...")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

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
        log("attention: xformers unavailable (%s); using SDPA" % type(exc).__name__)
    math_sdp_off = False
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        math_sdp_off = True
        log("attention: math SDPA disabled (memory-efficient only)")
    except Exception as exc:
        log("attention: could not restrict SDPA backends (%s)" % type(exc).__name__)

    # ---- Seed frame: primary portrait COVER-CROPPED to the output aspect ----
    from PIL import ImageOps

    seed = ImageOps.fit(ref_img, (W, H), method=Image.LANCZOS, centering=(0.5, 0.42))
    grey = Image.new("RGB", (W, H), (128, 128, 128))
    mask_black = Image.new("L", (W, H), 0)
    mask_white = Image.new("L", (W, H), 255)
    # The reference list is capped per configuration (num_references); the vacecast
    # dataset holds SIX C01 views of the same person and the render job used only
    # four of them.
    refs = ref_imgs[:8]

    # ---- The adapter schedule ------------------------------------------------
    # The transformer is called once per CFG branch with the current timestep, so
    # wrapping its forward lets the LoRA weight depend on WHERE the denoising
    # trajectory is. The schedule arithmetic lives in tools/lora_schedule.py so
    # it is unit-tested (tests/test_lora_schedule.py).
    from tools.lora_schedule import schedule_at

    state = {"mode": "static", "strength": 0.0, "start": 0.0, "timesteps": None}

    original_forward = pipe.transformer.forward

    def scheduled_forward(*call_args, **call_kwargs):
        if state["timesteps"] is None:
            try:
                state["timesteps"] = [float(x) for x in pipe.scheduler.timesteps]
            except Exception:
                state["timesteps"] = []
        t_value = None
        if "timestep" in call_kwargs and call_kwargs["timestep"] is not None:
            t_value = float(call_kwargs["timestep"].reshape(-1)[0].item())
        else:
            for value in call_args:
                if torch.is_tensor(value) and value.ndim == 1:
                    t_value = float(value.reshape(-1)[0].item())
                    break
        if t_value is not None:
            weight = schedule_at(
                state["mode"], state["strength"], state["start"], state["timesteps"], t_value
            )
            pipe.transformer.set_adapters("default", weights=[weight])
        return original_forward(*call_args, **call_kwargs)

    pipe.transformer.forward = scheduled_forward

    # ---- Probe every configuration on the SAME clip ----
    results = []
    for cfg in a["configs"]:
        name = cfg["name"]
        t0 = time.time()
        seed_value = int(cfg.get("seed", 42))
        n_refs = int(cfg.get("num_references", len(refs)))
        guidance = float(cfg.get("guidance_scale", 5.0))
        config_refs = refs[:n_refs]
        # (Re)load the adapter for this configuration. prefix=None is required:
        # the default prefix ("transformer") filters every saved key ("blocks.*")
        # out and loads nothing silently.
        if cfg.get("adapter_dir"):
            try:
                pipe.transformer.delete_adapters("default")
            except Exception:
                pass
            pipe.transformer.load_lora_adapter(
                cfg["adapter_dir"],
                weight_name="pytorch_lora_weights.safetensors",
                prefix=None,
                adapter_name="default",
            )
            if "default" not in getattr(pipe.transformer, "peft_config", {}):
                raise SystemExit("LoRA adapter did not register on load: %s" % name)
        state["mode"] = cfg.get("schedule", "static")
        state["strength"] = float(cfg.get("strength", 1.0))
        state["start"] = float(cfg.get("start", 0.0))
        state["timesteps"] = None
        log("config %s: adapter=%s strength=%.2f schedule=%s seed=%d refs=%d guidance=%.1f "
            "-- rendering %d frames"
            % (name, cfg.get("adapter_dir"), state["strength"], state["mode"],
               seed_value, len(config_refs), guidance, N))

        out = pipe(video=[seed, *[grey] * (N - 1)],
                   mask=[mask_black, *[mask_white] * (N - 1)],
                   reference_images=config_refs,
                   prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                   num_frames=N, num_inference_steps=steps, guidance_scale=guidance,
                   conditioning_scale=1.0, output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(seed_value)).frames[0]
        new_frames = out[1:]

        # ---- Measure the clip: identity, sharpness, motion ----
        sims = []
        sharps = []
        faces_found = 0
        embeddings = []
        for fr in new_frames:
            bgr = cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR)
            det = detect_face(bgr)
            if det is None:
                continue
            faces_found += 1
            emb = swapper.compute_embedding(bgr)
            if emb is not None:
                sims.append(float(emb @ ref_emb))
                embeddings.append(np.asarray(emb, dtype=np.float64))
            bx, by, bw, bh = (int(v) for v in det[0][:4])
            roi = bgr[max(0, by):by + bh, max(0, bx):bx + bw]
            if roi.size:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                sharps.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        # Self-consistency: every frame against the clip's OWN median face. This
        # is the question "does the face morph into somebody else during the
        # clip", which the absolute score against the reference photo mixes with
        # pose and lighting.
        self_mean = -1.0
        self_min = -1.0
        if embeddings:
            matrix = np.stack(embeddings)
            median = np.median(matrix, axis=0)
            norm = float(np.linalg.norm(median))
            if norm > 0:
                median = median / norm
                to_median = matrix @ median
                self_mean = float(np.mean(to_median))
                self_min = float(np.min(to_median))
        drift = (sims[-1] - sims[0]) / max(1, len(sims) - 1) if len(sims) > 1 else 0.0

        diffs = []
        for i in range(1, len(new_frames)):
            diffs.append(float(np.abs(
                np.asarray(new_frames[i - 1], dtype=np.float32)
                - np.asarray(new_frames[i], dtype=np.float32)).mean()))
        still = sum(1 for d in diffs if d < 0.5)

        entry = {
            "name": name,
            "adapter_dir": cfg.get("adapter_dir"),
            "strength": state["strength"],
            "schedule": state["mode"],
            "start": state["start"],
            "seed": seed_value,
            "num_references": len(config_refs),
            "guidance_scale": guidance,
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
            "faces_found": faces_found,
            "frames": len(new_frames),
            "motion_mean": round(float(np.mean(diffs)), 3) if diffs else 0.0,
            "motion_median": round(float(np.median(diffs)), 3) if diffs else 0.0,
            "motion_max": round(float(np.max(diffs)), 3) if diffs else 0.0,
            "still_frames": still,
            "still_total": len(diffs),
            "seconds": round(time.time() - t0, 1),
        }
        results.append(entry)
        log("RESULT %s identity %.4f (min %.4f, drift %+.5f/frame) self %.4f/%.4f "
            "sharpness %.1f motion %.3f still %d/%d %.0fs"
            % (name, entry["identity_mean"], entry["identity_min"],
               entry["identity_drift_per_frame"], entry["self_consistency_mean"],
               entry["self_consistency_min"], entry["sharpness_mean"],
               entry["motion_mean"], still, len(diffs), entry["seconds"]))
        export_to_video(new_frames, str(out_dir / ("probe_%s.mp4" % name)), fps=16)
        json.dump({"results": results}, open(str(out_dir / "probe.json"), "w"), indent=2)
        gc.collect()
        torch.cuda.empty_cache()

    log("PROBE SUMMARY " + json.dumps(results))
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
    "configs": CONFIGS,
    "frames_per_segment": 21,
    "width": 832,
    "height": 480,
    "steps": 50,
    "prompt": (
        "real video footage of the same woman shown in the reference photographs, "
        "standing still in a studio, breathing gently, blinking naturally, mouth closed, "
        "subtle natural head movement, fixed camera, head and shoulders framing, "
        "plain grey studio wall behind, soft even natural light, detailed natural skin "
        "with visible pores, natural hair, sharp focus, shot on 35mm film, photorealistic"
    ),
    "negative": (
        "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
        "blurry, deformed face, extra eyes, second face, overlapping face, double face, "
        "text, watermark, overexposed, dark blotches on skin, static, still picture, "
        "talking, open mouth, camera movement"
    ),
}

args_path = WORK_DIR / "lora_probe_args.json"
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

log_path = OUT_DIR / "probe.log"
stdout_part = r.stdout[-20000:] if r.stdout else ""
stderr_part = r.stderr[-8000:] if r.stderr else ""
log_path.write_text(
    (stdout_part + "\n" + stderr_part).strip() if r.returncode else stdout_part.strip(),
    encoding="utf-8",
)

log(f"Probe job rc={r.returncode}")
if r.returncode != 0:
    log(f"STDERR: {r.stderr[-2000:]}")
else:
    log("Job completed successfully!")

# ================================================================
# STEP 6: Recover + verify artefacts
# ================================================================
# Kaggle archives ONLY /kaggle/working/out (D-9.10). The names below must match
# what the step script writes -- a stale check made the v4 sweep report ERROR
# after it had already printed DONE - SUCCESS.
for stray in (Path("/tmp/kaggle_out"), WORK_DIR / "out"):
    if not stray.exists():
        continue
    for p in sorted(stray.iterdir()):
        if p.is_file() and not (OUT_DIR / p.name).exists():
            shutil.copy2(p, OUT_DIR / p.name)
            log(f"Recovered stray output -> {p.name}")

expected = ["probe.json", "probe.log"] + [f"probe_{cfg['name']}.mp4" for cfg in CONFIGS]
for name in expected:
    p = OUT_DIR / name
    log(f"artefact {name}: {'MISSING' if not p.exists() else f'{p.stat().st_size} bytes'}")

if not (OUT_DIR / "probe.json").exists():
    log("FATAL: probe.json was not produced inside the archived output directory")
    raise SystemExit(2)

log("done")
