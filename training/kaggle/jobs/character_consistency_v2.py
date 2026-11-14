# Job: Generate character-consistent PHOTOREALISTIC video on Kaggle
# Simplified, robust version: downloads models directly, no scene dependencies
# Usage: python -m tools.kaggle_runner run training/kaggle/jobs/character_consistency_v2.py --wait
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
# STEP 2: Find face detection/recognition models in mounted Kaggle dataset
# ================================================================
# The dataset worldreel-data-facemodels is mounted at:
# /kaggle/input/datasets/wb3none/worldreel-data-facemodels/
# when we add --scene data:facemodels to the run command.
MODEL_DIR = Path("/kaggle/input/datasets/wb3none/worldreel-data-facemodels")

# Kaggle only archives /kaggle/working/out. Every artefact MUST be written there or it is
# silently lost (D-9.10). The bootstrap exports OUT; the fallback keeps the job standalone.
OUT_DIR = Path(os.environ.get("OUT", "/kaggle/working/out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

yunet_path = None
sface_path = None

# Search the mounted dataset for the ONNX models
if MODEL_DIR.exists():
    for p in MODEL_DIR.rglob("face_detection_yunet_2023mar.onnx"):
        yunet_path = str(p)
        log(f"Found YuNet model: {yunet_path}")
        break
    for p in MODEL_DIR.rglob("face_recognition_sface_2021dec.onnx"):
        sface_path = str(p)
        log(f"Found SFace model: {sface_path}")
        break

# Also search the broader /kaggle/input tree in case the mount path differs
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

# Clear error with early exit if models are missing — makes failures obvious
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
# STEP 3: Find or create a reference portrait
# ================================================================
PORTRAIT_DIR = Path("/tmp/worldreel_portraits")
PORTRAIT_DIR.mkdir(exist_ok=True)

# Try to find portrait in Kaggle input datasets
portrait_path = None
input_root = Path("/kaggle/input")
if input_root.exists():
    for p in input_root.rglob("C01_*.png"):
        portrait_path = str(p)
        log(f"Found portrait: {portrait_path}")
        break
    for p in input_root.rglob("C01_*.jpg"):
        portrait_path = str(p)
        log(f"Found portrait: {portrait_path}")
        break

if not portrait_path:
    # Create a synthetic test portrait
    import cv2
    import numpy as np

    img = np.zeros((480, 832, 3), dtype=np.uint8)
    img[:] = (120, 100, 80)  # skin tone background
    cv2.ellipse(img, (416, 240), (160, 200), 0, 0, 360, (220, 180, 160), -1)
    cv2.ellipse(img, (384, 200), (30, 40), 0, 0, 360, (200, 160, 140), -1)
    cv2.ellipse(img, (448, 200), (30, 40), 0, 0, 360, (200, 160, 140), -1)
    cv2.ellipse(img, (416, 240), (60, 70), 0, 0, 180, (200, 150, 140), -1)
    portrait_path = str(PORTRAIT_DIR / "C01_synthetic.png")
    cv2.imwrite(portrait_path, img)
    log(f"Created synthetic portrait: {portrait_path}")

# ================================================================
# STEP 4: Write the step script
# ================================================================
# NOTE: do NOT redefine OUT_DIR here. Kaggle only archives /kaggle/working/out, so a stray
# "/tmp/kaggle_out" default silently discards every artefact (D-9.10).
WORK_DIR = Path(os.environ.get("WORK", "/tmp/kaggle_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

CODE_DIR = Path(os.environ.get("CODE", "."))

step = WORK_DIR / "step_face_consistency.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, sys, time, traceback, torch, cv2, numpy as np
    from pathlib import Path
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    log = print

    # ---- Load arguments ----
    args_path = sys.argv[1]
    with open(args_path) as f:
        a = json.load(f)

    W, H = a["width"], a["height"]
    N = a["frames_per_segment"]
    OVERLAP = a["overlap_frames"]
    model_id = a.get("model_id", "Wan-AI/Wan2.1-VACE-1.3B-diffusers")

    log(f"Configuration: {W}x{H}, {N} frames/segment, overlap={OVERLAP}")
    log(f"Model: {model_id}")
    log(f"Portrait: {a['portrait']}")
    log(f"YuNet: {a['yunet']}")
    log(f"SFace: {a['sface']}")

    # ---- Face detection + recognition ----
    # Uses the repo implementation (shipped in worldreel_code.zip) so local and
    # Kaggle runs share exactly one algorithm. We never call
    # cv2.FaceRecognizerSF.alignCrop: it is broken in OpenCV 5.x (returns an
    # all-black crop, so every embedding collapses to the same constant) and
    # its landmark order differs from YuNet's. The repo aligns with the
    # canonical ArcFace/SFace template instead (stages/generate/face_align.py).
    from stages.generate.face_swap import FaceSwapper

    swapper = FaceSwapper(a["yunet"], a["sface"])
    detect_face = swapper.detect_face
    swap_face_in_frame = swapper.swap_face

    # ---- Load reference portrait ----
    ref_img = Image.open(a["portrait"]).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
    ref_face = detect_face(ref_bgr)
    if ref_face is None:
        raise SystemExit("no face found in reference portrait")
    ref_bbox, ref_lm = ref_face
    log(f"Reference face bbox: {ref_bbox.tolist()}")

    ref_emb = swapper.compute_embedding(ref_bgr)
    if ref_emb is None:
        raise SystemExit("could not compute reference face embedding")
    log(f"Reference face embedding norm: {float(np.linalg.norm(ref_emb)):.4f}")

    # ---- Load Wan2.1-VACE pipeline ----
    log("Loading Wan2.1-VACE pipeline...")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

    # Encode prompts
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

    # ---- Face identity injector into transformer ----
    class FaceConditionedTransformer:
        def __init__(self, base_transformer, face_embedding_dim=128):
            self.base = base_transformer
            self.face_embedding_dim = face_embedding_dim
            hidden_dim = (getattr(base_transformer.config, "hidden_size", None)
                          or getattr(base_transformer.config, "inner_dim", None)
                          or 4096)
            self.hidden_dim = hidden_dim
            self.face_projector = torch.nn.Linear(face_embedding_dim, hidden_dim)
            self.face_projector.weight.data.zero_()
            if face_embedding_dim <= hidden_dim:
                self.face_projector.weight.data[:face_embedding_dim, :face_embedding_dim] = torch.eye(face_embedding_dim)
            self.face_projector.bias.data.zero_()
            self.scale = 10.0

        @property
        def config(self):
            return self.base.config

        @property
        def dtype(self):
            return self.base.dtype

        @property
        def device(self):
            return self.base.device

        def __call__(self, hidden_states=None, encoder_hidden_states=None,
                     timestep=None, **kwargs):
            if encoder_hidden_states is not None:
                device = hidden_states.device
                dtype = hidden_states.dtype
                ref_emb_t = torch.from_numpy(ref_emb).unsqueeze(0).to(device, dtype)
                self.face_projector = self.face_projector.to(device, dtype)
                face_tokens = self.face_projector(ref_emb_t)
                face_tokens = face_tokens * self.scale
                face_tokens = face_tokens.unsqueeze(1)
                encoder_hidden_states = torch.cat([face_tokens, encoder_hidden_states], dim=1)
            return self.base(hidden_states=hidden_states,
                             encoder_hidden_states=encoder_hidden_states,
                             timestep=timestep, **kwargs)

        def __getattr__(self, name):
            return getattr(self.base, name)

    pipe.transformer = FaceConditionedTransformer(pipe.transformer)
    log("Face identity injector installed in transformer")

    # ---- Generate video segments ----
    x = (W - H) // 2
    first = Image.new("RGB", (W, H), (128, 128, 128))
    first.paste(ref_img.resize((H, H), Image.LANCZOS), (x, 0))

    all_frames = []
    report = {"segments": [], "reference_portrait": a["portrait"],
              "total_seconds": 0.0}
    prev_last_frames = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)

    for s in range(a["segments"]):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        if s == 0:
            video = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)
            mask = [Image.new("L", (W, H), 0)] + [Image.new("L", (W, H), 255)] * (N - 1)
            reference_images = [ref_img]
        else:
            video = [*prev_last_frames[-OVERLAP:],
                     *[Image.new("RGB", (W, H), (128, 128, 128))] * (N - OVERLAP)]
            mask = [Image.new("L", (W, H), 0)] * OVERLAP + \
                   [Image.new("L", (W, H), 255)] * (N - OVERLAP)
            reference_images = [ref_img, prev_last_frames[-1]]

        log(f"Segment {s}: starting generation")
        out = pipe(video=video, mask=mask, reference_images=reference_images,
                   prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                   num_frames=N, num_inference_steps=30, guidance_scale=5.0,
                   conditioning_scale=1.0, output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(42 + s)).frames[0]

        new_frames = out[OVERLAP:] if s > 0 else out

        # ---- Face swap post-processing on every new frame ----
        swapped_frames = []
        for i, frame in enumerate(new_frames):
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            swapped = swap_face_in_frame(frame_bgr, ref_bgr)
            swapped_frames.append(Image.fromarray(cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)))

        all_frames += swapped_frames
        prev_last_frames = out

        # Measure identity on middle frame
        mid = OVERLAP + (N - OVERLAP) // 2 if s > 0 else N // 2
        frame_bgr = cv2.cvtColor(np.array(out[mid]), cv2.COLOR_RGB2BGR)
        swapped_idx = mid - (OVERLAP if s > 0 else 0)
        swapped_bgr = cv2.cvtColor(np.array(swapped_frames[swapped_idx]), cv2.COLOR_RGB2BGR)
        face = detect_face(frame_bgr)
        swapped_face = detect_face(swapped_bgr)
        # REAL cosine similarity, not "a face was still detectable" (D-9.9/D-9.11).
        pre_emb = swapper.compute_embedding(frame_bgr)
        post_emb = swapper.compute_embedding(swapped_bgr)
        pre_sim = float(pre_emb @ ref_emb) if pre_emb is not None else -1.0
        post_sim = float(post_emb @ ref_emb) if post_emb is not None else -1.0
        seg_seconds = (N - OVERLAP) / 16.0 if s > 0 else N / 16.0
        report["total_seconds"] += seg_seconds
        report["segments"].append({
            "segment": s,
            "seconds": round(time.time() - t0, 1),
            "video_seconds": round(seg_seconds, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "face_found": face is not None,
            "face_swapped": swapped_face is not None,
            "pre_swap_sim": round(pre_sim, 4),
            "post_swap_sim": round(post_sim, 4),
            "sim_gain": round(post_sim - pre_sim, 4),
        })
        log(f"Segment {s}: face_found={face is not None} face_swapped={swapped_face is not None} "
            f"pre_sim={pre_sim:.4f} post_sim={post_sim:.4f} gain={post_sim - pre_sim:+.4f}")
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Export video ----
    out_dir = Path(a["out"])
    out_mp4 = str(out_dir / "face_consistency.mp4")
    export_to_video(all_frames, out_mp4, fps=16)
    log(f"Video exported: {out_mp4}")

    json.dump(report, open(str(out_dir / "face_consistency.json"), "w"), indent=2)
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
    "yunet": str(yunet_path),
    "sface": str(sface_path),
    "out": str(OUT_DIR),
    "segments": 3,
    "frames_per_segment": 48,
    "overlap_frames": 6,
    "width": 512,
    "height": 288,
    "prompt": "real video footage of a person standing still, breathing gently, blinking naturally, mouth closed, fixed camera, head and shoulders, plain grey studio wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, static, still picture, talking, open mouth, camera movement",
}

args_path = WORK_DIR / "face_consistency_args.json"
args_path.write_text(json.dumps(args, indent=2))

env = {
    **os.environ,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "OUT": str(OUT_DIR),
    "WORK": str(WORK_DIR),
    "CODE": str(CODE_DIR),
    # The step script imports stages.generate.face_swap from the uploaded code
    # dataset; without PYTHONPATH the subprocess only sees its own directory.
    "PYTHONPATH": str(CODE_DIR),
}

r = subprocess.run(
    [str(sys.executable), str(step), str(args_path)],
    capture_output=True,
    text=True,
    timeout=6 * 3600,
    env=env,
)

log_path = OUT_DIR / "face_consistency.log"
stdout_part = r.stdout[-20000:] if r.stdout else ""
stderr_part = r.stderr[-8000:] if r.stderr else ""
log_path.write_text(
    (stdout_part + "\\n" + stderr_part).strip() if r.returncode else stdout_part.strip(),
    encoding="utf-8",
)

log(f"Face consistency job rc={r.returncode}")
if r.returncode != 0:
    log(f"STDERR: {r.stderr[-2000:]}")
else:
    log("Job completed successfully!")

# ================================================================
# STEP 6: Recover + verify artefacts
# ================================================================
# Kaggle archives ONLY /kaggle/working/out. If anything landed elsewhere it is lost, so
# sweep known stray locations and then hard-verify the expected files exist (D-9.10).
for stray in (Path("/tmp/kaggle_out"), WORK_DIR / "out"):
    if not stray.exists():
        continue
    for p in sorted(stray.iterdir()):
        if p.is_file() and not (OUT_DIR / p.name).exists():
            shutil.copy2(p, OUT_DIR / p.name)
            log(f"Recovered stray output -> {p.name}")

expected = ["face_consistency.mp4", "face_consistency.json", "face_consistency.log"]
present = {name: (OUT_DIR / name) for name in expected}
for name, p in present.items():
    size = p.stat().st_size if p.exists() else -1
    log(f"artefact {name}: {'MISSING' if size < 0 else f'{size} bytes'}")

if not present["face_consistency.mp4"].exists():
    log("FATAL: face_consistency.mp4 was not produced inside the archived output directory")
    raise SystemExit(2)

log("done")
