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

# Try to find portraits in the Kaggle input datasets.
# Collect EVERY C01_* frame of the same identity, not just the first: VACE's
# `reference_images` accepts a list, and several views of one person pin the
# identity down far harder than a single image (one reference is a weak hint).
portrait_path = None
reference_paths: list[str] = []
input_root = Path("/kaggle/input")
if input_root.exists():
    for ext in ("png", "jpg", "jpeg"):
        found = sorted(input_root.rglob(f"C01_*.{ext}"))
        if found:
            reference_paths = [str(p) for p in found]
            portrait_path = reference_paths[0]
            log(f"Found {len(reference_paths)} reference portraits; primary={portrait_path}")
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
    reference_paths = [portrait_path]
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
    model_id = a.get("model_id", "Wan-AI/Wan2.1-VACE-1.3B-diffusers")

    log(f"Configuration: {W}x{H}, {N} frames/segment, portrait-seeded")
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

    # ---- Load reference portraits ----
    # VACE takes a LIST of reference images. Several views of the same identity
    # anchor the character far more strongly than a single photo, which is what
    # keeps the face consistent across segments WITHOUT a post-hoc face swap.
    ref_paths = a.get("references") or [a["portrait"]]
    ref_imgs = []
    for rp in ref_paths:
        try:
            ref_imgs.append(Image.open(rp).convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            log(f"skipping unreadable reference {rp}: {exc}")
    if not ref_imgs:
        raise SystemExit("no readable reference portraits")
    ref_img = ref_imgs[0]
    ref_bgr = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
    ref_face = detect_face(ref_bgr)
    if ref_face is None:
        raise SystemExit("no face found in reference portrait")
    ref_bbox, ref_lm = ref_face
    log(f"Loaded {len(ref_imgs)} reference portraits; primary bbox={ref_bbox.tolist()}")

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

    # Identity is handled by VACE's own `reference_images` conditioning (below).
    # We deliberately do NOT wrap the transformer with a hand-rolled "face token"
    # injector: projecting a raw 128-d SFace embedding into the text stream is
    # out-of-distribution for a model trained only on T5 token embeddings, and
    # with real reference conditioning in place it added instability, not
    # identity. We also do NOT paste a landmark-warp face swap over the result —
    # that is what produced a second, misaligned face on top of the generated
    # one (the "overlapping face" artifact).

    # ---- Seed frame: primary portrait COVER-CROPPED to the output aspect ----
    # Cover-crop rather than letterbox, so frame 0 has no grey side bars that the
    # model would otherwise have to invent around. centering=0.42 keeps the head
    # (which sits above the middle of these portraits) in frame.
    from PIL import ImageOps

    seed = ImageOps.fit(ref_img, (W, H), method=Image.LANCZOS, centering=(0.5, 0.42))

    grey = Image.new("RGB", (W, H), (128, 128, 128))
    mask_black = Image.new("L", (W, H), 0)   # black = keep / condition on this frame
    mask_white = Image.new("L", (W, H), 255)  # white = generate this frame

    # Cap the reference count: each extra reference becomes one more latent step
    # in the attention sequence, and three views of one identity already pin it.
    refs = ref_imgs[:4]

    all_frames = []
    report = {"segments": [], "reference_portrait": a["portrait"],
              "reference_images": len(refs), "resolution": f"{W}x{H}",
              "steps": int(a.get("steps", 50)), "total_seconds": 0.0,
              "identity_sim": []}
    steps = int(a.get("steps", 50))
    n_frames = N

    # Attention backend. A T4 is sm_75, so FlashAttention is unavailable and
    # PyTorch's SDPA can fall back to the "math" kernel, which materialises an
    # explicit seq x seq attention matrix. At 832x480 that matrix -- not the
    # weights -- is what exhausts a 16 GB card, so ask for xformers first, then
    # the memory-efficient SDPA kernel, and only accept the math fallback last.
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log("attention: xformers memory-efficient enabled")
    except Exception as exc:  # noqa: BLE001 - xformers is optional
        log(f"attention: xformers unavailable ({type(exc).__name__}); using SDPA")
    math_sdp_off = False
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        math_sdp_off = True
        log("attention: math SDPA disabled (memory-efficient only)")
    except Exception as exc:  # noqa: BLE001 - never fail on a hint
        log(f"attention: could not restrict SDPA backends ({type(exc).__name__})")

    # Every segment regenerates from the reference portrait ALONE -- segment 0's
    # config, the only one proven excellent in all three runs (v9/v10/v11: start
    # 0.918-0.928, flat slope, framing ~112 IOD, zero face defects). Both
    # chaining orders were measured and rejected (D-9.17/D-9.18): with a tail in
    # the head, generation bridges the WHOLE conditioning window instead of
    # continuing from its last frame, so a degraded tail poisons every handoff --
    # v10 [portrait, tail] paid ~-0.04 identity + a luminance step per cut plus a
    # framing walk (IOD 112 -> 86, corr(identity, iod) = +0.874), and v11
    # [tail, portrait] collapsed harder (steps to -0.385, min 0.317 below the
    # 0.363 same-person threshold, 1 multi-face + 3 no-face frames). Dropping
    # the tail costs a pose reset at each cut (a jump-cut look) in exchange for
    # identity + framing re-pinned to the reference in every segment. The
    # conditioned portrait frame is dropped from the output.
    def _build(nf: int) -> tuple[list, list, int]:
        # Conditioning for one segment: portrait (black mask) + grey canvas.
        video = [seed, *[grey] * (nf - 1)]
        mask = [mask_black, *[mask_white] * (nf - 1)]
        return video, mask, 1

    for s in range(a["segments"]):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        # Each segment is an independent regeneration from the portrait (see
        # _build); only the noise seed differs (42 + s).
        out = None
        nf = n_frames
        floor = 9  # portrait + 8 frames worth generating
        while True:
            video, mask, keep = _build(nf)
            log(f"Segment {s}: generating {W}x{H}, {nf} frames, {steps} steps, "
                f"{len(refs)} reference images, keep={keep}")
            try:
                out = pipe(video=video, mask=mask, reference_images=refs,
                           prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                           num_frames=nf, num_inference_steps=steps, guidance_scale=5.0,
                           conditioning_scale=1.0, output_type="pil",
                           generator=torch.Generator("cpu").manual_seed(42 + s)).frames[0]
                break
            except Exception as exc:  # noqa: BLE001 - re-raised unless recoverable
                msg = str(exc).lower()
                if "out of memory" not in msg:
                    # Restricting the SDPA backends can surface as an unrelated
                    # error if no eligible kernel exists: re-enable the math
                    # kernel and retry this frame count once before giving up.
                    if math_sdp_off:
                        math_sdp_off = False
                        try:
                            torch.backends.cuda.enable_math_sdp(True)
                            log(f"Segment {s}: {type(exc).__name__} with math SDPA "
                                f"disabled -> re-enabled, retrying")
                            continue
                        except Exception:
                            pass
                    raise
                # Never let one segment abort the run: halve the frames and retry.
                # The floor IS attempted (the loop does not exit before it), and
                # once it OOMs too we stop rather than spin on the same size.
                gc.collect()
                torch.cuda.empty_cache()
                smaller = max(floor, ((nf // 2) | 1))
                if smaller >= nf:
                    raise RuntimeError(
                        f"Segment {s}: CUDA OOM at the floor of {nf} frames"
                    ) from exc
                nf = smaller
                log(f"Segment {s}: CUDA OOM -> retrying with {nf} frames")
        if out is None:
            raise RuntimeError(f"Segment {s}: generation failed at every frame count")
        n_frames = nf

        # The conditioned frame is the portrait itself, so drop it when
        # concatenating: it is an input, not new content.
        new_frames = out[keep:]
        all_frames += new_frames

        # Identity: a REAL cosine between two independently computed SFace
        # embeddings (D-9.11) — no swap, so this measures what the model itself
        # produced against the reference portrait. Measured in the GENERATED
        # region: `out` starts with `keep` conditioned frames, and indexing into
        # those would score the portrait against itself and report a fake 1.0.
        mid = len(new_frames) // 2
        frame_bgr = cv2.cvtColor(np.array(new_frames[mid]), cv2.COLOR_RGB2BGR)
        face = detect_face(frame_bgr)
        emb = swapper.compute_embedding(frame_bgr)
        sim = float(emb @ ref_emb) if emb is not None else -1.0
        report["identity_sim"].append(round(sim, 4))

        seg_seconds = len(new_frames) / 16.0
        report["total_seconds"] += seg_seconds
        report["segments"].append({
            "segment": s,
            "seconds": round(time.time() - t0, 1),
            "video_seconds": round(seg_seconds, 2),
            "frames": len(new_frames),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "face_found": face is not None,
            "identity_sim": round(sim, 4),
        })
        log(f"Segment {s}: face_found={face is not None} identity_sim={sim:.4f} "
            f"peak_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")
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
    # Every view of the same identity. VACE conditions on these natively, which
    # is what keeps the face consistent -- no post-hoc face swap (D-9.13).
    "references": reference_paths,
    "yunet": str(yunet_path),
    "sface": str(sface_path),
    "out": str(OUT_DIR),
    "segments": 5,
    # Wan's causal VAE wants num_frames == 4k+1 (49 = 4*12 + 1). Every segment
    # keeps only 1 conditioned frame (the portrait, dropped from output), so
    # 5 segments give 5 * 48 = 240 frames = 15.0 s at 16 fps.
    "frames_per_segment": 49,
    # 832x480 is VACE-1.3B's trained 480P resolution. The previous 512x288 was
    # well below it, which is a large part of why the output looked soft.
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
