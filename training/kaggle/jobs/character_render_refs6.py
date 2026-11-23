# Job: Render the character with SIX reference views (D-9.28) -- no adapter.
#
# D-9.30 (this revision): the user rejected S017_refs6 for blur and
# inconsistency, and the D-9.29 probe localised the cause in the CONDITIONING
# set itself -- the six references are 512x512 squares with mixed lighting, face
# scale and backgrounds, and diffusers' preprocess_conditions letterboxes each
# onto a WHITE canvas (176 px white bars at 480p). This revision therefore
# normalises the references in-step (tools/prepare_refs.py: output aspect, one
# face scale, one tone), uses the probe's strengthened prompt/negative, and
# exports at quality=10. Measured on the probe's same clip/seed (arm B):
#   sharpness 91 -> 441 encoded (728 pre-encode), identity 0.9227 -> 0.9256,
#   face-scale cv 0.039 -> 0.013, lighting sd 2.51 -> 1.72, 0 still frames.
# The low-denoise V2V "sharpen" pass was measured and REFUTED in the same
# probe (identity 0.92 -> 0.77 with mottled artifacts at strength 0.25), so
# there is no refine step here.
#
# This is the delivery render that replaces the adapter approach, and the
# evidence behind it is a 2x2 probe (character_render_lora_probe.py, same 21-frame
# / 50-step clip, seed 42 and 43):
#
#   refs  seed  id.mean  id.min  drift/frame  self-consistency  sharpness  motion
#     4    42   0.8543  0.7611   -0.00798     0.9553/0.8785      99.7      3.04
#     6    42   0.8964  0.8581   -0.00303     0.9724/0.9394      92.4      3.06
#     4    43   0.8988  0.8699   -0.00063     0.9748/0.9479     174.3      3.67
#     6    43   0.9145  0.8982   +0.00029     0.9762/0.9259     202.1      3.38
#
# The base model's identity decays INSIDE every segment (seg 0 0.9176 -> 0.8435
# over 48 frames, seg 3 down to 0.7870) because the reference-image conditioning
# weakens with the temporal distance from the conditioning frame. The vacecast
# dataset holds SIX C01 views of the same person and the render job used only
# FOUR: two views of the same identity were thrown away. With all six the decay
# halves (seed 42) or disappears (seed 43), the worst frame gains +0.10, and the
# motion and the sharpness do not suffer -- unlike every LoRA configuration,
# which buys identity by freezing the pose (D-9.27).
#
# `segment_seeds` names the noise seed of every segment explicitly. Each segment
# is regenerated from the portrait alone (D-9.17/D-9.18: no chaining), so a run
# can render a subset of the timeline and the segments can still be assembled.
# Splitting the render over several jobs keeps every job far from the 6 h
# timeout and delivers the first segments in half the time.
#
# Usage:
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_render_refs6.py
#       --scene data:vacecast --scene data:facemodels --wait
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
# peft wants >=0.16) -- this killed character_lora_v4's add_adapter on run 1, and loading
# an adapter injects modules through the same dispatcher. Uninstall it so the plain
# nn.Linear path is used.
u = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    capture_output=True,
    text=True,
)
log(f"torchao uninstall rc={u.returncode} {u.stdout.strip()[-200:]}")

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
# STEP 2b: Find the trained LoRA adapter (mounted with --scene data:lorav5)
# ================================================================
# D-9.24: character_lora_v4 writes out/lora_adapter_best/pytorch_lora_weights.safetensors.
# Loading it needs weight_name (the loader defaults to .bin) and prefix=None (the default
# prefix "transformer" silently filters out every saved key) -- both found by dry run.
#
# D-9.28: NO adapter. Every LoRA configuration was measured and rejected: the
# adapter's identity gain and its pose-freezing are the same effect, so it
# cannot give a consistent face without turning the video into a still image
# (D-9.27). The identity comes from the reference conditioning instead.
log("No adapter: the base model with six reference views (D-9.28)")

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
    import gc, json, shutil, sys, time, torch, cv2, numpy as np
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
    #
    # D-9.30: normalise the references first (tools/prepare_refs.py) -- the arm
    # the D-9.29 probe measured at 441 vs 91 mean sharpness (encoded), identity
    # flat at 0.92, face-scale cv 3x tighter and lighting sd down. The raw
    # references are 512x512 squares with mixed lighting, face scale and
    # backgrounds, and diffusers' preprocess_conditions letterboxes each one
    # onto a WHITE canvas, so the model saw 480x480 images with 176 px white
    # bars on each side. Normalised: output aspect (no bars), one face scale,
    # one tone. The ORIGINAL primary portrait stays the metric reference below,
    # so every in-job identity number stays comparable with earlier runs.
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
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            log(f"skipping unreadable reference {rp}: {exc}")
    if not ref_imgs:
        raise SystemExit("no readable reference portraits")
    log("normalised references: %s" % [Path(p).name for p in norm_paths])
    # Keep the exact conditioning with the artefacts, FLAT in the output
    # directory: the archive and the recovery sweep both pick up top-level
    # files, and the normalised references are how the next defect gets found.
    for p in norm_paths:
        shutil.copy2(p, Path(a["out"]) / ("ref_" + Path(p).name))
    ref_img = ref_imgs[0]

    metric_img = Image.open(a["portrait"]).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(metric_img), cv2.COLOR_RGB2BGR)
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

    # D-9.28: no adapter at all -- the base model. Every LoRA configuration was
    # measured and rejected (D-9.27): the identity gain and the pose-freezing
    # are the same effect, so an adapter can only trade movement for identity.
    log("Base model (no adapter): the identity comes from the reference views")

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

    # D-9.28: use ALL the C01 views of the same person. The render job used four
    # while the dataset holds six, and the missing two were what let the face
    # drift inside a segment (the 2x2 probe: drift -0.00798/frame with four
    # references, -0.00303 with six on the same seed, +0.00029 on the good
    # seed). Each extra reference is one more latent step in the attention
    # sequence: the render costs ~27% more than with four, and the probe peaked
    # at 7.53 GB with four, so six fits comfortably in the T4's 15.4 GB.
    refs = ref_imgs[: int(a.get("num_references", 6))]
    log("Using %d reference views (of %d found): %s"
        % (len(refs), len(ref_imgs), [Path(p).name for p in ref_paths[: len(refs)]]))

    # The seed of every segment, named explicitly: each segment is an
    # independent regeneration from the portrait, so a run can render any
    # subset of the timeline (D-9.28). Only the first `segments` seeds are
    # used, and the output name is derived from exactly those seeds.
    segment_seeds = list(a.get("segment_seeds") or [42 + s for s in range(int(a["segments"]))])
    segment_seeds = segment_seeds[: int(a["segments"])]
    if len(segment_seeds) < int(a["segments"]):
        raise SystemExit(
            "segment_seeds has %d entries but %d segments were requested"
            % (len(segment_seeds), int(a["segments"]))
        )
    log("Segment seeds: %s" % (segment_seeds,))

    all_frames = []
    report = {"segments": [], "reference_portrait": a["portrait"],
              "reference_images": len(refs), "resolution": f"{W}x{H}",
              "steps": int(a.get("steps", 50)), "total_seconds": 0.0,
              "segment_seeds": list(segment_seeds),
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

    for s in range(int(a["segments"])):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        seed_value = int(segment_seeds[s])

        # Each segment is an independent regeneration from the portrait (see
        # _build); only the noise seed differs.
        out = None
        nf = n_frames
        floor = 9  # portrait + 8 frames worth generating
        while True:
            video, mask, keep = _build(nf)
            log(f"Segment {s} (seed {seed_value}): generating {W}x{H}, {nf} frames, {steps} steps, "
                f"{len(refs)} reference images, keep={keep}")
            try:
                out = pipe(video=video, mask=mask, reference_images=refs,
                           prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                           num_frames=nf, num_inference_steps=steps, guidance_scale=5.0,
                           conditioning_scale=1.0, output_type="pil",
                           generator=torch.Generator("cpu").manual_seed(seed_value)).frames[0]
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
        #
        # EVERY frame of the segment is measured, because the defect this job
        # fixes is the drift WITHIN a segment (D-9.28): the identity starts at
        # ~0.92 and decays towards the end. A single mid-frame score hides it.
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

        # Self-consistency: every frame against the segment's own median face.
        self_min = -1.0
        if seg_embs:
            matrix = np.stack(seg_embs)
            median = np.median(matrix, axis=0)
            norm = float(np.linalg.norm(median))
            if norm > 0:
                self_min = float(np.min(matrix @ (median / norm)))
        drift = (
            (seg_sims[-1] - seg_sims[0]) / max(1, len(seg_sims) - 1) if len(seg_sims) > 1 else 0.0
        )
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
        })
        log("Segment %d (seed %d): faces %d/%d identity %.4f (min %.4f, drift %+.5f/frame) "
            "self_min %.4f sharpness %.1f peak_gb %.2f"
            % (s, seed_value, faces_found, len(new_frames), seg_identity,
               float(np.min(seg_sims)) if seg_sims else -1.0, drift, self_min,
               float(np.mean(seg_sharps)) if seg_sharps else -1.0,
               torch.cuda.max_memory_allocated() / 1e9))
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Export video ----
    out_dir = Path(a["out"])
    seed_tag = "%d-%d" % (segment_seeds[0], segment_seeds[-1]) if segment_seeds else "none"
    stem = "refs%d_segs%s" % (len(refs), seed_tag)
    out_mp4 = str(out_dir / ("%s.mp4" % stem))
    # D-9.30: quality=10. The default (quality=5) cost the probe's arm B 40% of
    # its sharpness through the FIRST encode alone (728 pre-encode -> 441 in the
    # mp4), and the chain re-encodes again downstream.
    export_to_video(all_frames, out_mp4, fps=16, quality=10)
    log(f"Video exported: {out_mp4}")

    # ---- Motion: the number that exposed the v4 "shaking image" defect -------
    # Mean frame-to-frame difference and the share of frames that barely change.
    # The base model scores ~2.5 / 0%; the v4 adapter scored 0.41 / 88%.
    #
    # `all_frames` holds PIL images (the pipeline returns PIL frames), so they
    # have to be converted to arrays first -- calling .astype() on a PIL Image
    # raised AttributeError here and cost the first run its JSON report (the
    # video itself was already exported and was fine).
    diffs = []
    try:
        for i in range(1, len(all_frames)):
            prev = np.asarray(all_frames[i - 1], dtype=np.float32)
            cur = np.asarray(all_frames[i], dtype=np.float32)
            diffs.append(float(np.abs(prev - cur).mean()))
    except Exception as exc:  # noqa: BLE001 - a stats failure must not lose the report
        log(f"motion measurement failed: {type(exc).__name__}: {exc}")
    motion_mean = float(np.mean(diffs)) if diffs else 0.0
    still_frac = float(np.mean([1.0 if d < 0.5 else 0.0 for d in diffs])) if diffs else 0.0
    report["motion_mean"] = round(motion_mean, 4)
    report["motion_median"] = round(float(np.median(diffs)), 4) if diffs else 0.0
    report["motion_max"] = round(float(np.max(diffs)), 4) if diffs else 0.0
    report["still_fraction"] = round(still_frac, 4)
    report["motion_pairs"] = len(diffs)
    report["num_references"] = len(refs)
    report["reference_names"] = [Path(p).name for p in ref_paths[: len(refs)]]
    log("MOTION mean=%.3f median=%.3f max=%.3f frames<0.5: %d/%d"
        % (motion_mean, report["motion_median"], report["motion_max"],
           int(round(still_frac * len(diffs))), len(diffs)))

    with open(str(out_dir / ("%s.json" % stem)), "w") as fh:
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
    # Every view of the same identity. VACE conditions on these natively, which
    # is what keeps the face consistent -- no post-hoc face swap (D-9.13).
    "references": reference_paths,
    "yunet": str(yunet_path),
    "sface": str(sface_path),
    "out": str(OUT_DIR),
    # D-9.28: SIX views of the same person, not four. The reference-image
    # conditioning is what holds the face together inside a segment, and the
    # 2x2 probe measured the difference on the same clip and seed.
    "num_references": 6,
    # The noise seed of every segment, named explicitly. The segments are
    # independent regenerations from the portrait, so a run can render any
    # subset of the timeline. Seed 43 is the one the probe measured as the
    # good draw (6 refs: identity 0.9145, no decay); the rest are fresh draws.
    #
    # The render is split over two jobs (D-9.28): the first half (43, 44, 45)
    # is fetched and measured before the second half is pushed, because Kaggle
    # keeps only the newest kernel version's outputs. D-9.30 re-renders the same
    # five segments with the normalised references + strengthened prompt; one job
    # cannot hold 5 segments inside the 6 h timeout (5 x ~68 min + T5 download).
    # Second half: seeds 46 and 47 (the first half, 43/44/45, is fetched and
    # kept in runs/analysis/refsnorm_half1).
    "segment_seeds": [46, 47],
    "segments": 2,
    # Wan's causal VAE wants num_frames == 4k+1 (49 = 4*12 + 1). Every segment
    # keeps only 1 conditioned frame (the portrait, dropped from output), so
    # 5 segments give 5 * 48 = 240 frames = 15.0 s at 16 fps.
    "frames_per_segment": 49,
    # 832x480 is VACE-1.3B's trained 480P resolution. The previous 512x288 was
    # well below it, which is a large part of why the output looked soft.
    "width": 832,
    "height": 480,
    "steps": 50,
    # D-9.30: the strengthened prompt the D-9.29 probe measured WITH the
    # normalised references (arm B). The flicker/morphing terms are the
    # community's standard guard against texture shimmer and face morphing; the
    # camera-lock and framing terms attack the head-scale walk the contact
    # sheets exposed.
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
    # Reference tone-match strength (tools/prepare_refs.py); 1.0 is the value
    # the probe ran.
    "tone": 1.0,
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

log_path = OUT_DIR / "refs6_render.log"
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

# The output name is derived from exactly the seeds the step script used: the
# first `segments` entries of segment_seeds.
used_seeds = list(args["segment_seeds"])[: int(args["segments"])]
seed_tag = f"{used_seeds[0]}-{used_seeds[-1]}"
stem = f"refs{args['num_references']}_segs{seed_tag}"
expected = [f"{stem}.mp4", f"{stem}.json", "refs6_render.log"]
present = {name: (OUT_DIR / name) for name in expected}
for name, p in present.items():
    size = p.stat().st_size if p.exists() else -1
    log(f"artefact {name}: {'MISSING' if size < 0 else f'{size} bytes'}")

if not present[f"{stem}.mp4"].exists():
    log(f"FATAL: {stem}.mp4 was not produced inside the archived output directory")
    raise SystemExit(2)

log("done")
