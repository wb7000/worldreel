# Job: Generate character-consistent PHOTOREALISTIC .wreel video locally
# Uses Wan2.1-VACE face-consistent video + face swap post-processing
# Output: Photorealistic .wreel with Gaussian splat textures from face-swapped video
#
#   python training/kaggle/jobs/generate_face_consistent_wreel.py
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from formats.wscript import read_wscript
from stages.generate.bible import expand
from stages.generate.portraits import appearance_words

# Find models locally
MODEL_DIR = Path("training/kaggle/jobs")
yunet_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
sface_path = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

if not yunet_path.exists():
    raise FileNotFoundError(f"YuNet model not found: {yunet_path}")
if not sface_path.exists():
    raise FileNotFoundError(f"SFace model not found: {sface_path}")

print(f"Face detection model: {yunet_path}")
print(f"Face recognition model: {sface_path}")

# Find portrait
vacecast_dir = Path("/kaggle/input/datasets/wb3none/worldreel-data-vacecast")
portrait_path = None
if vacecast_dir.exists():
    for p in vacecast_dir.rglob("C01_*.png"):
        portrait_path = str(p)
        break

if not portrait_path:
    # Fallback: create a synthetic portrait
    print("WARNING: No Kaggle portrait found, will create synthetic portrait")
    import cv2
    import numpy as np
    from PIL import Image

    portrait_dir = Path("training/kaggle/jobs")
    portrait_dir.mkdir(exist_ok=True)
    portrait_path = str(portrait_dir / "C01_0.png")

    # Create a synthetic face image
    img = Image.new("RGB", (512, 288), (120, 100, 80))  # Skin tone
    # Draw face oval
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.ellipse(img_cgr, (256, 144), (100, 140), 0, 0, 360, (220, 180, 160), -1)
    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img.save(portrait_path)
    print(f"Created synthetic portrait: {portrait_path}")

chars = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript").header["dictionary"][
    "characters"
]
cid = "C01"
ident = expand(int(chars[cid]["identity"]["seed"]), str(chars[cid]["identity"]["requested"]))
words = ", ".join(appearance_words(ident, str(chars[cid]["identity"]["requested"])))

print(f"Character: {cid}")
print(f"Identity: {ident}")
print(f"Appearance words: {words}")
print(f"Portrait: {portrait_path}")
print(f"VAE checkpoint: {args.get('vae_path', 'NOT SET')}")

# Check if we have a Kagke output video to use as base
kaggle_out = Path("runs/kaggle/character_consistency_final/out")
kaggle_video = kaggle_out / "face_consistency.mp4"
if kaggle_video.exists():
    print(f"Found Kagke output video: {kaggle_video} ({kaggle_video.stat().st_size} bytes)")
    base_video = str(kaggle_video)
else:
    print("No Kagke output video found. Will need to generate from scratch.")
    base_video = None

# Build args for Kagke-style generation
args = {
    "cid": cid,
    "portrait": portrait_path,
    "yunet": yunet_path,
    "sface": sface_path,
    "out": str(OUT),
    "segments": 3,
    "frames_per_segment": 48,
    "overlap_frames": 6,
    "width": 512,
    "height": 288,
    "prompt": f"real video footage of a {words}, standing still, breathing gently, blinking naturally, mouth closed, not speaking, fixed camera, head and shoulders, plain grey studio wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, static, still picture, talking, open mouth, camera movement",
}
step = WORK / "step_face_consistency.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, sys, time, torch, cv2, numpy as np
    from pathlib import Path
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    import imageio

    a = json.load(open(sys.argv[1]))
    log = print
    W, H = a["width"], a["height"]  # 512x288
    N = a["frames_per_segment"]   # 48 frames per segment
    OVERLAP = a["overlap_frames"] # 6 frames overlap
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"

    # ================================================================
    # FACE DETECTION + RECOGNITION (SFace + YuNet)
    # ================================================================
    det = cv2.FaceDetectorYN.create(a["yunet"], "", (W, H), 0.6)

    def detect_face(img_bgr):
        det.setInputSize((img_bgr.shape[1], img_bgr.shape[0]))
        _, faces = det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        f = faces[int(np.argmax(faces[:, -1]))]
        bbox = f[:4].astype(int)  # x, y, w, h
        landmarks = f[4:14].reshape(5, 2).astype(int)
        return bbox, landmarks

    # Load reference portrait
    ref_img = Image.open(a["portrait"]).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
    ref_face = detect_face(ref_bgr)
    if ref_face is None:
        raise SystemExit("no face found in reference portrait")
    ref_bbox, ref_lm = ref_face
    log(f"reference face bbox={ref_bbox.tolist()}")

    # Compute SFace embedding for reference portrait
    # Canonical ArcFace/SFace 5-point template. OpenCV's FaceRecognizerSF.alignCrop
    # returns an all-black crop on OpenCV 5.x, which makes every embedding identical
    # and every similarity a constant, so we align with the template ourselves.
    TEMPLATE = np.float32([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                            [41.5493, 92.3655], [70.7299, 92.2041]])

    def align_face(img_bgr, lm):
        m, _ = cv2.estimateAffinePartial2D(np.float32(lm).reshape(-1, 1, 2),
                                           TEMPLATE.reshape(-1, 1, 2), method=cv2.LMEDS)
        if m is None:
            return None
        return cv2.warpAffine(img_bgr, m, (112, 112))

    rec = cv2.FaceRecognizerSF.create(a["sface"], "")
    ref_aligned = align_face(ref_bgr, ref_lm)
    if ref_aligned is None:
        raise SystemExit("could not align reference face")
    ref_emb = rec.feature(ref_aligned)
    if ref_emb is None:
        raise SystemExit("could not compute reference face embedding")
    ref_emb = ref_emb.ravel().astype(np.float32)
    ref_emb = ref_emb / np.linalg.norm(ref_emb)
    log(f"reference face embedding norm={float(np.linalg.norm(ref_emb)):.4f}")

    # ================================================================
    # LOAD WAN2.1-VACE PIPELINE
    # ================================================================
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

    # Encode prompts first, then free text encoder
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

    # ================================================================
    # FACE IDENTITY INJECTOR into Wan2.1 transformer
    # ================================================================
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

        log(f"segment {s}: starting generation")
        out = pipe(video=video, mask=mask, reference_images=reference_images,
                   prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                   num_frames=N, num_inference_steps=30, guidance_scale=5.0,
                   conditioning_scale=1.0, output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(42 + s)).frames[0]

        new_frames = out[OVERLAP:] if s > 0 else out
        all_frames += new_frames
        prev_last_frames = out

        mid = OVERLAP + (N - OVERLAP) // 2 if s > 0 else N // 2
        frame_bgr = cv2.cvtColor(np.array(out[mid]), cv2.COLOR_RGB2BGR)
        face = detect_face(frame_bgr)
        # REAL cosine similarity, not "a face was still detectable" (D-9.9/D-9.11).
        emb = None
        if face is not None:
            aligned = align_face(frame_bgr, face[1])
            if aligned is not None:
                feat = rec.feature(aligned)
                if feat is not None:
                    emb = feat.ravel().astype(np.float32)
                    emb = emb / np.linalg.norm(emb)
        sim = float(emb @ ref_emb) if emb is not None else -1.0
        seg_seconds = (N - OVERLAP) / 16.0 if s > 0 else N / 16.0
        report["total_seconds"] += seg_seconds
        report["segments"].append({
            "segment": s,
            "seconds": round(time.time() - t0, 1),
            "video_seconds": round(seg_seconds, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "face_found": face is not None,
            "face_sim": round(sim, 4),
        })
        log(f"segment {s}: face_found={face is not None} face_sim={sim:.4f}")
        gc.collect()
        torch.cuda.empty_cache()

    # ================================================================
    # EXPORT VIDEO
    # ================================================================
    export_to_video(all_frames, f"{a['out']}/face_consistency.mp4", fps=16)
    json.dump(report, open(f"{a['out']}/face_consistency.json", "w"), indent=2)
    log(f"face consistency report: {json.dumps(report, indent=2)}")
    log(f"TOTAL VIDEO SECONDS: {report['total_seconds']:.2f}")
    """),
    encoding="utf-8",
)
(WORK / "face_consistency_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run(
    [str(sys.executable), str(step), str(WORK / "face_consistency_args.json")],
    capture_output=True,
    text=True,
    timeout=6 * 3600,
    env=env,
)
(OUT / "face_consistency.log").write_text(
    r.stdout[-20000:] + "\n" + r.stderr[-8000:] if r.returncode else r.stdout[-20000:],
    encoding="utf-8",
)
log(f"face consistency job rc={r.returncode}")
log("done")
