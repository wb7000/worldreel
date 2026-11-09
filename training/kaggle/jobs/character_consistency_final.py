# Job: Generate character-consistent .wreel video using face identity injection
# into the Wan2.1-VACE diffusion transformer (IP-Adapter-FaceID style).
#
# Key change vs previous version: face embedding (SFace) is injected as
# additional cross-attention context tokens into the Wan2.1 transformer
# during EVERY denoising step, not just into the VAE decoder. This forces
# the diffusion model to condition on the reference identity throughout
# generation, matching how Stand-In / InstantID / PuLID achieve face consistency.
#
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_consistency_final.py --scene data:vacecast --scene data:facemodels --wait
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from formats.wscript import read_wscript
from stages.generate.bible import expand
from stages.generate.portraits import appearance_words

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
py = Path("/tmp/faceid/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/faceid"], check=True)
r = subprocess.run(
    [*UV, "pip", "install", "--python", str(py), "diffusers", "transformers",
     "accelerate", "safetensors", "torch", "ftfy", "imageio", "imageio-ffmpeg",
     "sentencepiece", "opencv-python-headless"],
    capture_output=True, text=True)
log(f"face consistency env install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

# Find models in mounted Kaggle datasets
import glob as _glob
INPUT = Path("/kaggle/input")
MODEL_DIR = Path("/kaggle/input/datasets/wb3none/worldreel-data-facemodels")
yunet_path = None
sface_path = None
for p in MODEL_DIR.rglob("face_detection_yunet_2023mar.onnx"):
    yunet_path = str(p)
for p in MODEL_DIR.rglob("face_recognition_sface_2021dec.onnx"):
    sface_path = str(p)

# Find portrait in mounted datasets
vacecast_dir = Path("/kaggle/input/datasets/wb3none/worldreel-data-vacecast")
portrait_path = None
for p in vacecast_dir.rglob("C01_*.png"):
    portrait_path = str(p)
    break

chars = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript").header["dictionary"][
    "characters"
]
cid = "C01"
ident = expand(int(chars[cid]["identity"]["seed"]), str(chars[cid]["identity"]["requested"]))
words = ", ".join(appearance_words(ident, str(chars[cid]["identity"]["requested"])))

args = {
    "cid": cid,
    "portrait": portrait_path,
    "yunet": yunet_path,
    "sface": sface_path,
    "out": str(OUT),
    "segments": 5,  # 5 segments x 1 second = 5 seconds total
    "frames_per_segment": 16,  # 1s at 16 fps
    "overlap_frames": 10,  # keep last 10 frames as reference for chaining
    "prompt": f"real video footage of a {words}, standing still, breathing gently, blinking "
    "naturally, mouth closed, not speaking, fixed camera, head and shoulders, plain grey studio "
    "wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, "
    "shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
    "blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, "
    "static, still picture, talking, open mouth, camera movement",
}
step = WORK / "step_face_identity.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, sys, time, torch, cv2, numpy as np
    from pathlib import Path
    from PIL import Image
    import torch.nn as nn
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    a = json.load(open(sys.argv[1]))
    log = print
    W, H = 512, 288
    N = a["frames_per_segment"]
    OVERLAP = a["overlap_frames"]
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"

    # --- Face detection + recognition (SFace) ---
    det = cv2.FaceDetectorYN.create(a["yunet"], "", (W, H), 0.6)
    rec = cv2.FaceRecognizerSF.create(a["sface"], "")

    def embed(img_bgr):
        det.setInputSize((img_bgr.shape[1], img_bgr.shape[0]))
        _, faces = det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        f = faces[int(np.argmax(faces[:, -1]))]
        feat = rec.alignCrop(img_bgr, f)
        feat = rec.feature(feat).ravel()
        return feat / np.linalg.norm(feat)

    ref = Image.open(a["portrait"]).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2BGR)
    ref_emb = embed(ref_bgr)
    if ref_emb is None:
        raise SystemExit("no face found in reference portrait")
    log(f"reference face embedding norm: {float(np.linalg.norm(ref_emb)):.4f}")

    ref_emb_np = ref_emb.astype(np.float32)

    # --- Load Wan2.1-VACE pipeline ---
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

    # Encode prompts first, then free text encoder (saves ~10GB VRAM)
    pipe.text_encoder.to("cuda")
    with torch.no_grad():
        pos, neg = pipe.encode_prompt(prompt=a["prompt"], negative_prompt=a["negative"],
                                      do_classifier_free_guidance=True, device=torch.device("cuda"))
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()

    # --- CPU offload before wrapping transformer ---
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    # --- Face identity injector (IP-Adapter-FaceID style) ---
    # Injects SFace embedding as extra cross-attention context tokens into
    # the Wan2.1 transformer during every denoising step.
    class FaceConditionedTransformer(nn.Module):
        def __init__(self, base_transformer, face_embedding_dim=512):
            super().__init__()
            self.base = base_transformer
            self.face_embedding_dim = face_embedding_dim
            hidden_dim = (getattr(base_transformer.config, "hidden_size", None)
                          or getattr(base_transformer.config, "inner_dim", None)
                          or 1536)
            self.hidden_dim = hidden_dim
            self.face_projector = nn.Linear(face_embedding_dim, hidden_dim)
            # Identity-mapping init: face embedding lands in first 512 hidden dims
            self.face_projector.weight.data.zero_()
            if face_embedding_dim <= hidden_dim:
                self.face_projector.weight.data[:face_embedding_dim, :face_embedding_dim] = torch.eye(face_embedding_dim)
            self.face_projector.bias.data.zero_()
            self.scale = 10.0  # scale face tokens to match text embedding magnitude

        @property
        def config(self):
            return self.base.config

        @property
        def dtype(self):
            return self.base.dtype

        @property
        def device(self):
            return self.base.device

        def __getattr__(self, name):
            # Delegate any attribute not found on wrapper to the base transformer
            # (e.g. cache_context, forward hooks, etc. used by VACE pipeline).
            # nn.Module stores Module values in _modules, not __dict__, so check both.
            base = self.__dict__.get("base")
            if base is None and "_modules" in self.__dict__:
                base = self.__dict__["_modules"].get("base")
            if base is not None:
                return getattr(base, name)
            raise AttributeError(name)

        def forward(self, hidden_states=None, encoder_hidden_states=None,
                    timestep=None, **kwargs):
            if encoder_hidden_states is not None:
                device = hidden_states.device
                dtype = hidden_states.dtype
                # Load face embedding lazily from file (CPU offload moves models around)
                ref_emb_t = torch.from_numpy(ref_emb_np).unsqueeze(0).to(device, dtype)
                # Ensure projector is on same device
                self.face_projector = self.face_projector.to(device, dtype)
                face_tokens = self.face_projector(ref_emb_t)  # [B, hidden_dim]
                face_tokens = face_tokens * self.scale
                face_tokens = face_tokens.unsqueeze(1)  # [B, 1, hidden_dim]
                encoder_hidden_states = torch.cat([face_tokens, encoder_hidden_states], dim=1)
            return self.base(hidden_states=hidden_states,
                             encoder_hidden_states=encoder_hidden_states,
                             timestep=timestep, **kwargs)

    # Wrap the transformer AFTER CPU offload is set up
    pipe.transformer = FaceConditionedTransformer(pipe.transformer)

    x = (W - H) // 2
    first = Image.new("RGB", (W, H), (128, 128, 128))
    first.paste(ref.resize((H, H), Image.LANCZOS), (x, 0))

    all_frames = []
    report = {"segments": [], "reference_portrait": a["portrait"],
              "total_seconds": 0.0}
    prev_last_frames = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)

    for s in range(a["segments"]):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        if s == 0:
            # First segment: portrait as reference image
            video = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)
            mask = [Image.new("L", (W, H), 0)] + [Image.new("L", (W, H), 255)] * (N - 1)
            reference_images = [ref]
        else:
            # Chain: keep last OVERLAP frames as reference + regenerate rest
            video = [*prev_last_frames[-OVERLAP:],
                     *[Image.new("RGB", (W, H), (128, 128, 128))] * (N - OVERLAP)]
            mask = [Image.new("L", (W, H), 0)] * OVERLAP + \
                   [Image.new("L", (W, H), 255)] * (N - OVERLAP)
            reference_images = [ref, prev_last_frames[-1]]

        out = pipe(video=video, mask=mask, reference_images=reference_images,
                   prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                   num_frames=N, num_inference_steps=30, guidance_scale=5.0,
                   conditioning_scale=1.0, output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(42 + s)).frames[0]

        new_frames = out[OVERLAP:] if s > 0 else out
        all_frames += new_frames
        prev_last_frames = out

        # Measure identity on middle frame of new content
        mid = OVERLAP + (N - OVERLAP) // 2 if s > 0 else N // 2
        frame_bgr = cv2.cvtColor(np.array(out[mid]), cv2.COLOR_RGB2BGR)
        emb = embed(frame_bgr)
        sim = float(emb @ ref_emb) if emb is not None else -1.0
        seg_seconds = (N - OVERLAP) / 16.0 if s > 0 else N / 16.0
        report["total_seconds"] += seg_seconds
        report["segments"].append({
            "segment": s,
            "seconds": round(time.time() - t0, 1),
            "video_seconds": round(seg_seconds, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "sface_vs_ref": round(sim, 4),
            "face_found": emb is not None,
            "passes_same_person": bool(emb is not None and sim >= 0.363),
        })
        log(f"segment {s}: SFace={sim:.4f} face_found={emb is not None} "
            f"passes_same_person={report['segments'][-1]['passes_same_person']}")
        gc.collect()
        torch.cuda.empty_cache()

    export_to_video(all_frames, f"{a['out']}/face_consistency.mp4", fps=16)
    json.dump(report, open(f"{a['out']}/face_consistency.json", "w"), indent=2)
    log(f"face consistency report: {json.dumps(report, indent=2)}")
    all_pass = all(s["passes_same_person"] for s in report["segments"])
    log(f"ALL SEGMENTS PASS SAME-PERSON MARK (0.363): {all_pass}")
    log(f"TOTAL VIDEO SECONDS: {report['total_seconds']:.2f}")
    """),
    encoding="utf-8",
)
(WORK / "face_identity_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "face_identity_args.json")],
                   capture_output=True, text=True, timeout=6 * 3600, env=env)
(OUT / "face_consistency.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-8000:] if r.returncode else r.stdout[-20000:], encoding="utf-8")
log(f"face consistency job rc={r.returncode}")
log("done")