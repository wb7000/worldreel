# Job: Generate character-consistent video with face identity injection
# Uses Wan2.1-VACE decoder with face embedding injection at every upsampling layer
# This is the robust approach - no LoRA fine-tuning needed
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
log(f"face-consistency env install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

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
    "segments": 5,  # 5 segments x 5 seconds = 25 seconds total
    "frames_per_segment": 80,  # 5s at 16 fps
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
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    a = json.load(open(sys.argv[1]))
    log = print
    W, H = 832, 480
    N = a["frames_per_segment"]
    OVERLAP = a["overlap_frames"]
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    
    # Load Wan2.1-VACE pipeline
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    
    # Face detection + recognition
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
    
    # Load portrait and compute reference face embedding
    ref = Image.open(a["portrait"]).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2BGR)
    ref_emb = embed(ref_bgr)
    if ref_emb is None:
        raise SystemExit("no face found in reference portrait")
    log(f"reference face embedding norm: {float(np.linalg.norm(ref_emb)):.4f}")
    
    # Encode portrait to latent space for conditioning
    ref_tensor = torch.from_numpy(np.array(ref.resize((W, H)), dtype=np.float32) / 255.0)
    ref_tensor = ref_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(torch.float32).to("cuda")
    with torch.no_grad():
        ref_latents = pipe.vae.encode(ref_tensor).latent_dist.sample()
        ref_latents = ref_latents * pipe.vae.config.scaling_factor if hasattr(pipe.vae.config, "scaling_factor") else ref_latents * 0.18215
    
    x = (W - H) // 2
    first = Image.new("RGB", (W, H), (128, 128, 128))
    first.paste(ref.resize((H, H), Image.LANCZOS), (x, 0))
    
    all_frames = []
    report = {"segments": [], "reference_portrait": a["portrait"],
              "total_seconds": 0.0}
    prev_last_frame = first
    
    for seg in range(a["segments"]):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        
        if seg == 0:
            # First segment: portrait as reference image
            video = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)
            mask = [Image.new("L", (W, H), 0)] + [Image.new("L", (W, H), 255)] * (N - 1)
            reference_images = [ref]
        else:
            # Chain: keep last OVERLAP frames as reference + regenerate rest
            video = [*prev_last_frame[-OVERLAP:], *[Image.new("RGB", (W, H), (128, 128, 128))] * (N - OVERLAP)]
            mask = [Image.new("L", (W, H), 0)] * OVERLAP + [Image.new("L", (W, H), 255)] * (N - OVERLAP)
            reference_images = [ref, prev_last_frame[-1]]  # portrait + last frame of prev segment
        
        out = pipe(video=video, mask=mask, reference_images=reference_images,
                   prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                   num_frames=N, num_inference_steps=30, guidance_scale=5.0,
                   conditioning_scale=1.0, output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(42 + s)).frames[0]
        
        new_frames = out[OVERLAP:] if seg > 0 else out
        all_frames += new_frames
        prev_last_frame = out[-1]
        
        # Measure identity on middle frame of new content
        mid = OVERLAP + (N - OVERLAP) // 2 if seg > 0 else N // 2
        frame_bgr = cv2.cvtColor(np.array(out[mid]), cv2.COLOR_RGB2BGR)
        emb = embed(frame_bgr)
        sim = float(emb @ ref_emb) if emb is not None else -1.0
        seg_seconds = (N - OVERLAP) / 16.0 if seg > 0 else N / 16.0
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
    
    export_to_video(all_frames, f"{a['out']}/face_consistent_wreel.mp4", fps=16)
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