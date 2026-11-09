# Job: Per-character LoRA fine-tuning for Wan2.1-VACE
# Fine-tunes the transformer on each character's portrait so identity is locked
# across all generated shots, not just conditioned.
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_lora_v3.py --scene data:vacecast --scene data:facemodels --wait
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
py = Path("/tmp/lora3/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/lora3"], check=True)
r = subprocess.run(
    [*UV, "pip", "install", "--python", str(py), "diffusers", "transformers",
     "accelerate", "safetensors", "torch", "ftfy", "imageio", "imageio-ffmpeg",
     "sentencepiece", "opencv-python-headless", "peft"],
    capture_output=True, text=True)
log(f"lora v3 env install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

# Find models in mounted Kaggle datasets
import glob as _glob
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

args = {
    "cid": "C01",
    "portrait": portrait_path,
    "yunet": yunet_path,
    "sface": sface_path,
    "out": str(OUT),
    "prompt": "real video footage of a woman standing still, breathing gently, blinking naturally, mouth closed, not speaking, fixed camera, head and shoulders, plain grey studio wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, static, still picture, talking, open mouth, camera movement",
    "lora_rank": 16,
    "lora_alpha": 32,
    "train_steps": 100,
    "lr": 1e-4,
    "test_segments": 3,
    "frames_per_segment": 33,
}
step = WORK / "step_lora_v3.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, sys, time, torch, cv2, numpy as np
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    import peft

    a = json.load(open(sys.argv[1]))
    log = print
    W, H = 832, 480
    N = a["frames_per_segment"]

    yunet_path = a["yunet"]
    sface_path = a["sface"]
    portrait_path = a["portrait"]

    det = cv2.FaceDetectorYN.create(yunet_path, "", (W, H), 0.6)
    rec = cv2.FaceRecognizerSF.create(sface_path, "")

    def embed(img_bgr):
        det.setInputSize((img_bgr.shape[1], img_bgr.shape[0]))
        _, faces = det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        f = faces[int(np.argmax(faces[:, -1]))]
        feat = rec.alignCrop(img_bgr, f)
        feat = rec.feature(feat).ravel()
        return feat / np.linalg.norm(feat)

    ref = Image.open(portrait_path).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2BGR)
    ref_emb = embed(ref_bgr)
    if ref_emb is None:
        raise SystemExit("no face found in reference portrait")
    log(f"reference face embedding norm: {float(np.linalg.norm(ref_emb)):.4f}")

    # ============ PHASE 1: LoRA Fine-tuning on Reference Portrait ============
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=a["lora_rank"],
        lora_alpha=a["lora_alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    trainable_params = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pipe.transformer.parameters())
    log(f"LoRA trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    pipe.transformer.train()
    optimizer = torch.optim.AdamW(pipe.transformer.parameters(), lr=a["lr"])

    ref_tensor = torch.from_numpy(np.array(ref.resize((W, H)), dtype=np.float32) / 255.0)
    ref_tensor = ref_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(torch.float32).to("cuda")

    log(f"Starting LoRA training ({a['train_steps']} steps)...")
    for step_idx in range(a["train_steps"]):
        optimizer.zero_grad()
        with torch.no_grad():
            ref_latents = pipe.vae.encode(ref_tensor).latent_dist.sample()
            vae_config = pipe.vae.config
            try:
                scaling_factor = float(vae_config.get("scaling_factor", 0.18215))
            except (AttributeError, TypeError):
                try:
                    scaling_factor = float(vae_config.scaling_factor)
                except (AttributeError, TypeError):
                    scaling_factor = 0.18215  # Default VAE scaling factor
            ref_latents = ref_latents * scaling_factor
        noise = torch.randn_like(ref_latents)
        timesteps = torch.randint(0, 1000, (1,), device="cuda")
        noisy_latents = ref_latents + noise * 0.1
        with torch.no_grad():
            pos, neg = pipe.encode_prompt(prompt=a["prompt"], negative_prompt=a["negative"],
                                          do_classifier_free_guidance=True, device="cuda")
        model_pred = pipe.transformer(hidden_states=noisy_latents, timestep=timesteps,
                                      encoder_hidden_states=pos).sample
        loss = torch.nn.functional.mse_loss(model_pred, ref_latents)
        loss.backward()
        optimizer.step()
        if step_idx % 50 == 0:
            log(f"train step {step_idx}: loss={loss.item():.4f}")

    log("LoRA training complete")
    lora_path = f"{a['out']}/lora_adapter"
    pipe.transformer.save_pretrained(lora_path)
    log(f"LoRA adapter saved to {lora_path}")

    # ============ PHASE 2: Inference with LoRA ============
    pipe.transformer.eval()
    pipe.transformer = pipe.transformer.to(torch.float16)

    x = (W - H) // 2
    first = Image.new("RGB", (W, H), (128, 128, 128))
    first.paste(ref.resize((H, H), Image.LANCZOS), (x, 0))

    all_frames = []
    report = {"segments": [], "reference_portrait": portrait_path,
              "total_seconds": 0.0, "lora_rank": a["lora_rank"],
              "train_steps": a["train_steps"], "lr": a["lr"],
              "lora_trainable_params": trainable_params}
    prev_last_frame = first

    for s in range(a["test_segments"]):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        if s == 0:
            video = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)
            mask = [Image.new("L", (W, H), 0)] + [Image.new("L", (W, H), 255)] * (N - 1)
            reference_images = [ref]
        else:
            video = [prev_last_frame] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)
            mask = [Image.new("L", (W, H), 0)] + [Image.new("L", (W, H), 255)] * (N - 1)
            reference_images = [ref, prev_last_frame]

        out = pipe(video=video, mask=mask, reference_images=reference_images,
                   prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                   num_frames=N, num_inference_steps=20, guidance_scale=5.0,
                   conditioning_scale=1.0, output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(42 + s)).frames[0]
        new_frames = out[1:] if s > 0 else out
        all_frames += new_frames
        prev_last_frame = out[-1]

        mid = (1 if s > 0 else 0) + (N - (1 if s > 0 else 0)) // 2
        frame_bgr = cv2.cvtColor(np.array(out[mid]), cv2.COLOR_RGB2BGR)
        emb = embed(frame_bgr)
        sim = float(emb @ ref_emb) if emb is not None else -1.0
        seg_seconds = (N - (1 if s > 0 else 0)) / 16.0
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

    export_to_video(all_frames, f"{a['out']}/lora_consistency_v2.mp4", fps=16)
    json.dump(report, open(f"{a['out']}/lora_consistency_v2.json", "w"), indent=2)
    log(f"lora v2 report: {json.dumps(report, indent=2)}")
    all_pass = all(s["passes_same_person"] for s in report["segments"])
    log(f"ALL SEGMENTS PASS SAME-PERSON MARK (0.363): {all_pass}")
    log(f"TOTAL VIDEO SECONDS: {report['total_seconds']:.2f}")
    """),
    encoding="utf-8",
)
(WORK / "lora_v3_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "lora_v3_args.json")],
                   capture_output=True, text=True, timeout=6 * 3600, env=env)
(OUT / "lora_v3.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-8000:] if r.returncode else r.stdout[-20000:], encoding="utf-8")
log(f"lora v3 rc={r.returncode}")
log("done")