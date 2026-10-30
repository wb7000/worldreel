# Job: realism-pass test (D-8.12). Our Blender render of Ade (33-frame camera orbit, 832 x 480,
# dataset `adeorbit`) re-painted by Wan2.1-T2V-1.3B through diffusers' WanVideoToVideoPipeline
# (Apache-2.0) at two strengths: the 3D scene keeps the camera, blocking and identity, the video
# model adds photographic realism (the Higgsfield-style look, with open weights).
#   python -m tools.kaggle_runner push-data runs/humans/C02/orbit --name adeorbit
#   python -m tools.kaggle_runner run training/kaggle/jobs/realism_pass.py --scene data:adeorbit
import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
py = Path("/tmp/wan/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/wan"], check=True)
r = subprocess.run([*UV, "pip", "install", "--python", str(py), "diffusers", "transformers",
                    "accelerate", "safetensors", "torch", "ftfy", "imageio", "imageio-ffmpeg",
                    "sentencepiece"], capture_output=True, text=True)  # fmt: skip
log(f"wan install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

frames_dir = WORK / "orbit"
first = next(INPUT.rglob("f000.png"), None)
if first is None:
    z = next(p for p in INPUT.rglob("data.zip") if "adeorbit" in str(p))
    with zipfile.ZipFile(z) as zf:
        zf.extractall(frames_dir)
    first = next(frames_dir.rglob("f000.png"))
frames = sorted(first.parent.glob("f*.png"))
log(f"{len(frames)} input frames from {first.parent}")

args = {
    "frames": [str(f) for f in frames],
    "out": str(OUT),
    "strengths": [0.35, 0.5],
    "prompt": "a real photograph of an elderly Black man in his seventies with short white hair, "
    "white stubble, deep wrinkles and brown eyes, wearing a plain white t-shirt, standing still "
    "against a dark grey studio wall, soft natural light, 35mm film, sharp focus, realistic skin "
    "texture, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, blurry, "
    "deformed face, extra eyes, text, watermark, overexposed",
}
step = WORK / "step_realism.py"
step.write_text(
    textwrap.dedent("""
    import json, sys, time, torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    a = json.load(open(sys.argv[1]))
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    # T4 has no native bf16: fp16 transformer, fp32 VAE (diffusers' advice for decoding quality)
    pipe = WanVideoToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.enable_model_cpu_offload()
    # run 1 ran out of memory in the fp32 VAE encoding all 33 frames at once (14.56 GB T4)
    pipe.vae.enable_tiling()
    video = [Image.open(f).convert("RGB") for f in a["frames"]]
    report = {}
    for s in a["strengths"]:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        out = pipe(video=video, prompt=a["prompt"], negative_prompt=a["negative"], height=480,
                   width=832, guidance_scale=5.0, strength=s, num_inference_steps=30,
                   generator=torch.Generator("cpu").manual_seed(7)).frames[0]
        tag = f"s{int(s * 100):02d}"
        export_to_video(out, f"{a['out']}/realism_{tag}.mp4", fps=16)
        for i in (0, len(out) // 2, len(out) - 1):
            img = out[i]
            if not isinstance(img, Image.Image):
                import numpy as np
                img = Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8"))
            img.save(f"{a['out']}/realism_{tag}_f{i:03d}.png")
        report[tag] = {"seconds": round(time.time() - t0, 1),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                       "frames": len(out)}
        print(tag, report[tag], flush=True)
    json.dump(report, open(f"{a['out']}/realism.json", "w"))
"""),
    encoding="utf-8",
)
(WORK / "realism_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "realism_args.json")],
                   capture_output=True, text=True, timeout=4 * 3600, env=env)  # fmt: skip
(OUT / "realism.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
log(f"realism rc={r.returncode} {r.stderr[-800:] if r.returncode else ''}")
log("done")
