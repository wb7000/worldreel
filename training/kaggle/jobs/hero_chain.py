# Job: a 10 s character clip from 2 s pieces (D-8.13b). Frame 0 is the portrait; each later segment
# keeps the last frames of the one before, with the portrait as reference throughout.
#   python -m tools.kaggle_runner run training/kaggle/jobs/hero_chain.py --scene data:vacecast
import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

from formats.wscript import read_wscript
from stages.generate.bible import expand
from stages.generate.portraits import appearance_words

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = [sys.executable, "-m", "uv"]
py = Path("/tmp/wan/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/wan"], check=True)
r = subprocess.run([*UV, "pip", "install", "--python", str(py), "diffusers", "transformers",
                    "accelerate", "safetensors", "torch", "ftfy", "imageio", "imageio-ffmpeg",
                    "sentencepiece"], capture_output=True, text=True)  # fmt: skip
log(f"wan install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

root = WORK / "cast"
marker = next(INPUT.rglob("C01.png"), None)
if marker is None:
    z = next(p for p in INPUT.rglob("data.zip") if "vacecast" in str(p))
    with zipfile.ZipFile(z) as zf:
        zf.extractall(root)
    marker = next(root.rglob("C01.png"))
root = marker.parent
chars = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript").header["dictionary"][
    "characters"
]
cid = "C03"
ident = expand(int(chars[cid]["identity"]["seed"]), str(chars[cid]["identity"]["requested"]))
words = ", ".join(appearance_words(ident, str(chars[cid].get("description", ""))))
args = {
    "cid": cid,
    "portrait": str(root / f"{cid}.png"),
    "out": str(OUT),
    "segments": 6,
    "overlap": 5,
    "prompt": f"real video footage of a {words}, standing still, breathing gently, blinking "
    "naturally, mouth closed, not speaking, fixed camera, head and shoulders, plain grey studio "
    "wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, "
    "shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
    "blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, "
    "static, still picture, talking, open mouth, camera movement",
}
step = WORK / "step_chain.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, sys, time, torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    a = json.load(open(sys.argv[1]))
    W, H, N, KEEP = 832, 480, 33, a["overlap"]
    INSET = 6  # the portrait's outer pixels are regenerated too, so no seam shows
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
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
    grey = Image.new("RGB", (W, H), (128, 128, 128))
    white = Image.new("L", (W, H), 255)
    black = Image.new("L", (W, H), 0)
    ref = Image.open(a["portrait"]).convert("RGB")
    first = grey.copy()
    x = (W - H) // 2
    first.paste(ref.resize((H, H), Image.LANCZOS), (x, 0))
    keep0 = white.copy()
    keep0.paste(0, (x + INSET, INSET, x + H - INSET, H - INSET))
    frames, joins, report = [], [], {"segments": []}
    for s in range(a["segments"]):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        if s == 0:
            video = [first, *[grey] * (N - 1)]
            mask = [keep0, *[white] * (N - 1)]
        else:
            video = [*frames[-KEEP:], *[grey] * (N - KEEP)]
            mask = [*[black] * KEEP, *[white] * (N - KEEP)]
        out = pipe(video=video, mask=mask, reference_images=[ref], prompt_embeds=pos,
                   negative_prompt_embeds=neg, height=H, width=W, num_frames=N,
                   num_inference_steps=30, guidance_scale=5.0, conditioning_scale=1.0,
                   output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(11 + s)).frames[0]
        if s:
            joins.append(len(frames))
        frames += out if s == 0 else out[KEEP:]
        report["segments"].append({"seconds": round(time.time() - t0, 1),
                                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                   "total_frames": len(frames)})
        report["joins"] = joins
        print(s, report["segments"][-1], flush=True)
        export_to_video(frames, f"{a['out']}/chain.mp4", fps=16)
        json.dump(report, open(f"{a['out']}/chain.json", "w"), indent=1)
        gc.collect()
        torch.cuda.empty_cache()
"""),
    encoding="utf-8",
)
(WORK / "chain_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "chain_args.json")],
                   capture_output=True, text=True, timeout=5 * 3600, env=env)  # fmt: skip
(OUT / "chain.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
log(f"chain rc={r.returncode} {r.stderr[-800:] if r.returncode else ''}")
log("done")
