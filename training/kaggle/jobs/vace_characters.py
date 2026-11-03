# Job: realistic characters without painting the photo (D-8.13). Each character's Blender depth
# orbit (structure only: camera, pose, body; dataset `vacecast`) + the portrait as a reference +
# appearance words -> Wan2.1-VACE-1.3B (Apache-2.0) generates skin, hair, lips and facial hair.
# The Higgsfield idea (identity reference -> generated video) with open weights.
#   python -m tools.kaggle_runner push-data runs/humans/vacecast --name vacecast
#   python -m tools.kaggle_runner run training/kaggle/jobs/vace_characters.py --scene data:vacecast
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
items = []
for cid, c in sorted(chars.items()):
    ident = expand(int(c["identity"]["seed"]), str(c["identity"]["requested"]))
    desc = str(c.get("description", ""))
    words = ", ".join(appearance_words(ident, desc))
    prompt = (
        f"real video footage of a {words}, {desc}, standing still, calm neutral expression, "
        "head and shoulders, plain dark grey wall behind, soft even natural light, "
        "natural skin with pores, natural hair, sharp focus, shot on 35mm film, photorealistic"
    )
    items.append({"cid": cid, "prompt": prompt, "portrait": str(root / f"{cid}.png"),
                  "frames": [str(f) for f in sorted((root / cid).glob("f*.png"))]})  # fmt: skip
    log(f"{cid}: {len(items[-1]['frames'])} depth frames")
args = {
    "items": items,
    "out": str(OUT),
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
    "blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin",
}
step = WORK / "step_vace.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, sys, time, torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    a = json.load(open(sys.argv[1]))
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    # run 1 ran out of memory: offload only swaps out the previous model, so the ~10 GB text
    # encoder stayed on the GPU while the VAE encoded the video. Encode prompts first, free it.
    pipe.text_encoder.to("cuda")
    embeds = {}
    with torch.no_grad():
        for it in a["items"]:
            embeds[it["cid"]] = pipe.encode_prompt(
                prompt=it["prompt"], negative_prompt=a["negative"],
                do_classifier_free_guidance=True, device=torch.device("cuda"))
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    report = {}
    for it in a["items"]:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        depth = [Image.open(f).convert("RGB") for f in it["frames"]]
        ref = Image.open(it["portrait"]).convert("RGB")
        pos, neg = embeds[it["cid"]]
        out = pipe(video=depth, reference_images=[ref], prompt_embeds=pos,
                   negative_prompt_embeds=neg, height=480, width=832,
                   num_frames=len(depth), num_inference_steps=30, guidance_scale=5.0,
                   conditioning_scale=1.0, output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(11)).frames[0]
        export_to_video(out, f"{a['out']}/{it['cid']}.mp4", fps=16)
        for i in (0, len(out) // 2, len(out) - 1):
            out[i].save(f"{a['out']}/{it['cid']}_f{i:03d}.png")
        report[it["cid"]] = {"seconds": round(time.time() - t0, 1),
                             "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                             "frames": len(out), "prompt": it["prompt"]}
        print(it["cid"], report[it["cid"]], flush=True)
        json.dump(report, open(f"{a['out']}/vace.json", "w"), indent=1)
        gc.collect()
        torch.cuda.empty_cache()
"""),
    encoding="utf-8",
)
(WORK / "vace_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "vace_args.json")],
                   capture_output=True, text=True, timeout=5 * 3600, env=env)  # fmt: skip
(OUT / "vace.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
log(f"vace rc={r.returncode} {r.stderr[-800:] if r.returncode else ''}")
log("done")
