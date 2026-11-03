# Job: hero still -> video (D-8.13), the Higgsfield order with open weights. The depth-only run
# (`vace_characters.py`) copied the bald depth render, so here the chosen portrait IS frame 0:
# VACE keeps its pixels, outpaints the sides to 832 x 480 and generates the motion.
# Grid, fixed before the run:
#   A first frame only, all four characters
#   B depth orbit, conditioning_scale 0.5, reference only (C03)
#   C first frame + depth orbit on later frames, conditioning_scale 0.6 (C03)
#   python -m tools.kaggle_runner run training/kaggle/jobs/hero_motion.py --scene data:vacecast
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
        f"real video footage of a {words}, {desc}, breathing gently, blinking naturally, "
        "the camera slowly moves around them, head and shoulders, plain grey studio wall behind, "
        "soft even natural light, natural skin with pores, natural hair, sharp focus, "
        "shot on 35mm film, photorealistic"
    )
    items.append({"cid": cid, "prompt": prompt, "portrait": str(root / f"{cid}.png"),
                  "frames": [str(f) for f in sorted((root / cid).glob("f*.png"))]})  # fmt: skip
    log(f"{cid}: {len(items[-1]['frames'])} depth frames")
args = {
    "items": items,
    "out": str(OUT),
    "runs": [
        {"name": f"A_{it['cid']}", "cid": it["cid"], "first": True, "depth": False, "scale": 1.0}
        for it in items
    ] + [
        {"name": "B_C03", "cid": "C03", "first": False, "depth": True, "scale": 0.5},
        {"name": "C_C03", "cid": "C03", "first": True, "depth": True, "scale": 0.6},
    ],
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
    "blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, "
    "static, still picture",
}
step = WORK / "step_hero.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, sys, time, torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    a = json.load(open(sys.argv[1]))
    W, H, N = 832, 480, 33
    INSET = 6  # the portrait's outer pixels are regenerated too, so no seam shows
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    items = {it["cid"]: it for it in a["items"]}
    pipe.text_encoder.to("cuda")
    embeds = {}
    with torch.no_grad():
        for cid, it in items.items():
            embeds[cid] = pipe.encode_prompt(
                prompt=it["prompt"], negative_prompt=a["negative"],
                do_classifier_free_guidance=True, device=torch.device("cuda"))
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    grey = Image.new("RGB", (W, H), (128, 128, 128))
    white = Image.new("L", (W, H), 255)
    report = {}
    for run in a["runs"]:
        it = items[run["cid"]]
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        ref = Image.open(it["portrait"]).convert("RGB")
        depth = [Image.open(f).convert("RGB") for f in it["frames"]]
        video = depth[:N] if run["depth"] else [grey] * N
        mask = [white] * N
        if run["first"]:
            face = ref.resize((H, H), Image.LANCZOS)
            x = (W - H) // 2
            first = grey.copy()
            first.paste(face, (x, 0))
            keep = white.copy()
            keep.paste(0, (x + INSET, INSET, x + H - INSET, H - INSET))
            video = [first, *video[1:]]
            mask = [keep, *mask[1:]]
        pos, neg = embeds[run["cid"]]
        out = pipe(video=video, mask=mask, reference_images=[ref], prompt_embeds=pos,
                   negative_prompt_embeds=neg, height=H, width=W, num_frames=N,
                   num_inference_steps=30, guidance_scale=5.0,
                   conditioning_scale=run["scale"], output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(11)).frames[0]
        export_to_video(out, f"{a['out']}/{run['name']}.mp4", fps=16)
        for i in (0, len(out) // 2, len(out) - 1):
            out[i].save(f"{a['out']}/{run['name']}_f{i:03d}.png")
        report[run["name"]] = {**run, "seconds": round(time.time() - t0, 1),
                               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                               "frames": len(out), "prompt": it["prompt"]}
        print(run["name"], report[run["name"]], flush=True)
        json.dump(report, open(f"{a['out']}/hero.json", "w"), indent=1)
        gc.collect()
        torch.cuda.empty_cache()
"""),
    encoding="utf-8",
)
(WORK / "hero_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "hero_args.json")],
                   capture_output=True, text=True, timeout=5 * 3600, env=env)  # fmt: skip
(OUT / "hero.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
log(f"hero rc={r.returncode} {r.stderr[-800:] if r.returncode else ''}")
log("done")
