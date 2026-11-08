# Job: character consistency test — generate 3 shots of the same character,
# measure SFace drift across shots (target: stay above 0.363 same-person mark).
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_consistency.py --scene data:vacecast --wait
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
py = Path("/tmp/consistency/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/consistency"], check=True)
r = subprocess.run(
    [*UV, "pip", "install", "--python", str(py), "diffusers", "transformers",
     "accelerate", "safetensors", "torch", "ftfy", "imageio", "imageio-ffmpeg",
     "sentencepiece", "opencv-python-headless"],
    capture_output=True, text=True)
log(f"consistency env install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

# Download face models from OpenCV Zoo (not available on Kaggle by default)
MODEL_DIR = WORK / "models"
MODEL_DIR.mkdir(exist_ok=True)
subprocess.run([
    str(py), "-m", "pip", "install", "--python", str(py), "-q", "gdown"
], capture_output=True, text=True)
# YuNet face detector and SFace face recognizer from OpenCV Zoo
for name, url in (
    ("face_detection_yunet_2023mar.onnx",
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    ("face_recognition_sface_2021dec.onnx",
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
):
    dst = MODEL_DIR / name
    if not dst.exists():
        r = subprocess.run(
            ["curl", "-sL", "-o", str(dst), url],
            capture_output=True, text=True)
        log(f"download {name} rc={r.returncode} size={dst.stat().st_size if dst.exists() else 0}")

root = WORK / "cast"
# Look for any portrait file matching this character (C01_*.png, C01.png, etc.)
marker = next(INPUT.rglob(f"{a['cid']}_*.png"), None) or next(INPUT.rglob(f"{a['cid']}.png"), None)
if marker is None:
    # Fallback: look for data.zip with vacecast in path
    zip_files = list(INPUT.rglob("data.zip"))
    vace_zips = [p for p in zip_files if "vacecast" in str(p)]
    if vace_zips:
        with zipfile.ZipFile(vace_zips[0]) as zf:
            zf.extractall(root)
        marker = next(root.rglob(f"{cid}_*.png"), None) or next(root.rglob(f"{cid}.png"), None)
if marker is None:
    raise FileNotFoundError(f"no portrait file for {cid} found in mounted datasets")
root = marker.parent

chars = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript").header["dictionary"][
    "characters"
]
cid = "C01"
ident = expand(int(chars[cid]["identity"]["seed"]), str(chars[cid]["identity"]["requested"]))
words = ", ".join(appearance_words(ident, str(chars[cid].get("description", ""))))
portrait = str(root / f"{cid}_5.png")  # chosen portrait from D-8.10b

args = {
    "cid": cid,
    "portrait": portrait,
    "yunet": str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"),
    "sface": str(MODEL_DIR / "face_recognition_sface_2021dec.onnx"),
    "out": str(OUT),
    "shots": 3,
    "frames_per_shot": 33,  # ~2 s at 16 fps
    "prompt": f"real video footage of a {words}, standing still, breathing gently, blinking "
    "naturally, mouth closed, not speaking, fixed camera, head and shoulders, plain grey studio "
    "wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, "
    "shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
    "blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, "
    "static, still picture, talking, open mouth, camera movement",
}
step = WORK / "step_consistency.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, sys, time, torch, cv2, numpy as np
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    a = json.load(open(sys.argv[1]))
    W, H, N = 832, 480, a["frames_per_shot"]
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

    ref = Image.open(a["portrait"]).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2BGR)
    ref_emb = embed(ref_bgr)
    if ref_emb is None:
        raise SystemExit("no face found in reference portrait")
    log(f"reference face embedding norm: {float(np.linalg.norm(ref_emb)):.4f}")

    x = (W - H) // 2
    first = Image.new("RGB", (W, H), (128, 128, 128))
    first.paste(ref.resize((H, H), Image.LANCZOS), (x, 0))

    all_frames = []
    report = {"shots": [], "reference_portrait": a["portrait"]}
    for s in range(a["shots"]):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        if s == 0:
            video = [first, *[Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)]
            mask = [Image.new("L", (W, H), 0)] + [Image.new("L", (W, H), 255)] * (N - 1)
        else:
            video = [*all_frames[-5:], *[Image.new("RGB", (W, H), (128, 128, 128))] * (N - 5)]
            mask = [Image.new("L", (W, H), 0)] * 5 + [Image.new("L", (W, H), 255)] * (N - 5)
        out = pipe(video=video, mask=mask, reference_images=[ref], prompt_embeds=pos,
                   negative_prompt_embeds=neg, height=H, width=W, num_frames=N,
                   num_inference_steps=30, guidance_scale=5.0, conditioning_scale=1.0,
                   output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(11 + s)).frames[0]
        all_frames += out if s == 0 else out[5:]
        # measure identity on middle frame of this shot
        mid = N // 2
        frame_bgr = cv2.cvtColor(np.array(out[mid]), cv2.COLOR_RGB2BGR)
        emb = embed(frame_bgr)
        sim = float(emb @ ref_emb) if emb is not None else -1.0
        shot_report = {
            "shot": s,
            "seconds": round(time.time() - t0, 1),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "sface_vs_ref": round(sim, 4),
            "face_found": emb is not None,
            "passes_same_person": bool(emb is not None and sim >= 0.363),
        }
        report["shots"].append(shot_report)
        log(f"shot {s}: SFace={sim:.4f} face_found={emb is not None} "
            f"passes_same_person={shot_report['passes_same_person']}")
        gc.collect()
        torch.cuda.empty_cache()

    export_to_video(all_frames, f"{a['out']}/consistency.mp4", fps=16)
    json.dump(report, open(f"{a['out']}/consistency.json", "w"), indent=2)
    log(f"consistency report: {json.dumps(report, indent=2)}")
    all_pass = all(s["passes_same_person"] for s in report["shots"])
    log(f"ALL SHOTS PASS SAME-PERSON MARK (0.363): {all_pass}")
    """),
    encoding="utf-8",
)
(WORK / "consistency_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "consistency_args.json")],
                   capture_output=True, text=True, timeout=5 * 3600, env=env)  # fmt: skip
(OUT / "consistency.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:], encoding="utf-8")
log(f"consistency rc={r.returncode} {r.stderr[-800:] if r.returncode else ''}")
log("done")
