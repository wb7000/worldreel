# Job: long-shot character consistency — generate 5-second segments chained
# together, measure SFace drift across the whole sequence.
# Target: identity stays above 0.363 same-person mark across 15+ seconds.
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_consistency_long.py --scene data:vacecast --wait
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
py = Path("/tmp/consistency_long/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/consistency_long"], check=True)
r = subprocess.run(
    [*UV, "pip", "install", "--python", str(py), "diffusers", "transformers",
     "accelerate", "safetensors", "torch", "ftfy", "imageio", "imageio-ffmpeg",
     "sentencepiece", "opencv-python-headless"],
    capture_output=True, text=True)
log(f"long consistency env install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

# Copy face models from mounted Kaggle dataset (reliable, no GitHub rate limits)
MODEL_DIR = WORK / "models"
MODEL_DIR.mkdir(exist_ok=True)
import shutil
MODEL_DATASET = Path("/kaggle/input/datasets/wb3none/worldreel-data-facemodels")
for name in (
    "face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx",
):
    src = MODEL_DATASET / name
    dst = MODEL_DIR / name
    if src.exists():
        shutil.copy2(str(src), str(dst))
        log(f"copied {name} size={dst.stat().st_size}")
    else:
        log(f"WARNING: {name} not found in mounted dataset at {src}")

chars = read_wscript(CODE / "eval" / "scripts" / "lantern_keeper.wscript").header["dictionary"][
    "characters"
]
cid = "C01"
ident = expand(int(chars[cid]["identity"]["seed"]), str(chars[cid]["identity"]["requested"]))
words = ", ".join(appearance_words(ident, str(chars[cid].get("description", ""))))

args = {
    "cid": cid,
    "portrait": None,
    "yunet": str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"),
    "sface": str(MODEL_DIR / "face_recognition_sface_2021dec.onnx"),
    "out": str(OUT),
    "segments": 5,  # 5 segments x 5s = 25 seconds total
    "frames_per_segment": 80,  # 5s at 16 fps
    "overlap_frames": 10,  # keep last 10 frames as reference for next segment
    "prompt": f"real video footage of a {words}, standing still, breathing gently, blinking "
    "naturally, mouth closed, not speaking, fixed camera, head and shoulders, plain grey studio "
    "wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, "
    "shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
    "blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, "
    "static, still picture, talking, open mouth, camera movement",
}
step = WORK / "step_consistency_long.py"
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
    # If portrait path is None (bug in outer script), search for it in mounted datasets
    if a.get("portrait") is None:
        from pathlib import Path as _P
        import zipfile as _zipfile
        INPUT2 = _P("/kaggle/input/datasets/wb3none/worldreel-data-vacecast")
        marker2 = next(INPUT2.rglob(f"{a['cid']}_*.png"), None) or next(INPUT2.rglob(f"{a['cid']}.png"), None)
        if marker2 is None:
            for z in INPUT2.rglob("data.zip"):
                if "vacecast" in str(z):
                    with _zipfile.ZipFile(z) as zf:
                        zf.extractall(INPUT2 / "cast")
                    break
            marker2 = next(INPUT2.rglob(f"{a['cid']}_*.png"), None)
        a["portrait"] = str(marker2.parent / f"{a['cid']}_5.png")
        log(f"found portrait at: {a['portrait']}")
    N = a["frames_per_segment"]
    OVERLAP = a["overlap_frames"]
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

    # Find model files in mounted datasets (search recursively)
    import glob as _glob
    yunet_path = a["yunet"]
    sface_path = a["sface"]
    for pattern in ["/kaggle/input/**/face_detection_yunet_2023mar.onnx",
                    "/kaggle/input/**/face_recognition_sface_2021dec.onnx"]:
        for found in _glob.glob(pattern, recursive=True):
            if "face_detection_yunet" in found:
                yunet_path = found
            if "face_recognition_sface" in found:
                sface_path = found
    log(f"using yunet: {yunet_path}, sface: {sface_path}")
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
    report = {"segments": [], "reference_portrait": a["portrait"],
              "total_seconds": 0.0}
    prev_last_frames = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)
    for seg in range(a["segments"]):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        if seg == 0:
            video = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)
            mask = [Image.new("L", (W, H), 0)] + [Image.new("L", (W, H), 255)] * (N - 1)
        else:
            # Chain: keep last OVERLAP frames as reference, regenerate the rest
            video = [*prev_last_frames[-OVERLAP:],
                     *[Image.new("RGB", (W, H), (128, 128, 128))] * (N - OVERLAP)]
            mask = [Image.new("L", (W, H), 0)] * OVERLAP + \
                   [Image.new("L", (W, H), 255)] * (N - OVERLAP)
        out = pipe(video=video, mask=mask, reference_images=[ref], prompt_embeds=pos,
                   negative_prompt_embeds=neg, height=H, width=W, num_frames=N,
                   num_inference_steps=30, guidance_scale=5.0, conditioning_scale=1.0,
                   output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(21 + seg)).frames[0]
        # Only keep new frames (not the overlap reference frames)
        new_frames = out[OVERLAP:] if seg > 0 else out
        all_frames += new_frames
        prev_last_frames = out
        # measure identity on middle frame of new content
        mid_new = OVERLAP + (N - OVERLAP) // 2 if seg > 0 else N // 2
        frame_bgr = cv2.cvtColor(np.array(out[mid_new]), cv2.COLOR_RGB2BGR)
        emb = embed(frame_bgr)
        sim = float(emb @ ref_emb) if emb is not None else -1.0
        seg_seconds = (N - OVERLAP) / 16.0 if seg > 0 else N / 16.0
        report["total_seconds"] += seg_seconds
        report["segments"].append({
            "segment": seg,
            "seconds": round(time.time() - t0, 1),
            "video_seconds": round(seg_seconds, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "sface_vs_ref": round(sim, 4),
            "face_found": emb is not None,
            "passes_same_person": bool(emb is not None and sim >= 0.363),
        })
        log(f"segment {seg}: SFace={sim:.4f} face_found={emb is not None} "
            f"passes_same_person={report['segments'][-1]['passes_same_person']}")
        gc.collect()
        torch.cuda.empty_cache()

    export_to_video(all_frames, f"{a['out']}/consistency_long.mp4", fps=16)
    json.dump(report, open(f"{a['out']}/consistency_long.json", "w"), indent=2)
    log(f"consistency long report: {json.dumps(report, indent=2)}")
    all_pass = all(s["passes_same_person"] for s in report["segments"])
    log(f"ALL SEGMENTS PASS SAME-PERSON MARK (0.363): {all_pass}")
    log(f"TOTAL VIDEO SECONDS: {report['total_seconds']:.2f}")
    """),
    encoding="utf-8",
)
(WORK / "consistency_long_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "consistency_long_args.json")],
                   capture_output=True, text=True, timeout=6 * 3600, env=env)  # fmt: skip
(OUT / "consistency_long.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-8000:] if r.returncode else r.stdout[-20000:], encoding="utf-8")
log(f"long consistency rc={r.returncode}")
log("done")
