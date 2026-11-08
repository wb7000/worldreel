# Job: chained character-consistent video generation with previous-frame reference
# Uses previous segment's last frame as reference image (along with portrait) to
# maintain identity across shots without blur or glow from post-processing.
#   python -m tools.kaggle_runner run training/kaggle/jobs/video_chain_v2.py --scene data:vacecast --scene data:facemodels --wait
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
py = Path("/tmp/chain2/bin/python")
subprocess.run([*UV, "venv", "--python", "3.12", "/tmp/chain2"], check=True)
r = subprocess.run(
    [*UV, "pip", "install", "--python", str(py), "diffusers", "transformers",
     "accelerate", "safetensors", "torch", "ftfy", "imageio", "imageio-ffmpeg",
     "sentencepiece", "opencv-python-headless"],
    capture_output=True, text=True)
log(f"chain v2 env install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

# Find models
import glob as _glob
MODEL_DIR = Path("/kaggle/input/datasets/wb3none/worldreel-data-facemodels")
yunet_path = None
sface_path = None
for p in MODEL_DIR.rglob("face_detection_yunet_2023mar.onnx"):
    yunet_path = str(p)
for p in MODEL_DIR.rglob("face_recognition_sface_2021dec.onnx"):
    sface_path = str(p)

# Find portrait
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
words = ", ".join(appearance_words(ident, str(chars[cid].get("description", ""))))

args = {
    "cid": cid,
    "portrait": portrait_path,
    "yunet": yunet_path,
    "sface": sface_path,
    "out": str(OUT),
    "segments": 5,
    "frames_per_segment": 80,  # 5s at 16 fps
    "prompt": f"real video footage of a {words}, standing still, breathing gently, blinking "
    "naturally, mouth closed, not speaking, fixed camera, head and shoulders, plain grey studio "
    "wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, "
    "shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
    "blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, "
    "static, still picture, talking, open mouth, camera movement",
}
step = WORK / "step_chain_v2.py"
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
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
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
    report = {"segments": [], "reference_portrait": a["portrait"],
              "total_seconds": 0.0}
    prev_last_frame = first  # Start with portrait as first reference

    for s in range(a["segments"]):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        if s == 0:
            # First segment: portrait as reference, rest grey
            video = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)
            mask = [Image.new("L", (W, H), 0)] + [Image.new("L", (W, H), 255)] * (N - 1)
            reference_images = [ref]  # portrait only for first segment
        else:
            # Chain: use previous segment's last frame as reference (along with portrait)
            # This maintains identity across segments
            video = [prev_last_frame] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)
            mask = [Image.new("L", (W, H), 0)] + [Image.new("L", (W, H), 255)] * (N - 1)
            reference_images = [ref, prev_last_frame]  # portrait + last frame

        out = pipe(video=video, mask=mask, reference_images=reference_images,
                   prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                   num_frames=N, num_inference_steps=30, guidance_scale=5.0,
                   conditioning_scale=1.0, output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(21 + s)).frames[0]
        new_frames = out[1:] if s > 0 else out  # Skip the reference frame for s>0
        all_frames += new_frames
        prev_last_frame = out[-1]  # Last frame of this segment becomes reference for next

        # Measure identity on middle frame of new content
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

    export_to_video(all_frames, f"{a['out']}/consistency_chain_v2.mp4", fps=16)
    json.dump(report, open(f"{a['out']}/consistency_chain_v2.json", "w"), indent=2)
    log(f"chain v2 report: {json.dumps(report, indent=2)}")
    all_pass = all(s["passes_same_person"] for s in report["segments"])
    log(f"ALL SEGMENTS PASS SAME-PERSON MARK (0.363): {all_pass}")
    log(f"TOTAL VIDEO SECONDS: {report['total_seconds']:.2f}")
    """),
    encoding="utf-8",
)
(WORK / "chain_v2_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(py), str(step), str(WORK / "chain_v2_args.json")],
                   capture_output=True, text=True, timeout=6 * 3600, env=env)
(OUT / "chain_v2.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-8000:] if r.returncode else r.stdout[-20000:], encoding="utf-8")
log(f"chain v2 rc={r.returncode}")
log("done")