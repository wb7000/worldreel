# Job: Generate character-consistent .wreel video with face swap post-processing.
# Uses Kaggle's system Python (PyTorch pre-installed) to avoid slow uv venv setup.
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

# Install only missing packages on Kaggle's system Python
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "diffusers", "opencv-python-headless", "scipy"],
    capture_output=True, text=True)
log(f"pip install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

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
    "segments": 3,
    "frames_per_segment": 12,
    "overlap_frames": 6,
    "prompt": f"real video footage of a {words}, standing still, breathing gently, blinking "
    "naturally, mouth closed, not speaking, fixed camera, head and shoulders, plain grey studio "
    "wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, "
    "shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, "
    "blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, "
    "static, still picture, talking, open mouth, camera movement",
}
step = WORK / "step_face_swap.py"
step.write_text(
    textwrap.dedent("""
    import gc, json, sys, time, torch, cv2, numpy as np
    from pathlib import Path
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    a = json.load(open(sys.argv[1]))
    log = print
    W, H = 512, 288
    N = a["frames_per_segment"]
    OVERLAP = a["overlap_frames"]
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"

    # ================================================================
    # FACE SWAP ENGINE (Deep-Live-Cam style)
    # ================================================================
    det = cv2.FaceDetectorYN.create(a["yunet"], "", (W, H), 0.6)

    def detect_face(img_bgr):
        det.setInputSize((img_bgr.shape[1], img_bgr.shape[0]))
        _, faces = det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        f = faces[int(np.argmax(faces[:, -1]))]
        bbox = f[:4].astype(int)
        landmarks = f[4:14].reshape(5, 2).astype(int)
        return bbox, landmarks

    ref_img = Image.open(a["portrait"]).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
    ref_face = detect_face(ref_bgr)
    if ref_face is None:
        raise SystemExit("no face found in reference portrait")
    ref_bbox, ref_lm = ref_face
    log(f"reference face bbox={ref_bbox.tolist()}")

    # Build Delaunay triangulation on reference landmarks (once)
    ref_hull = cv2.convexHull(ref_lm)
    ref_rect = cv2.boundingRect(ref_hull)
    ref_subdiv = cv2.Subdiv2D(ref_rect)
    ref_subdiv.insert([(int(p[0]), int(p[1])) for p in ref_lm])
    ref_triangles = ref_subdiv.getTriangleList()
    ref_tri_idx = []
    for t in ref_triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for p in pts:
            d = np.linalg.norm(ref_lm - np.array(p), axis=1)
            idx.append(int(np.argmin(d)))
        ref_tri_idx.append(idx)
    log(f"reference triangulation: {len(ref_tri_idx)} triangles")

    def swap_face_in_frame(frame_bgr):
        face = detect_face(frame_bgr)
        if face is None:
            return frame_bgr
        dst_bbox, dst_lm = face

        h, w = frame_bgr.shape[:2]
        warped = frame_bgr.copy()
        dst_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(dst_mask, cv2.convexHull(dst_lm), 255)

        for tri_idx in ref_tri_idx:
            src_pts = np.float32([ref_lm[i] for i in tri_idx])
            dst_pts = np.float32([dst_lm[i] for i in tri_idx])

            (dx, dy, dw, dh) = cv2.boundingRect(dst_pts)
            if dw <= 0 or dh <= 0:
                continue

            (sx, sy, sw_, sh_) = cv2.boundingRect(src_pts)
            if sw_ <= 0 or sh_ <= 0:
                continue

            src_patch = ref_bgr[sy:sy + sh_, sx:sx + sw_].copy()

            src_pts_local = np.float32([[p[0] - sx, p[1] - sy] for p in src_pts])
            dst_pts_local = np.float32([[p[0] - dx, p[1] - dy] for p in dst_pts])

            M = cv2.getAffineTransform(src_pts_local, dst_pts_local)
            warped_patch = cv2.warpAffine(src_patch, M, (dw, dh),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT_101)

            tri_mask = np.zeros((dh, dw), dtype=np.uint8)
            cv2.fillConvexPoly(tri_mask, np.int32(dst_pts_local), 255)

            roi = warped[dy:dy + dh, dx:dx + dw]
            roi_mask = tri_mask[:, :, None] > 0
            warped[dy:dy + dh, dx:dx + dw] = np.where(roi_mask, warped_patch, roi)

        center = (int(dst_lm[:, 0].mean()), int(dst_lm[:, 1].mean()))
        center = (max(10, min(w - 10, center[0])), max(10, min(h - 10, center[1])))
        try:
            result = cv2.seamlessClone(warped, frame_bgr, dst_mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            result = warped
        return result

    # ================================================================
    # LOAD WAN2.1-VACE PIPELINE
    # ================================================================
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
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

    x = (W - H) // 2
    first = Image.new("RGB", (W, H), (128, 128, 128))
    first.paste(ref_img.resize((H, H), Image.LANCZOS), (x, 0))

    all_frames = []
    report = {"segments": [], "reference_portrait": a["portrait"],
              "total_seconds": 0.0}
    prev_last_frames = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)

    for s in range(a["segments"]):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        if s == 0:
            video = [first] + [Image.new("RGB", (W, H), (128, 128, 128))] * (N - 1)
            mask = [Image.new("L", (W, H), 0)] + [Image.new("L", (W, H), 255)] * (N - 1)
            reference_images = [ref_img]
        else:
            video = [*prev_last_frames[-OVERLAP:],
                     *[Image.new("RGB", (W, H), (128, 128, 128))] * (N - OVERLAP)]
            mask = [Image.new("L", (W, H), 0)] * OVERLAP + \
                   [Image.new("L", (W, H), 255)] * (N - OVERLAP)
            reference_images = [ref_img, prev_last_frames[-1]]

        log(f"segment {s}: starting generation")
        out = pipe(video=video, mask=mask, reference_images=reference_images,
                   prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                   num_frames=N, num_inference_steps=30, guidance_scale=5.0,
                   conditioning_scale=1.0, output_type="pil",
                   generator=torch.Generator("cpu").manual_seed(42 + s)).frames[0]

        # ================================================================
        # POST-PROCESS: swap face on every generated frame
        # ================================================================
        processed = []
        frames_swapped = 0
        for i, frame in enumerate(out):
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            swapped = swap_face_in_frame(frame_bgr)
            if not np.array_equal(swapped, frame_bgr):
                frames_swapped += 1
            processed.append(Image.fromarray(cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)))
            if i % 4 == 0:
                log(f"  segment {s}: processed {i + 1}/{len(out)} frames")
        out = processed

        new_frames = out[OVERLAP:] if s > 0 else out
        all_frames += new_frames
        prev_last_frames = out

        mid = OVERLAP + (N - OVERLAP) // 2 if s > 0 else N // 2
        face = detect_face(cv2.cvtColor(np.array(out[mid]), cv2.COLOR_RGB2BGR))
        sim = 1.0 if face is not None else -1.0
        seg_seconds = (N - OVERLAP) / 16.0 if s > 0 else N / 16.0
        report["total_seconds"] += seg_seconds
        report["segments"].append({
            "segment": s,
            "seconds": round(time.time() - t0, 1),
            "video_seconds": round(seg_seconds, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "frames_swapped": frames_swapped,
            "face_found": face is not None,
        })
        log(f"segment {s}: done frames_swapped={frames_swapped}/{len(out)} face_found={face is not None}")
        gc.collect()
        torch.cuda.empty_cache()

    export_to_video(all_frames, f"{a['out']}/face_consistency.mp4", fps=16)
    json.dump(report, open(f"{a['out']}/face_consistency.json", "w"), indent=2)
    log(f"face consistency report: {json.dumps(report, indent=2)}")
    log(f"TOTAL VIDEO SECONDS: {report['total_seconds']:.2f}")
    """),
    encoding="utf-8",
)
(WORK / "face_swap_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(sys.executable), str(step), str(WORK / "face_swap_args.json")],
                   capture_output=True, text=True, timeout=6 * 3600, env=env)
(OUT / "face_consistency.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-8000:] if r.returncode else r.stdout[-20000:], encoding="utf-8")
log(f"face consistency job rc={r.returncode}")
log("done")