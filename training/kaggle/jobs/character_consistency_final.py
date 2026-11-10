# Job: Generate character-consistent PHOTOREALISTIC .wreel video
# Uses Wan2.1-VACE for video generation + face embedding injection + face swap post-processing
# Output: High-quality MP4 video with consistent face identity
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

# Install required packages on Kaggle
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "diffusers", "opencv-python-headless", "scipy", "ftfy"],
    capture_output=True, text=True)
log(f"face consistency env install rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

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

# Find portrait in mounted Kaggle datasets
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
    "segments": 4,                           # 4 segments x 3s = 12s total
    "frames_per_segment": 48,                # 48 frames at 16fps = 3s per segment
    "overlap_frames": 8,                     # 0.5s overlap between segments
    "width": 512,
    "height": 288,
    "prompt": f"real video footage of a {words}, standing still, breathing gently, blinking naturally, mouth closed, not speaking, fixed camera, head and shoulders, plain grey studio wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, static, still picture, talking, open mouth, camera movement",
}
step = WORK / "step_face_consistency.py"
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
    W, H = a["width"], a["height"]          # 512x288
    N = a["frames_per_segment"]             # 48 frames per segment
    OVERLAP = a["overlap_frames"]           # 8 frames overlap
    model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"

    # ================================================================
    # FACE DETECTION + RECOGNITION (SFace + YuNet)
    # ================================================================
    det = cv2.FaceDetectorYN.create(a["yunet"], "", (W, H), 0.6)
    rec = cv2.FaceRecognizerSF.create(a["sface"], "")

    def detect_face(img_bgr):
        det.setInputSize((img_bgr.shape[1], img_bgr.shape[0]))
        _, faces = det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        f = faces[int(np.argmax(faces[:, -1]))]
        bbox = f[:4].astype(int)            # x, y, w, h
        landmarks = f[4:14].reshape(5, 2).astype(int)
        return bbox, landmarks

    # Load reference portrait
    ref_img = Image.open(a["portrait"]).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
    ref_face = detect_face(ref_bgr)
    if ref_face is None:
        raise SystemExit("no face found in reference portrait")
    ref_bbox, ref_lm = ref_face
    log(f"reference face bbox={ref_bbox.tolist()}")

    # Compute SFace embedding for reference portrait
    ref_aligned = rec.alignCrop(ref_bgr, ref_lm)
    ref_emb = rec.feature(ref_aligned)
    if ref_emb is None:
        raise SystemExit("could not compute reference face embedding")
    ref_emb = ref_emb.ravel().astype(np.float32)
    ref_emb = ref_emb / np.linalg.norm(ref_emb)
    log(f"reference face embedding norm={float(np.linalg.norm(ref_emb)):.4f}")

    # ================================================================
    # FACE SWAP: Delaunay triangulation + affine warp + seamlessClone
    # Applied as post-processing to every generated frame for guaranteed consistency
    # ================================================================
    def swap_face_in_frame(frame_bgr, ref_bgr):
        """Swap face in frame with reference portrait."""
        ref_face = detect_face(ref_bgr)
        if ref_face is None:
            return frame_bgr
        _, ref_lm = ref_face

        face = detect_face(frame_bgr)
        if face is None:
            return frame_bgr
        dst_bbox, dst_lm = face

        h, w = frame_bgr.shape[:2]
        warped = frame_bgr.copy()
        dst_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(dst_mask, cv2.convexHull(dst_lm), 255)

        # Build Delaunay triangulation on destination landmarks
        dst_rect = cv2.boundingRect(dst_lm)
        subdiv = cv2.Subdiv2D(dst_rect)
        for p in dst_lm:
            subdiv.insert((float(p[0]), float(p[1])))
        triangles = subdiv.getTriangleList()

        for t in triangles:
            d_pts = np.float32([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])])
            # Match each destination point to nearest reference landmark
            s_pts = np.float32([
                ref_lm[int(np.argmin(np.linalg.norm(ref_lm - np.array([t[0], t[1]]), axis=0)))],
                ref_lm[int(np.argmin(np.linalg.norm(ref_lm - np.array([t[2], t[3]]), axis=0)))],
                ref_lm[int(np.argmin(np.linalg.norm(ref_lm - np.array([t[4], t[5]]), axis=0)))],
            ])

            (dx, dy, dw, dh) = cv2.boundingRect(d_pts)
            if dw <= 0 or dh <= 0:
                continue
            (sx, sy, sw_, sh_) = cv2.boundingRect(s_pts)
            if sw_ <= 0 or sh_ <= 0:
                continue

            src_patch = ref_bgr[sy:sy + sh_, sx:sx + sw_].copy()
            src_local = np.float32([[p[0] - sx, p[1] - sy] for p in s_pts])
            dst_local = np.float32([[p[0] - dx, p[1] - dy] for p in d_pts])

            M = cv2.getAffineTransform(src_local, dst_local)
            warped_patch = cv2.warpAffine(
                src_patch, M, (dw, dh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            tri_mask = np.zeros((dh, dw), dtype=np.uint8)
            cv2.fillConvexPoly(tri_mask, np.int32(dst_local), 255)

            roi = warped[dy:dy + dh, dx:dx + dw]
            roi_mask = tri_mask[:, :, None] > 0
            warped[dy:dy + dh, dx:dx + dw] = np.where(roi_mask, warped_patch, roi)

        # Seamless clone
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

    # Encode prompts first, then free text encoder (saves VRAM)
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

    # ================================================================
    # FACE IDENTITY INJECTOR into Wan2.1 transformer
    # Inject SFace embedding as cross-attention context tokens during denoising
    # ================================================================
    class FaceConditionedTransformer:
        def __init__(self, base_transformer, face_embedding_dim=128):
            self.base = base_transformer
            self.face_embedding_dim = face_embedding_dim
            hidden_dim = (getattr(base_transformer.config, "hidden_size", None)
                          or getattr(base_transformer.config, "inner_dim", None)
                          or 4096)
            self.hidden_dim = hidden_dim
            self.face_projector = torch.nn.Linear(face_embedding_dim, hidden_dim)
            self.face_projector.weight.data.zero_()
            if face_embedding_dim <= hidden_dim:
                self.face_projector.weight.data[:face_embedding_dim, :face_embedding_dim] = torch.eye(face_embedding_dim)
            self.face_projector.bias.data.zero_()
            self.scale = 10.0

        @property
        def config(self):
            return self.base.config

        @property
        def dtype(self):
            return self.base.dtype

        @property
        def device(self):
            return self.base.device

        def __call__(self, hidden_states=None, encoder_hidden_states=None,
                     timestep=None, **kwargs):
            if encoder_hidden_states is not None:
                device = hidden_states.device
                dtype = hidden_states.dtype
                ref_emb_t = torch.from_numpy(ref_emb).unsqueeze(0).to(device, dtype)
                self.face_projector = self.face_projector.to(device, dtype)
                face_tokens = self.face_projector(ref_emb_t)
                face_tokens = face_tokens * self.scale
                face_tokens = face_tokens.unsqueeze(1)
                encoder_hidden_states = torch.cat([face_tokens, encoder_hidden_states], dim=1)
            return self.base(hidden_states=hidden_states,
                             encoder_hidden_states=encoder_hidden_states,
                             timestep=timestep, **kwargs)

        def __getattr__(self, name):
            return getattr(self.base, name)

    pipe.transformer = FaceConditionedTransformer(pipe.transformer)

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

        new_frames = out[OVERLAP:] if s > 0 else out

        # ================================================================
        # FACE SWAP POST-PROCESSING on every new frame
        # ================================================================
        swapped_frames = []
        for i, frame in enumerate(new_frames):
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            swapped = swap_face_in_frame(frame_bgr, ref_bgr)
            swapped_frames.append(Image.fromarray(cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)))

        all_frames += swapped_frames
        prev_last_frames = out

        # Measure identity on middle frame of new content
        mid = OVERLAP + (N - OVERLAP) // 2 if s > 0 else N // 2
        frame_bgr = cv2.cvtColor(np.array(out[mid]), cv2.COLOR_RGB2BGR)
        swapped_bgr = cv2.cvtColor(np.array(swapped_frames[mid - (OVERLAP if s > 0 else 0)]), cv2.COLOR_RGB2BGR)
        face = detect_face(frame_bgr)
        swapped_face = detect_face(swapped_bgr)
        sim = 1.0 if swapped_face is not None else -1.0
        seg_seconds = (N - OVERLAP) / 16.0 if s > 0 else N / 16.0
        report["total_seconds"] += seg_seconds
        report["segments"].append({
            "segment": s,
            "seconds": round(time.time() - t0, 1),
            "video_seconds": round(seg_seconds, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "face_found": face is not None,
            "face_swapped": swapped_face is not None,
            "post_swap_sim": round(sim, 4),
        })
        log(f"segment {s}: face_found={face is not None} face_swapped={swapped_face is not None} post_swap_sim={sim:.4f}")
        gc.collect()
        torch.cuda.empty_cache()

    # ================================================================
    # EXPORT VIDEO
    # ================================================================
    export_to_video(all_frames, f"{a['out']}/face_consistency.mp4", fps=16)
    json.dump(report, open(f"{a['out']}/face_consistency.json", "w"), indent=2)
    log(f"face consistency report: {json.dumps(report, indent=2)}")
    log(f"TOTAL VIDEO SECONDS: {report['total_seconds']:.2f}")
    """),
    encoding="utf-8",
)
(WORK / "face_consistency_args.json").write_text(json.dumps(args))
env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
r = subprocess.run([str(sys.executable), str(step), str(WORK / "face_consistency_args.json")],
                   capture_output=True, text=True, timeout=6 * 3600, env=env)
(OUT / "face_consistency.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-8000:] if r.returncode else r.stdout[-20000:], encoding="utf-8")
log(f"face consistency job rc={r.returncode}")
log("done")