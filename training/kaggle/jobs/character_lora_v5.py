# Job: Character LoRA v5 -- train against MOTION-rich targets (D-9.26).
#
# The v4 adapter (D-9.24) was trained on a still portrait plus micro-jitter and it
# learned exactly that: the render came back as a still image that shakes -- 211 of
# 239 consecutive frames differed by < 0.5 grey levels (base model: 0 of 239), and the
# face Laplacian collapsed from 66 to 15 (D-9.25). The identity was excellent
# (0.9261 mean) and the framing stopped walking between takes, but there was no
# motion left to watch: the reward loop only measured identity, and the only target
# it had was a photograph.
#
# v5 fixes the cause, not the reward:
#   * The TARGET is a real rendered video (the D-9.25 reference render, published as
#     the private dataset worldreel-data-v12render). Each training variant is
#     [conditioning seed frame] + [48 frames of one take], so the target carries the
#     character's real motion AND its real texture -- the two things the v4 target
#     destroyed. The takes are ranked by measured identity and the best ones are
#     used, which also lifts the target's identity ceiling (raw takes: 0.84-0.91).
#   * The REWARD is still identity, but it is now GATED on the other two axes the
#     viewer notices: a checkpoint whose motion or face sharpness falls below a
#     fraction of the base model's is rolled back even if its identity is better.
#     That is the D-9.25 lesson: a reward that scores one axis buys it with the
#     others, so the other axes have to be enforced too.
#   * Every evaluation writes its own adapter snapshot, so the best checkpoint can
#     be compared against the others offline (identity vs motion vs texture) instead
#     of trusting a single scalar.
#
# The conditioning, flow matching, LoRA setup, wall-clock guard and disk round trip
# are unchanged from v4 -- those parts are proven (D-9.24 runs 1-4).
#
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_lora_v5.py \
#       --scene data:vacecast --scene data:facemodels --scene data:v12render --wait
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

r = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "diffusers",
        "accelerate",
        "peft",
        "opencv-python-headless",
        "scipy",
        "ftfy",
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
    ],
    capture_output=True,
    text=True,
)
log(f"pip install rc={r.returncode}")
if r.returncode != 0:
    log(f"pip stderr: {r.stderr[-500:]}")
    raise SystemExit(1)

# peft's LoRA dispatcher raises when an incompatible torchao is merely importable
# (Kaggle ships 0.10.0, peft wants >= 0.16) -- D-9.24 run 1 died there.
u = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    capture_output=True,
    text=True,
)
log(f"torchao uninstall rc={u.returncode} {u.stdout.strip()[-200:]}")

# Face models come from the mounted dataset only (never raw URLs) -- exit early if absent.
MODEL_DIR = Path("/kaggle/input/datasets/wb3none/worldreel-data-facemodels")
yunet_path = next((str(p) for p in MODEL_DIR.rglob("face_detection_yunet_2023mar.onnx")), None)
sface_path = next((str(p) for p in MODEL_DIR.rglob("face_recognition_sface_2021dec.onnx")), None)
vacecast_dir = Path("/kaggle/input/datasets/wb3none/worldreel-data-vacecast")
portrait_paths = sorted(str(p) for p in vacecast_dir.rglob("C01_*.png"))
if yunet_path is None or sface_path is None:
    log(f"face models missing under {MODEL_DIR}")
    raise SystemExit(1)
if not portrait_paths:
    log(f"no C01_*.png portraits under {vacecast_dir}")
    raise SystemExit(1)

# The motion + texture source: the D-9.25 reference render (v12), published as the
# private dataset worldreel-data-v12render. Without it there is nothing to train on
# that contains motion, so fail loudly instead of silently falling back to a still.
target_video = None
v12_dir = Path("/kaggle/input/datasets/wb3none/worldreel-data-v12render")
if v12_dir.exists():
    found = sorted(str(p) for p in v12_dir.rglob("*.mp4"))
    if found:
        target_video = found[0]
if target_video is None:
    alt = sorted(str(p) for p in Path("/kaggle/input").rglob("face_consistency.mp4"))
    if alt:
        target_video = alt[0]
if target_video is None:
    log("no target render found -- mount it with --scene data:v12render")
    raise SystemExit(1)
log(f"target render: {target_video}")

args = {
    "cid": "C01",
    "width": 832,
    "height": 480,
    "frames": 49,
    "portrait": portrait_paths[0],
    "references": portrait_paths[:4],
    "yunet": yunet_path,
    "sface": sface_path,
    "out": str(OUT),
    "model_id": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "prompt": "real video footage of a woman standing still, breathing gently, blinking naturally, mouth closed, not speaking, fixed camera, head and shoulders, plain grey studio wall behind, soft even natural light, natural skin with pores, natural hair, sharp focus, shot on 35mm film, photorealistic",
    "negative": "cartoon, CGI, 3d render, plastic skin, video game, doll, painting, mannequin, blurry, deformed face, extra eyes, text, watermark, overexposed, dark blotches on skin, static, still picture, talking, open mouth, camera movement",
    "lora_rank": 16,
    "lora_alpha": 32,
    "train_steps": 80,
    # Same wall-clock guard as v4: the outer job kills the step script at 6 h and a
    # kill would skip the final validate + round trip + report.
    "max_train_seconds": 4.2 * 3600,
    "lr": 1e-4,
    "reward_every": 20,
    "rollback_tol": 0.01,
    # D-9.26: the target is the rendered video, not a jittered portrait.
    "target_video": target_video,
    "target_take_len": 48,
    "target_takes": 5,
    "target_variants": 3,
    # Gates: a checkpoint may not trade motion or texture for identity.
    "motion_floor": 0.75,
    "sharpness_floor": 0.75,
    "val_steps": 10,
    "val_indices": [8, 24, 40],
}

step = WORK / "step_lora_v5.py"
step.write_text(
    textwrap.dedent(
        """
    import gc
    import json
    import sys
    import time

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps

    a = json.load(open(sys.argv[1]))
    log = print
    W, H, N = a["width"], a["height"], a["frames"]
    OUT = a["out"]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- SFace identity: same template alignment as the render jobs ------------
    TEMPLATE = np.float32([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                            [41.5493, 92.3655], [70.7299, 92.2041]])
    det = cv2.FaceDetectorYN.create(a["yunet"], "", (W, H), 0.6)
    rec = cv2.FaceRecognizerSF.create(a["sface"], "")

    def align_face(img_bgr, lm):
        m, _ = cv2.estimateAffinePartial2D(np.float32(lm).reshape(-1, 1, 2),
                                           TEMPLATE.reshape(-1, 1, 2), method=cv2.LMEDS)
        if m is None:
            return None
        return cv2.warpAffine(img_bgr, m, (112, 112))

    def embed(img_bgr):
        det.setInputSize((img_bgr.shape[1], img_bgr.shape[0]))
        _, faces = det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        f = faces[int(np.argmax(faces[:, -1]))]
        aligned = align_face(img_bgr, f[4:14].reshape(5, 2))
        if aligned is None:
            return None
        feat = rec.feature(aligned).ravel()
        nrm = float(np.linalg.norm(feat))
        if nrm <= 0.0:
            return None
        return feat / nrm

    def face_box(img_bgr):
        det.setInputSize((img_bgr.shape[1], img_bgr.shape[0]))
        _, faces = det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        f = faces[int(np.argmax(faces[:, -1]))]
        return [int(v) for v in f[:4]]

    def sharpness(img_bgr):
        box = face_box(img_bgr)
        if box is None:
            return -1.0
        x, y, w, h = box
        roi = img_bgr[max(0, y):y + h, max(0, x):x + w]
        if roi.size == 0:
            return -1.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def motion(frames_bgr):
        if len(frames_bgr) < 2:
            return 0.0
        diffs = []
        for i in range(1, len(frames_bgr)):
            d = np.abs(frames_bgr[i - 1].astype(np.float32) - frames_bgr[i].astype(np.float32))
            diffs.append(float(d.mean()))
        return float(np.mean(diffs))

    ref_img = Image.open(a["portrait"]).convert("RGB")
    ref_bgr = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
    ref_emb = embed(ref_bgr)
    if ref_emb is None:
        raise SystemExit("no face found in reference portrait")
    log("reference face embedding norm: %.4f" % float(np.linalg.norm(ref_emb)))

    refs_imgs = [Image.open(p).convert("RGB") for p in a["references"]]
    if not refs_imgs:
        raise SystemExit("no reference images")
    log("reference images: %d" % len(refs_imgs))

    # ---- pipeline, built exactly as the render job builds it -------------------
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    model_id = a["model_id"]
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
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

    try:
        pipe.enable_xformers_memory_efficient_attention()
        log("attention: xformers enabled")
    except Exception as exc:
        log("attention: xformers unavailable (%s); using SDPA" % type(exc).__name__)
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        log("attention: math SDPA disabled (memory-efficient only)")
    except Exception as exc:
        log("attention: could not restrict SDPA (%s)" % type(exc).__name__)

    # ---- LoRA adapter: gaussian A / zero B, so step 0 == the base model ---------
    from peft import LoraConfig

    pipe.transformer.enable_gradient_checkpointing()
    lora_config = LoraConfig(r=a["lora_rank"], lora_alpha=a["lora_alpha"],
                             target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                             init_lora_weights="gaussian")
    pipe.transformer.add_adapter(lora_config, adapter_name="default")
    transformer = pipe.transformer
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in transformer.parameters())
    log("LoRA trainable params: %d / %d (%.2f%%)" % (n_train, n_total, 100.0 * n_train / n_total))

    # peft creates the adapter in the BASE layer's dtype (fp16 here), so the
    # gradients come out fp16 and GradScaler refuses them (D-9.24 run 3).
    for p in trainable:
        if p.dtype != torch.float32:
            p.data = p.data.float()
    log("adapter dtypes after upcast: %s" % sorted({str(p.dtype) for p in trainable}))
    transformer.train()

    # ---- VACE conditioning, identical to inference -----------------------------
    device = torch.device("cuda")
    seed = ImageOps.fit(ref_img, (W, H), method=Image.LANCZOS, centering=(0.5, 0.42))
    seed_rgb = np.array(seed)
    grey = Image.new("RGB", (W, H), (128, 128, 128))
    mask_black = Image.new("L", (W, H), 0)
    mask_white = Image.new("L", (W, H), 255)
    video_list = [seed] + [grey] * (N - 1)
    mask_list = [mask_black] + [mask_white] * (N - 1)
    refs_for_pipe = refs_imgs[:4]
    n_ref = len(refs_for_pipe)
    gen = torch.Generator("cpu").manual_seed(0)

    # Built directly, NOT through pipe.__call__ (which carries @torch.no_grad()):
    # without it the VAE encode records an autograd graph over every encoder
    # activation of the 49-frame conditioning video and the T4 OOMs (D-9.24 run 2).
    with torch.no_grad():
        video_t, mask_t, refs_t = pipe.preprocess_conditions(video_list, mask_list,
                                                             refs_for_pipe,
                                                             1, H, W, N,
                                                             torch.float32, device)
        cond_vl = pipe.prepare_video_latents(video_t, mask_t, refs_t, gen, device)
        mask_lat = pipe.prepare_masks(mask_t, refs_t, gen)
    cond = torch.cat([cond_vl, mask_lat], dim=1).to(torch.float16)
    n_lat = (N - 1) // 4 + 1
    if cond.shape[2] != n_lat + n_ref:
        raise SystemExit("conditioning frames %d != expected %d"
                         % (cond.shape[2], n_lat + n_ref))
    log("conditioning latents: %s" % (tuple(cond.shape),))

    def encode_norm(video_tensor):
        # VAE mean/std normalisation -- the inference decode path, inverted.
        with torch.no_grad():
            lat = pipe.vae.encode(video_tensor.to(torch.float16)).latent_dist.mode()
        mean = torch.tensor(pipe.vae.config.latents_mean, dtype=torch.float32,
                            device=lat.device).view(1, -1, 1, 1, 1)
        std = 1.0 / torch.tensor(pipe.vae.config.latents_std, dtype=torch.float32,
                                 device=lat.device).view(1, -1, 1, 1, 1)
        return ((lat.float() - mean) * std).to(torch.float16)

    # ---- D-9.26: the target is a REAL video, ranked by measured identity -------
    # The v4 target was a portrait plus micro-jitter, so the adapter learned a
    # photograph that shakes. Here every variant is [seed] + one real take of the
    # rendered character: real motion, real texture, and the takes with the best
    # measured identity become the targets (raw takes score 0.84-0.91 against the
    # portrait, the best one 0.909).
    from tools.lora_targets import build_variants, motion_diffs, read_video, split_takes

    render_frames = read_video(a["target_video"])
    log("target render: %s (%d frames)" % (a["target_video"], len(render_frames)))
    takes = split_takes(render_frames, a["target_take_len"], a["target_takes"])
    if not takes:
        raise SystemExit("target render has no complete take")
    scored = []
    for take in takes:
        ids = [embed(f) for f in take.frames]
        sims = [float(e @ ref_emb) for e in ids if e is not None]
        sharps = [sharpness(f) for f in take.frames]
        valid_s = [v for v in sharps if v >= 0.0]
        mot = float(np.mean(motion_diffs(take.frames))) if len(take.frames) > 1 else 0.0
        take.identity_mean = float(np.mean(sims)) if sims else -1.0
        take.identity_min = float(np.min(sims)) if sims else -1.0
        take.sharpness_mean = float(np.mean(valid_s)) if valid_s else -1.0
        take.motion_mean = mot
        scored.append(take)
        log("target take %d [%d:%d]: identity %.4f (min %.4f) sharpness %.1f motion %.3f"
            % (take.index, take.start, take.start + a["target_take_len"],
               take.identity_mean, take.identity_min, take.sharpness_mean, take.motion_mean))
    ranked = sorted(scored, key=lambda t: t.identity_mean, reverse=True)[: a["target_variants"]]
    log("target takes selected: %s" % [t.index for t in ranked])

    variants = build_variants(ranked, seed_rgb)
    if len(variants[0]) != N:
        raise SystemExit("target variant has %d frames, expected %d" % (len(variants[0]), N))

    encoded = []
    for k, frames_rgb in enumerate(variants):
        vid = np.stack(frames_rgb).transpose(3, 0, 1, 2)
        vt = torch.from_numpy(vid).unsqueeze(0).float().to(device) / 127.5 - 1.0
        vlat = encode_norm(vt)
        log("target variant %d: latents %s" % (k, tuple(vlat.shape)))
        encoded.append(vlat)
        del vt

    ref_lats = []
    for rt in refs_t[0]:  # preprocess_conditions returns the batch dimension
        x = rt.unsqueeze(0).unsqueeze(2)
        ref_lats.append(encode_norm(x))
    ref_stack = torch.cat(ref_lats, dim=2)
    targets = [torch.cat([ref_stack, v], dim=2) for v in encoded]
    for t_lat in targets:
        if t_lat.shape[2] != n_lat + n_ref:
            raise SystemExit("target frames %d != expected %d"
                             % (t_lat.shape[2], n_lat + n_ref))
    del video_t, mask_t, refs_t, cond_vl, mask_lat, encoded, ref_lats, ref_stack
    gc.collect()
    torch.cuda.empty_cache()

    # ---- reward / punish bookkeeping -------------------------------------------
    history = []
    best = {"reward": -1.0, "step": -1, "motion": 0.0, "sharpness": 0.0}
    best_snap = None
    baseline = None
    baseline_motion = 0.0
    baseline_sharpness = 0.0
    losses = []

    def snapshot():
        return {n: p.detach().clone() for n, p in transformer.named_parameters() if p.requires_grad}

    def restore(snap):
        with torch.no_grad():
            for n, p in transformer.named_parameters():
                if n in snap:
                    p.copy_(snap[n])

    def save_best():
        transformer.save_lora_adapter(OUT + "/lora_adapter_best", adapter_name="default")

    def save_step(step_no):
        transformer.save_lora_adapter(OUT + "/lora_adapter_step%03d" % step_no,
                                      adapter_name="default")

    def record(step, reward, sims, motion_mean, sharp_mean, event):
        entry = {"step": step, "reward": round(reward, 6), "sims": [round(s, 4) for s in sims],
                 "motion": round(motion_mean, 4), "sharpness": round(sharp_mean, 2),
                 "event": event, "best": round(best["reward"], 6), "best_step": best["step"],
                 "loss": round(losses[-1], 6) if losses else None}
        history.append(entry)
        with open(OUT + "/lora_v5_reward.json", "w", encoding="utf-8") as f:
            json.dump({"baseline": baseline, "best": best, "history": history}, f, indent=2)

    def validate(tag, export=False):
        transformer.eval()
        with torch.no_grad():
            out = pipe(video=video_list, mask=mask_list, reference_images=refs_for_pipe,
                       prompt_embeds=pos, negative_prompt_embeds=neg, height=H, width=W,
                       num_frames=N, num_inference_steps=a["val_steps"], guidance_scale=5.0,
                       conditioning_scale=1.0, output_type="pil",
                       generator=torch.Generator("cpu").manual_seed(42)).frames[0]
        transformer.train()
        sims = []
        sharps = []
        for i in a["val_indices"]:
            bgr = cv2.cvtColor(np.array(out[i]), cv2.COLOR_RGB2BGR)
            e = embed(bgr)
            sims.append(float(e @ ref_emb) if e is not None else -1.0)
            s = sharpness(bgr)
            if s >= 0.0:
                sharps.append(s)
        # Motion is measured on the GENERATED frames only: out[0] is the
        # conditioned portrait, so including it would count the conditioning jump.
        gen_bgr = [cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR) for f in out[1:]]
        mot = motion(gen_bgr)
        sharp_mean = float(np.mean(sharps)) if sharps else -1.0
        reward = float(np.mean(sims))
        log("%s reward=%.4f sims=%s motion=%.3f sharpness=%.1f"
            % (tag, reward, ["%.3f" % s for s in sims], mot, sharp_mean))
        if export:
            export_to_video(out, OUT + "/lora_v5_final.mp4", fps=16)
        return reward, sims, mot, sharp_mean

    # ---- baseline (zero-init adapter == base model) ----------------------------
    r0, s0, m0, sh0 = validate("baseline (step 0 = base model)")
    baseline = r0
    baseline_motion = m0
    baseline_sharpness = sh0
    best = {"reward": r0, "step": 0, "motion": m0, "sharpness": sh0}
    best_snap = snapshot()
    save_best()
    record(0, r0, s0, m0, sh0, "baseline")

    # ---- training with the reward loop -----------------------------------------
    optimizer = torch.optim.AdamW(trainable, lr=a["lr"], betas=(0.9, 0.999), weight_decay=0.01)
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler()
    shift = 3.0
    t_start = time.time()

    def train_one(step):
        x0 = targets[int(torch.randint(len(targets), (1,)).item())]
        noise = torch.randn_like(x0)
        u = torch.rand((), device=device)
        sigma = shift * u / (1.0 + (shift - 1.0) * u)
        x_t = ((1.0 - sigma) * x0.float() + sigma * noise.float()).to(torch.float16)
        timestep = (sigma * 1000.0).reshape(1)
        optimizer.zero_grad(set_to_none=True)
        pred = transformer(hidden_states=x_t, timestep=timestep, encoder_hidden_states=pos,
                           control_hidden_states=cond, return_dict=False)[0]
        loss = F.mse_loss(pred.float(), (noise - x0).float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach().item())

    for step in range(1, a["train_steps"] + 1):
        # Wall-clock guard: the outer job kills the step script at 6 h, and a kill
        # would skip the final validate + round trip + report.
        if time.time() - t_start > a["max_train_seconds"]:
            log("TIME BUDGET reached at step %d after %.0fs -- stopping training early, "
                "keeping the best adapter" % (step, time.time() - t_start))
            break
        loss_val = train_one(step)
        losses.append(loss_val)
        if step % 10 == 0:
            log("train step %d: loss=%.5f elapsed=%.0fs peak_gb=%.2f"
                % (step, loss_val, time.time() - t_start,
                   torch.cuda.max_memory_allocated() / 1e9))
        if step % a["reward_every"] != 0:
            continue
        prev_best = best["reward"]
        r, sims, mot, sh = validate("step %d" % step)
        save_step(step)
        # D-9.26 gates: identity may never be bought with motion or texture.
        motion_ok = mot >= a["motion_floor"] * baseline_motion
        sharp_ok = sh >= a["sharpness_floor"] * baseline_sharpness
        if not (motion_ok and sharp_ok):
            restore(best_snap)
            log("PUNISH step=%d gates failed (motion %.3f vs floor %.3f, sharpness %.1f "
                "vs floor %.1f) -> rolled back to best"
                % (step, mot, a["motion_floor"] * baseline_motion,
                   sh, a["sharpness_floor"] * baseline_sharpness))
            record(step, r, sims, mot, sh, "punish-gate")
        elif r > prev_best:
            best = {"reward": r, "step": step, "motion": mot, "sharpness": sh}
            best_snap = snapshot()
            save_best()
            log("REWARD step=%d reward=%.4f > best %.4f -> adapter snapshot saved"
                % (step, r, prev_best))
            record(step, r, sims, mot, sh, "reward")
        elif r < prev_best - a["rollback_tol"]:
            restore(best_snap)
            log("PUNISH step=%d reward=%.4f < best %.4f - tol %.3f -> rolled back to best"
                % (step, r, prev_best, a["rollback_tol"]))
            record(step, r, sims, mot, sh, "punish-rollback")
        else:
            log("PUNISH step=%d reward=%.4f (best %.4f, within tol) -> weights kept"
                % (step, r, prev_best))
            record(step, r, sims, mot, sh, "punish-keep")
        gc.collect()
        torch.cuda.empty_cache()

    # ---- final: best adapter, export, disk round trip --------------------------
    restore(best_snap)
    save_best()
    r_final, s_final, m_final, sh_final = validate("final (best adapter)", export=True)
    record(len(losses) + 1, r_final, s_final, m_final, sh_final, "final")

    rt_ok = False
    rt_reward = None
    rt_motion = None
    rt_sharpness = None
    rt_err = None
    try:
        transformer.delete_adapters("default")
        # prefix=None is required: the default prefix ("transformer") filters every
        # saved key ("blocks.*") out, loading nothing silently.
        transformer.load_lora_adapter(
            OUT + "/lora_adapter_best",
            weight_name="pytorch_lora_weights.safetensors",
            prefix=None,
            adapter_name="default",
        )
        if "default" not in getattr(transformer, "peft_config", {}):
            raise RuntimeError("adapter not registered after load")
        rt_reward, _, rt_motion, rt_sharpness = validate("roundtrip (adapter reloaded from disk)")
        rt_ok = rt_reward >= best["reward"] - 0.02
        log("ROUNDTRIP reward=%.4f vs best %.4f -> %s"
            % (rt_reward, best["reward"], "OK" if rt_ok else "MISMATCH"))
    except Exception as exc:
        rt_err = "%s: %s" % (type(exc).__name__, exc)
        log("ROUNDTRIP FAILED: %s" % rt_err)

    report = {
        "baseline": baseline,
        "baseline_motion": round(baseline_motion, 4),
        "baseline_sharpness": round(baseline_sharpness, 2),
        "best": best,
        "final_reward": r_final,
        "final_sims": s_final,
        "final_motion": round(m_final, 4),
        "final_sharpness": round(sh_final, 2),
        "improvement": round(r_final - (baseline or 0.0), 6),
        "pass_same_person": bool(r_final >= 0.363),
        "roundtrip_ok": rt_ok,
        "roundtrip_reward": rt_reward,
        "roundtrip_motion": rt_motion,
        "roundtrip_sharpness": rt_sharpness,
        "roundtrip_error": rt_err,
        "target_video": a["target_video"],
        "target_takes": [t.index for t in ranked],
        "target_identity": [round(t.identity_mean, 4) for t in ranked],
        "target_motion": [round(t.motion_mean, 4) for t in ranked],
        "target_sharpness": [round(t.sharpness_mean, 2) for t in ranked],
        "train_steps": a["train_steps"],
        "train_steps_done": len(losses),
        "reward_every": a["reward_every"],
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "trainable_params": n_train,
        "history": history,
        "seconds": round(time.time() - t_start, 1),
    }
    with open(OUT + "/lora_v5_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log("REWARD SUMMARY baseline=%.4f best=%.4f(step %d) final=%.4f improvement=%+.4f "
        "pass(0.363)=%s roundtrip=%s"
        % (baseline, best["reward"], best["step"], r_final, report["improvement"],
           report["pass_same_person"], rt_ok))
    log("AXIS SUMMARY baseline motion=%.3f sharpness=%.1f | final motion=%.3f sharpness=%.1f"
        % (baseline_motion, baseline_sharpness, m_final, sh_final))
    if not rt_ok:
        raise SystemExit(2)
    """
    ),
    encoding="utf-8",
)
(WORK / "lora_v5_args.json").write_text(json.dumps(args), encoding="utf-8")
env = {
    **os.environ,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
    # The step script imports tools.lora_targets from the uploaded code dataset;
    # without PYTHONPATH the subprocess only sees its own directory.
    "PYTHONPATH": str(CODE),
}
r = subprocess.run(
    [sys.executable, str(step), str(WORK / "lora_v5_args.json")],
    capture_output=True,
    text=True,
    timeout=6 * 3600,
    env=env,
)
text = r.stdout[-100000:]
if r.returncode != 0:
    text += "\n--- stderr ---\n" + r.stderr[-8000:]
(OUT / "lora_v5.log").write_text(text, encoding="utf-8")
log(f"lora v5 rc={r.returncode}")
if r.returncode != 0:
    raise SystemExit(r.returncode)
log("done")
