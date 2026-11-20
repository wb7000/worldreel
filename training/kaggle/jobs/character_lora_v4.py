# Job: Character LoRA v4 -- real training with an SFace identity reward/punish loop.
#
# The two earlier character-LoRA jobs never trained at all: `character_lora` (v1) died on a
# 4D/5D latent mismatch and `character_lora_v3` on `control_hidden_states=None` (the VACE
# transformer requires control latents; the training call never built them). v3 also
# normalised latents with the 0.18215 scaling factor instead of the VAE mean/std.
#
# v4 fixes all three and adds the reward system the user asked for:
#   * VACE conditioning built exactly the way inference builds it (seed frame + mask +
#     reference images through preprocess_conditions / prepare_video_latents /
#     prepare_masks), so the training forward mirrors the render forward.
#   * Flow matching with the inference scheduler's flow-shifted sigmas (shift=3.0),
#     target = noise - x0, latents normalised with the VAE latents_mean/latents_std.
#   * Target = portrait cover-crop with smooth micro-jitter (frame 0 is the exact seed).
#   * Every `reward_every` steps a validation clip is generated with the production
#     prompt/seed and scored with SFace against the reference portrait: an improvement
#     snapshots the adapter (REWARD), a regression beyond `rollback_tol` restores the
#     best weights (PUNISH), smaller drops keep the weights but are logged as punished.
#     History is written to out/lora_v4_reward.json after every evaluation so partial
#     results survive a crash.
#   * The final act reloads the saved adapter from disk and re-scores it -- the exact
#     path the render job will use -- and exits non-zero if the round trip fails.
#
#   python -m tools.kaggle_runner run training/kaggle/jobs/character_lora_v4.py --scene data:vacecast --scene data:facemodels --wait
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

# peft's LoRA module dispatcher calls is_torchao_available(), which RAISES when an
# incompatible torchao is merely importable (Kaggle ships 0.10.0, peft wants >=0.16),
# so add_adapter died before the first training step on run 1 (D-9.24). Uninstalling
# it makes the check return False and the plain nn.Linear path is used -- the state the
# local adapter dry run validated in.
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
    # Budget from v12's measured 2830 s per 49-frame/50-step segment on T4
    # (~28 s per forward): a train step is ~4 forwards with checkpointing, a
    # validate is val_steps*2 (CFG) forwards. 80 steps + 7 validates ~= 4 h,
    # well under the 6 h subprocess timeout.
    "train_steps": 80,
    "lr": 1e-4,
    "reward_every": 20,
    "rollback_tol": 0.01,
    "target_variants": 4,
    "val_steps": 10,
    "val_indices": [8, 24, 40],
}

step = WORK / "step_lora_v4.py"
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
    transformer.train()

    # ---- VACE conditioning, identical to inference -----------------------------
    device = torch.device("cuda")
    seed = ImageOps.fit(ref_img, (W, H), method=Image.LANCZOS, centering=(0.5, 0.42))
    grey = Image.new("RGB", (W, H), (128, 128, 128))
    mask_black = Image.new("L", (W, H), 0)
    mask_white = Image.new("L", (W, H), 255)
    video_list = [seed] + [grey] * (N - 1)
    mask_list = [mask_black] + [mask_white] * (N - 1)
    refs_for_pipe = refs_imgs[:4]
    n_ref = len(refs_for_pipe)
    gen = torch.Generator("cpu").manual_seed(0)

    # Built directly, NOT through pipe.__call__ (which carries @torch.no_grad()):
    # without it the VAE encodes below record an autograd graph over every
    # encoder activation of the 49-frame 832x480 conditioning video, and the T4
    # OOMs at 14.3 GB before step 1 (D-9.24 run 2; v12 never hit this because
    # all of its VAE work happens inside the no_grad __call__).
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

    def jitter_frames(base_rgb, k, n):
        # Smooth micro-jitter variants of the seed: scale/translate/brightness ramped
        # from zero, so frame 0 stays the exact seed the conditioning promises.
        rng = np.random.default_rng(1000 + k)
        scale = float(rng.uniform(0.98, 1.02))
        tx = float(rng.uniform(-6.0, 6.0))
        ty = float(rng.uniform(-6.0, 6.0))
        gain = float(rng.uniform(0.96, 1.04))
        bias = float(rng.uniform(-3.0, 3.0))
        phase = float(rng.uniform(0.0, 6.2831853))
        h, w = base_rgb.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        frames = []
        for t in range(n):
            ramp = t / max(n - 1, 1)
            s = 1.0 + (scale - 1.0) * ramp
            dx = tx * ramp + 0.6 * np.sin(2 * np.pi * t / n + phase) * ramp
            dy = ty * ramp + 0.4 * np.cos(2 * np.pi * t / n + phase) * ramp
            m = np.float32([[s, 0.0, cx * (1.0 - s) + dx], [0.0, s, cy * (1.0 - s) + dy]])
            img = cv2.warpAffine(base_rgb, m, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            g = 1.0 + (gain - 1.0) * ramp + 0.008 * np.sin(2 * np.pi * t / n + phase) * ramp
            b = bias * ramp
            frames.append(np.clip(img.astype(np.float32) * g + b, 0, 255).astype(np.uint8))
        frames[0] = base_rgb
        return frames

    base_rgb = np.array(seed)
    variants = []
    for k in range(a["target_variants"]):
        fr = jitter_frames(base_rgb, k, N)
        vid = np.stack(fr).transpose(3, 0, 1, 2)
        vt = torch.from_numpy(vid).unsqueeze(0).float().to(device) / 127.5 - 1.0
        vlat = encode_norm(vt)
        log("target variant %d: latents %s" % (k, tuple(vlat.shape)))
        variants.append(vlat)

    ref_lats = []
    for rt in refs_t[0]:  # preprocess_conditions returns the batch dimension
        x = rt.unsqueeze(0).unsqueeze(2)
        ref_lats.append(encode_norm(x))
    ref_stack = torch.cat(ref_lats, dim=2)
    targets = [torch.cat([ref_stack, v], dim=2) for v in variants]
    for t_lat in targets:
        if t_lat.shape[2] != n_lat + n_ref:
            raise SystemExit("target frames %d != expected %d"
                             % (t_lat.shape[2], n_lat + n_ref))
    del video_t, mask_t, refs_t, cond_vl, mask_lat, vlat, vt
    gc.collect()
    torch.cuda.empty_cache()

    # ---- reward / punish bookkeeping -------------------------------------------
    history = []
    best = {"reward": -1.0, "step": -1}
    best_snap = None
    baseline = None
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

    def record(step, reward, sims, event):
        entry = {"step": step, "reward": round(reward, 6), "sims": [round(s, 4) for s in sims],
                 "event": event, "best": round(best["reward"], 6), "best_step": best["step"],
                 "loss": round(losses[-1], 6) if losses else None}
        history.append(entry)
        with open(OUT + "/lora_v4_reward.json", "w", encoding="utf-8") as f:
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
        for i in a["val_indices"]:
            bgr = cv2.cvtColor(np.array(out[i]), cv2.COLOR_RGB2BGR)
            e = embed(bgr)
            sims.append(float(e @ ref_emb) if e is not None else -1.0)
        reward = float(np.mean(sims))
        log("%s reward=%.4f sims=%s" % (tag, reward, ["%.3f" % s for s in sims]))
        if export:
            export_to_video(out, OUT + "/lora_v4_final.mp4", fps=16)
        return reward, sims

    # ---- baseline (zero-init adapter == base model) ----------------------------
    r0, s0 = validate("baseline (step 0 = base model)")
    baseline = r0
    best = {"reward": r0, "step": 0}
    best_snap = snapshot()
    save_best()
    record(0, r0, s0, "baseline")

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
        loss_val = train_one(step)
        losses.append(loss_val)
        if step % 10 == 0:
            log("train step %d: loss=%.5f elapsed=%.0fs" % (step, loss_val, time.time() - t_start))
        if step % a["reward_every"] != 0:
            continue
        prev_best = best["reward"]
        r, sims = validate("step %d" % step)
        if r > prev_best:
            best = {"reward": r, "step": step}
            best_snap = snapshot()
            save_best()
            log("REWARD step=%d reward=%.4f > best %.4f -> adapter snapshot saved"
                % (step, r, prev_best))
            record(step, r, sims, "reward")
        elif r < prev_best - a["rollback_tol"]:
            restore(best_snap)
            log("PUNISH step=%d reward=%.4f < best %.4f - tol %.3f -> rolled back to best"
                % (step, r, prev_best, a["rollback_tol"]))
            record(step, r, sims, "punish-rollback")
        else:
            log("PUNISH step=%d reward=%.4f (best %.4f, within tol) -> weights kept"
                % (step, r, prev_best))
            record(step, r, sims, "punish-keep")
        gc.collect()
        torch.cuda.empty_cache()

    # ---- final: best adapter, export, disk round trip --------------------------
    restore(best_snap)
    save_best()
    r_final, s_final = validate("final (best adapter)", export=True)
    record(a["train_steps"] + 1, r_final, s_final, "final")

    rt_ok = False
    rt_reward = None
    rt_err = None
    try:
        transformer.delete_adapters("default")
        # prefix=None is required: the default prefix ("transformer") -- and ""
        # -- filter every saved key ("blocks.*") out, loading nothing silently.
        transformer.load_lora_adapter(
            OUT + "/lora_adapter_best",
            weight_name="pytorch_lora_weights.safetensors",
            prefix=None,
            adapter_name="default",
        )
        if "default" not in getattr(transformer, "peft_config", {}):
            raise RuntimeError("adapter not registered after load")
        rt_reward, _ = validate("roundtrip (adapter reloaded from disk)")
        rt_ok = rt_reward >= best["reward"] - 0.02
        log("ROUNDTRIP reward=%.4f vs best %.4f -> %s"
            % (rt_reward, best["reward"], "OK" if rt_ok else "MISMATCH"))
    except Exception as exc:
        rt_err = "%s: %s" % (type(exc).__name__, exc)
        log("ROUNDTRIP FAILED: %s" % rt_err)

    report = {
        "baseline": baseline,
        "best": best,
        "final_reward": r_final,
        "final_sims": s_final,
        "improvement": round(r_final - (baseline or 0.0), 6),
        "pass_same_person": bool(r_final >= 0.363),
        "roundtrip_ok": rt_ok,
        "roundtrip_reward": rt_reward,
        "roundtrip_error": rt_err,
        "train_steps": a["train_steps"],
        "reward_every": a["reward_every"],
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "trainable_params": n_train,
        "history": history,
        "seconds": round(time.time() - t_start, 1),
    }
    with open(OUT + "/lora_v4_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log("REWARD SUMMARY baseline=%.4f best=%.4f(step %d) final=%.4f improvement=%+.4f "
        "pass(0.363)=%s roundtrip=%s"
        % (baseline, best["reward"], best["step"], r_final, report["improvement"],
           report["pass_same_person"], rt_ok))
    if not rt_ok:
        raise SystemExit(2)
    """
    ),
    encoding="utf-8",
)
(WORK / "lora_v4_args.json").write_text(json.dumps(args), encoding="utf-8")
env = {
    **os.environ,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
}
r = subprocess.run(
    [sys.executable, str(step), str(WORK / "lora_v4_args.json")],
    capture_output=True,
    text=True,
    timeout=6 * 3600,
    env=env,
)
text = r.stdout[-100000:]
if r.returncode != 0:
    text += "\n--- stderr ---\n" + r.stderr[-8000:]
(OUT / "lora_v4.log").write_text(text, encoding="utf-8")
log(f"lora v4 rc={r.returncode}")
if r.returncode != 0:
    raise SystemExit(r.returncode)
log("done")
