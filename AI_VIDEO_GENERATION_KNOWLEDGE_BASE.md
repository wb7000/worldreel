# AI Video Generation — Knowledge Base
**Status:** Living document — updated as new research and experiments confirm/deny methods  
**Last updated:** 2026-11-10  
**Purpose:** Track what works, what doesn't, the math behind each method, and how to operate video generation AI locally and on Kaggle.

---

## 1. Neural Network Fundamentals

### 1.1 The Math

A neuron computes:
```
z = sum(w_i * x_i) + b          # weighted sum + bias
a = f(z)                        # activation function introduces non-linearity
```

Without non-linearity, stacking layers collapses to a single linear transform.

**Common activation functions:**
| Function | Formula | Range | Best for | Problem |
|----------|---------|-------|----------|---------|
| Sigmoid | 1/(1+e^-z) | (0,1) | Output (binary classification) | Vanishing gradient |
| ReLU | max(0,z) | [0,∞) | Hidden layers | Dying ReLU (dead neurons) |
| SiLU/Swish | z * sigmoid(z) | ℝ | Transformers, diffusion | Slightly slower than ReLU |
| Mish | z * tanh(softplus(z)) | ℝ | Smooth activations | Computationally heavier |
| Tanh | tanh(z) | (-1,1) | Hidden layers | Vanishing gradient |

### 1.2 Backpropagation

Goal: compute ∂L/∂w for every weight w, then update w -= lr * ∂L/∂w.

Uses the **chain rule** backwards through the computation graph:
```
∂L/∂w^(l) = (∂L/∂a^(l)) * f'(z^(l)) * x^(l-1)
```

Gradient flows from output back to input. Problems:
- **Vanishing gradient:** sigmoid/tanh derivatives < 1, repeated multiplication shrinks gradients in deep nets
- **Exploding gradient:** gradients grow exponentially (mitigated by gradient clipping)

**Optimizers:**
| Optimizer | Key Idea | Best For |
|-----------|----------|----------|
| SGD | w -= lr * gradient | Simple, stable but slow |
| Adam | Adaptive lr per parameter + momentum | Most DL training, default choice |
| AdamW | Adam + decoupled weight decay | Transformers, diffusion models |

### 1.3 Weight Initialization

- **Xavier/Glorot:** Var(w) = 2/(fan_in + fan_out) — good for tanh/sigmoid
- **He/Kaiming:** Var(w) = 2/fan_in — good for ReLU
- Poor initialization → vanishing/exploding gradients, slow convergence

---

## 2. Diffusion Models & Flow Matching (Core of Video Gen)

### 2.1 How Diffusion Works

**Forward process (noising):** Gradually add Gaussian noise to data over T timesteps:
```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
```

**Reverse process (denoising):** Neural network learns to predict noise or the clean input:
```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t)
```

**Training objective (simplified MSE):**
```
L = E[||epsilon - epsilon_theta(x_t, t)||^2]
```

**Sampling:** Start from pure noise x_T ~ N(0,I), iteratively denoise for T steps to get x_0.

### 2.2 Flow Matching (Used by Wan2.1, HunyuanVideo)

Instead of predicting noise, predict a **velocity vector field** that transports noise to data:

```
dX_t = u_theta(X_t, t) dt     # ODE, t in [0,1]
X_0 ~ N(0,I)                  # start from noise
X_1 ~ p_data                  # end at data distribution
```

Training: learn u_theta by regressing toward the conditional vector field. Advantages:
- Theoretically grounded (continuous-time)
- Stable training via ODE solvers
- Equivalent to maximum likelihood
- No need for discrete timestep schedules

### 2.3 Classifier-Free Guidance (CFG)

Boost prompt adherence by interpolating conditioned and unconditioned predictions:
```
v_final = v_uncond + guidance_scale * (v_cond - v_uncond)
```
- guidance_scale = 1.0: no guidance (pure generation)
- guidance_scale = 5.0–7.5: typical video range
- Higher = more prompt-following, less diversity, more artifacts

---

## 3. Video Generation AI Architecture

### 3.1 The Standard Pipeline (DiT + VAE + Text Encoder)

```
Text Prompt → T5/CLIP Encoder → Text Embeddings
                                    ↓
Video (pixels) → 3D Causal VAE → Latent Video (x_t)
                                    ↓
Diffusion Transformer (DiT) denoises latent video using text embeddings + timestep
                                    ↓
Latent Video → 3D Causal VAE Decoder → Output Video (pixels)
```

### 3.2 Key Components

#### 3.2.1 3D Causal VAE (Wan-VAE)

Compresses video spatially and temporally:
- Input: (1+T) × H × W × 3 (e.g., 81 × 720 × 1280 × 3)
- Output latent: (1+T/4) × H/8 × W/8 × 16
- **3D causal convolutions:** prevent temporal leakage (future frames can't influence past)
- **RMSNorm** instead of GroupNorm: enables feature caching, memory efficient inference
- Trained in 3 phases: 2D pretraining → 3D inflation on short clips → resolution-varied fine-tuning with GAN loss

**Compression ratios:**
| Model | Spatial Compression | Temporal Compression | Latent Channels |
|-------|-------------------|---------------------|-----------------|
| Wan-VAE | 8× (H/8, W/8) | 4× (T/4) | 16 |
| Stable Video Diffusion | 8× | 4× | 4 |

#### 3.2.2 Diffusion Transformer (DiT)

Replaces U-Net with transformer blocks:
- **Patch-based:** Video latent split into spatiotemporal patches
- **Cross-attention:** Text embeddings attend to video patches
- **AdaLN (Adaptive Layer Norm):** Timestep + text condition modulate each block via 6 parameters (shift/scale for attention, MLP, gate)
- **Shared MLP:** Time embedding MLP shared across all transformer blocks (Wan2.1 innovation)

#### 3.2.3 Text Encoder (T5)

- T5-XXL (multilingual) used by Wan2.1
- Cross-attention injects text into every transformer block
- Supports Chinese and English prompts

### 3.3 Major Model Families (2025–2026)

| Model | Params | Architecture | Resolution | FPS | Notes |
|-------|--------|-------------|------------|-----|-------|
| **Sora 2** (OpenAI) | Unknown | DiT + Flow Matching | Up to 1080p | 24 | Closed source, Sep 2026 |
| **Wan2.1-14B** (Alibaba) | 14B | DiT + Flow Matching + 3D VAE | 720p–1080p | 16 | Open, best open-source quality |
| **Wan2.1-1.3B** (Alibaba) | 1.3B | DiT + Flow Matching | 720p | 16 | Resource-efficient, runs on consumer GPU |
| **Wan2.2-14B** (Alibaba) | 14B | Sparse MoE DiT | 720p | 16 | MoE routes experts across timesteps |
| **HunyuanVideo** (Tencent) | 13B | DiT + Causal 3D VAE | 720p | 16 | Open, strong motion |
| **Mochi 1** (Genmo) | Unknown | Asymmetric DiT | 480p | 30 | Open, fast |
| **LTX-Video** (Lightricks) | 2B | DiT | 720p | 24 | Fast, good temporal compression |
| **SANA-Video** (MIT) | 0.6B–1.6B | Block Linear DiT | 720p | 16 | Linear attention, very efficient |
| **CogVideoX-1.5** | 5B | DiT | 720p | 16 | Good efficiency (4.4× speedup) |

### 3.4 Wan2.1 VACE (All-in-One Video Creation & Editing)

Wan2.1-VACE supports:
- Text-to-Video (T2V)
- Image-to-Video (I2V)
- Video editing (mask-based inpainting)
- First-Last Frame interpolation (FLF2V)
- Reference image conditioning (for character consistency)

**Our Kaggle job uses:** Wan2.1-VACE-1.3B-diffusers with face embedding injection + face swap post-processing.

---

## 4. Efficient Methods

### 4.1 Consistency Models (CM)

**Core idea:** Learn a function f_theta that maps any point on the probability-flow ODE trajectory directly to the endpoint (clean data), enabling **1-step or few-step generation**.

| Method | Steps | Speed | Quality | Notes |
|--------|-------|-------|---------|-------|
| Standard Diffusion | 30–50 | Slow | High | Baseline |
| Consistency Distillation | 1–4 | ~50× faster | Slightly lower | One-step student |
| sCM (Continuous-Time CM) | 1–4 | Very fast | Good | Training-free from flow models |
| SANA-Sprint | 1–4 | 5–7 samples/s | Good | Image-focused, extendable to video |
| DCM (Dual-Expert CM) | 4 | Fast | Good | Semantic expert + timestep expert |
| Greedy Distill | 4–8 | Fast | High | AR + diffusion fusion, avoids error accumulation |

**Video-specific consistency methods:**
- **CausalForcing:** 4-step generation for all frames (Wan2.1-14B teacher)
- **OneForcing:** 4 steps on first frame, 1 step on subsequent frames
- **DOLLAR:** Consistency + latent reward optimization
- **DMD2:** Distribution matching distillation (no paired data needed)

### 4.2 MoE (Mixture of Experts)

**Wan2.2 approach:** Route different diffusion timesteps to specialized expert subnetworks.
- Expanded capacity without proportional compute increase
- Each expert specializes in a subset of the denoising trajectory
- Better quality at same compute budget

### 4.3 Attention Optimizations

| Technique | What it does | Benefit |
|-----------|--------------|---------|
| Linear attention (SANA-Video) | Replace O(n²) softmax attention with linear approx | Much faster for long sequences |
| Sequence parallelism | Split sequence across GPUs | Enables longer videos on multi-GPU |
| Block linear DiT | Block-structured linear attention | Best efficiency/quality tradeoff |
| STG (Spatio-Temporal Guidance) | Guidance at attention level for temporal coherence | Better consistency, slight diversity cost |

### 4.4 Temporal Consistency Methods

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Face embedding injection** (our approach) | Inject SFace embedding as cross-attention token into DiT | Simple, works during generation | Doesn't guarantee pixel-level consistency |
| **Face swap post-processing** (our approach) | Delaunay + affine warp + seamlessClone per frame | Guaranteed identity match | Post-processing step, can look uncanny if misaligned |
| **Temporal layers in diffusion** | Insert 3D attention across frames | Built into model architecture | Requires model retraining |
| **Kalman filtering** | Use previous frame to guide current frame | Smooth transitions | Can lag behind fast motion |
| **STG guidance** | Spatio-temporal attention guidance | Improves coherence | Reduces diversity |
| **Sliding window overlap** | Overlap segments by 1–2 frames | Reduces seams | Slightly more compute |

---

## 5. Face Consistency & Face Swap

### 5.1 Face Detection (YuNet)

- **Model:** `face_detection_yunet_2023mar.onnx` (232 KB, OpenCV Zoo)
- **Input:** Exact image size required (must call `setInputSize(w, h)` before detect)
- **Output:** Bounding box (x, y, w, h) + 5 landmarks + confidence score
- **Landmark order:** Right eye, left eye, nose, right mouth corner, left mouth corner

### 5.2 Face Recognition (SFace)

- **Model:** `face_recognition_sface_2021dec.onnx` (38 MB, OpenCV Zoo)
- **Pipeline:** Detect → AlignCrop (using 5 landmarks) → Feature extraction → L2-normalize
- **Output:** 128-dim embedding, norm = 1.0
- **Similarity:** Cosine similarity = dot product of normalized embeddings
- **Threshold:** ~0.363 for "same person" (OpenCV default)

### 5.3 Face Swap Methods

| Method | Technology | Speed | Quality | Best For |
|--------|-----------|-------|---------|----------|
| **Delaunay + affine + seamlessClone** (our impl) | Landmark-based triangle warp | Fast (CPU OK) | Good | Our pipeline, real-time capable |
| **InSwapper-128** (InsightFace) | StyleGAN2 encoder-decoder + ArcFace | Real-time | Good | Roop, FaceFusion, Deep-Live-Cam |
| **InSwapper-512** | Higher-res variant | Slower | Better | Production quality stills |
| **Diffusion-based swap** | Inpainting with mask + conditioning | Very slow | Best | Quality-critical, offline |
| **FaceFusion** | InSwapper + CodeFormer + tracking | Real-time | Good | Video, live |
| **DeepFaceLab** | Manual, frame-by-frame | Very slow | Best | Cinema, deepfakes |

### 5.4 Our Face Swap Implementation (`stages/generate/face_swap.py`)

Pipeline per frame (canonical-template warp, D-9.11):
1. Detect face in reference portrait (YuNet) → 5 landmarks
2. Detect face in generated frame (YuNet) → 5 landmarks
3. Estimate a **similarity transform** (rotation + uniform scale + translation) from each face's
   landmarks to the canonical 112x112 ArcFace/SFace template
4. Warp the reference into template space
5. Warp the reference back through the **inverse** of the destination transform → it lands exactly
   on the destination's face position/scale/rotation
6. Blend inside the face-oval ellipse with a feathered alpha, plus a per-channel Reinhard tone match
7. Output face-swapped frame

**Why the old Delaunay version failed:** a triangulation of five landmarks spans only the convex
hull of eyes/nose/mouth corners, so the forehead, jaw and chin kept the *original* identity, and
the SFace embedding barely moved. The template warp covers the whole face oval.

**Measured results** (`runs/kaggle/hero_chain_run1/out/chain.mp4`, 173 frames, portrait C01_0):

| Metric | Value |
|---|---|
| PRE swap similarity to reference | 0.2434 (min 0.1799) |
| POST swap similarity to reference | **0.8462** (min 0.8005) |
| Improvement | **+0.6028** |
| Frames below 0.85 after swap | 108/173 |

### 5.5 Face alignment: never trust `FaceRecognizerSF.alignCrop`

`cv2.FaceRecognizerSF.alignCrop` is **unusable in OpenCV 5.0.0**. Measured with
`tools/face_align_probe.py`:

| Property | `alignCrop` | Template alignment (`stages/generate/face_align.py`) |
|---|---|---|
| Crop content | all-black 112x112 (20/20 calls) | real face, 96% non-zero |
| Embedding norm | 3.5 | 14.2 |
| cos(alignCrop, template) | 0.03 | 1.00 |
| Pairwise similarity across different people | **1.0000 for every pair** | 0.2472 |
| Determinism | unstable (black on most runs, misaligned on others) | deterministic |

Because a black crop makes `feature()` return the same degenerate vector for every face, **every**
cosine similarity collapses to exactly `1.0000`. The metric cannot discriminate anything — and it
fails *silently*, looking like perfect consistency.

**The fix:** implement the canonical ArcFace/SFace 5-point template alignment directly
(`estimateAffinePartial2D` + `warpAffine`). It is the geometry SFace was trained with and behaves
identically on OpenCV 4.x and 5.x.

### 5.6 Landmark order (resolved empirically)

YuNet emits `right eye, left eye, nose, right mouth corner, left mouth corner`. For a frontal face
the subject's right eye is the **smaller-x** one in image coordinates, and the canonical template
lists the eye slots in the same left-to-right order — so the mapping is the **identity
permutation**. Measured on the labelled VaceCast set:

| Order | same character | different people | separation |
|---|---|---|---|
| **YuNet order** (identity) | **0.6890** | **0.2472** | **+0.4418** |
| swapped (`[1,0,2,4,3]`) | 0.5365 | 0.4815 | +0.0549 |

The intuitive "SFace order" permutation is wrong and would have degraded every metric.

### 5.7 Blend strategy (Poisson blending hurts identity)

| Strategy | similarity to reference |
|---|---|
| **feathered alpha blend** | 0.8401 |
| feathered alpha + Reinhard tone match | **0.8411** |
| `cv2.seamlessClone` (`NORMAL_CLONE`) | 0.7963 |
| `cv2.seamlessClone` (`MIXED_CLONE`) | 0.2025 |

Poisson cloning copies the **destination's image gradient** into the result, so strong source
features (here: a moustache) get re-imprinted on top of the swapped face. For *identity transfer*
that is counterproductive. Tone matching (per-channel mean/std transfer between the masked
reference and destination regions) achieves the colour harmonisation without the gradient
imprint.

### 5.8 Face-oval geometry

Chosen by grid search on identity similarity (test frame, portrait C01_0):

| Oval (template space) | similarity |
|---|---|
| **centre (56, 54), axes (42, 51)** | **0.8401** |
| same, clipped above the hairline (y < 16) | 0.8054 |
| centre (56, 56), axes (41, 47) | 0.8246 |
| centre (56, 60), axes (40, 44) | 0.7579 |

SFace reads the forehead as part of the identity, so keeping the full forehead in the swapped
region matters more than avoiding a hairline band.

### 5.9 The meta-lesson: a metric that cannot vary is not a metric

Three separate bugs hid behind "identity = 1.0":
1. `alignCrop` returning black → all embeddings identical → all similarities 1.0000.
2. A hard-coded `sim = 1.0 if face is not None else -1.0` — "a face was still detectable", not a
   similarity.
3. A landmark correspondence that summed x/y instead of taking a Euclidean distance.

**Rule:** every identity number must be a cosine between two *independently computed* SFace
embeddings, and the alignment path must be covered by a probe that fails loudly if it regresses.
`tools/face_align_probe.py` is that probe.

---

## 6. How to Operate / Run Video Generation AI

### 6.1 Local Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU only, very slow) | NVIDIA RTX 3060+ (12GB VRAM) |
| VRAM | 4GB | 16GB+ |
| RAM | 16GB | 32GB+ |
| Storage | 50GB SSD | 500GB+ NVMe |
| Python | 3.10+ | 3.12 |

**Our local environment:** Windows 11, Python 3.12, OpenCV 5.0.0, PyTorch 2.15.0+cpu, diffusers 0.42.1 — **NO CUDA** (Intel Iris Xe only). All video generation must run on Kaggle GPU.

### 6.2 Kaggle GPU Setup

- **Quota:** 30 GPU hours/week per account
- **GPU:** 2× Tesla T4 (15GB VRAM shared)
- **Queue:** 30–90 minutes typical wait time
- **Strategy:** Push job, wait, fetch results. Use `--requeue-after 20` if queued too long.
- **Models:** Download from HuggingFace inside the job (takes ~2 min for Wan2.1 components)

### 6.3 Running Wan2.1 on Kaggle

```
# Inside Kaggle kernel
pip install diffusers opencv-python-headless scipy ftfy
from diffusers import AutoencoderKLWan, WanVACEPipeline, UniPCMultistepScheduler
import torch

vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-VACE-1.3B-diffusers", subfolder="vae", torch_dtype=torch.float16)
pipe = WanVACEPipeline.from_pretrained("Wan-AI/Wan2.1-VACE-1.3B-diffusers", vae=vae, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

# Generate
out = pipe(video=video_frames, mask=mask_frames, reference_images=[ref_img],
           prompt_embeds=pos, negative_prompt_embeds=neg,
           height=H, width=W, num_frames=N, num_inference_steps=30,
           guidance_scale=5.0, conditioning_scale=1.0,
           output_type="pil", generator=torch.Generator("cpu").manual_seed(42))
```

### 6.4 Our Pipeline Workflow

```
Step 1 (Kaggle GPU): Generate video with Wan2.1-VACE + face embedding injection
  → Output: 512x288 MP4, ~2-12s depending on segments

Step 2 (Local CPU): Face swap post-processing
  → Input: Generated MP4 + reference portrait
  → Output: Face-swapped MP4 with consistent identity

Step 3 (Local): Package as .wreel
  → Input: Face-swapped MP4 + reference portrait
  → Output: Signed .wreel with embedded video media

Step 4 (Local): Validate
  → python -m formats validate <output.wreel>
```

### 6.5 Memory Optimization for Video Gen

| Technique | Effect |
|-----------|--------|
| `enable_model_cpu_offload()` | Offload model to CPU when not in use, saves VRAM |
| `vae.enable_tiling()` | Process VAE in tiles, reduces memory for high-res |
| `text_encoder.to("cpu")` + set to None after encoding | Free text encoder VRAM before diffusion |
| `torch.cuda.empty_cache()` + `gc.collect()` between segments | Prevent memory accumulation |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduce fragmentation |
| Flow matching with fewer steps (10–20) | Faster, less VRAM peak |

---

## 7. Working vs Not Working (Tracking Matrix)

### 7.1 What WORKS (verified)

| Method | Status | Evidence |
|--------|--------|----------|
| **VACE `reference_images` (4 views), no post-hoc swap** | ✅ | S017: 135/135 frames with exactly one face, identity mean **0.8230** / min 0.7540 (D-9.14) |
| YuNet face detection on 832x480 Kaggle shots | ✅ | All 6 shot images detected, SFace norm=1.0 |
| SFace embedding extraction | ✅ | 128-dim, normalized to 1.0 |
| Face swap + similarity measurement | ✅ | Avg similarity 0.684–0.984 to reference |
| Multi-segment chaining with overlap-drop | ✅ | 49+43+43 frames, one continuous shot, no duplicated seam frames |
| `.wreel` packaging with embedded MP4 | ✅ | S011, S012 and **S017** validate OK (0 errors, 0 warnings); MP4 byte-identical after packaging |
| Kaggle job pushes and completes | ✅ | Kernel v8 pushed, COMPLETE in 2h45m, peak 7.53 GB |
| Local OpenCV + PyTorch CPU environment | ✅ | OpenCV 5.0.0, PyTorch 2.15.0+cpu installed |

### 7.2 What DOES NOT WORK / Has Issues

| Method | Problem | Severity | Workaround |
|--------|---------|----------|------------|
| YuNet on C01_*.png Kagke portraits (360x480) | No faces detected | High | Use shot*.jpg images instead (832x480, faces detected) |
| Kaggle job output quality (50KB, 2.25s) | Too low resolution/length | High | Raise to 832x480 (trained 480P) + 49 frames + 50 steps |
| Face embedding injection alone | Doesn't guarantee pixel consistency | High | Use VACE's native `reference_images` instead (D-9.13) |
| **Post-hoc face swap on reference-conditioned output** | **Pastes a second, misaligned face over the generated one** (D-9.13) | **High** | **Condition on identity; never stack two identity mechanisms** |
| `FaceConditionedTransformer` (128-d SFace token in the T5 stream) | Out-of-distribution token; redundant with `reference_images` | High | Removed in D-9.13 |
| `.wreel` format for video playback | Format is for 3D scenes, not video | Medium | Embed MP4 as media file; native video layer needs spec change |
| Local CUDA (no NVIDIA GPU) | Can't run video gen locally | High | Must use Kaggle GPU |

### 7.3 What NEEDS TESTING

| Method | Priority | Notes |
|--------|----------|-------|
| VACE `reference_images` (4 views) without any post-hoc swap | ~~High~~ **Done** | S017 verified: one face in 135/135 frames, identity 0.8230 mean (D-9.14) |
| InSwapper-128 face swap vs our Delaunay method | Medium | Compare SSIM, identity score, blend consistency |
| Consistency models (1-4 step) on Wan2.1 | Medium | Could reduce Kaggle time from 9min to 2min |
| 720p resolution on Kaggle T4 | Low | VACE-1.3B is 480P-only; 14B does not fit a T4 (§15.7) |
| Multi-segment blending (overlap frames) | Medium | Current overlap=6 frames; test 8-12 for smoother seams |
| Audio synchronization | Low | Not yet implemented |

---

## 8. Research Directions & References

### 8.1 Key Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Sohl-Dickstein et al. | 2015 | Deep Unsupervised Learning using Nonequilibrium Thermodynamics (diffusion origin) |
| Ho et al. | 2020 | Denoising Diffusion Probabilistic Models (DDPM) |
| Song & Ermon | 2019 | Generative Modeling by Estimating Gradients of the Data Distribution |
| Song et al. | 2021 | Score-Based Generative Modeling through SDEs |
| Rombach et al. | 2022 | Latent Diffusion Models (Stable Diffusion) |
| Peebles & Xie | 2023 | Scalable Diffusion Models with Transformers (DiT) |
| Liu et al. | 2024 | Video Diffusion Models |
| Song et al. | 2023 | Consistency Models (1-step generation) |
| OpenAI | 2024 | Sora (video generation as world simulator) |
| Wan Team (Alibaba) | 2025 | Wan2.1 (Flow Matching + 3D Causal VAE) |
| Yin et al. | 2025 | CausVid (bidirectional to autoregressive video distillation) |
| Chen et al. | 2025 | SANA-Video (block linear DiT for efficient video) |
| Lv et al. | 2025 | DCM (Dual-Expert Consistency Model for video) |

### 8.2 Key GitHub Repos

- **Wan2.1:** https://github.com/Wan-Video/Wan2.1
- **OpenCV Zoo:** https://github.com/opencv/opencv_zoo (YuNet + SFace models)
- **InsightFace:** https://github.com/deepinsight/insightface (InSwapper, ArcFace)
- **Roop:** https://github.com/s0md3v/roop (face swap UI)
- **FaceFusion:** https://github.com/facefusion/facefusion (advanced face swap)
- **Deep-Live-Cam:** https://github.com/opendatalab/Deep-Live-Cam (live face swap)
- **ComfyUI:** https://github.com/comfyanonymous/ComfyUI (node-based workflow)

### 8.3 Model Cards (HuggingFace)

- Wan-AI/Wan2.1-T2V-1.3B
- Wan-AI/Wan2.1-T2V-14B
- Wan-AI/Wan2.1-I2V-14B-720P
- Wan-AI/Wan2.1-VACE-1.3B-diffusers
- Wan-AI/Wan2.2-T2V-A14B
- HunyuanVideo/HunyuanVideo

---

## 9. Practical Commands

### 9.1 Check Kaggle Job Status
```bash
python -m tools.kaggle_runner status training/kaggle/jobs/character_consistency_final.py
```

### 9.2 Push Kaggle Job
```bash
python -m tools.kaggle_runner run training/kaggle/jobs/character_consistency_final.py --scene data:vacecast --scene data:facemodels --requeue-after 0
```

### 9.3 Fetch Kaggle Output
```bash
python -m tools.kaggle_runner fetch training/kaggle/jobs/character_consistency_final.py
```

### 9.4 Face Swap Video (Local)
```bash
python stages/generate/face_consistent_video.py \
    --video runs/kaggle/character_consistency_final/out/face_consistency.mp4 \
    --portrait runs/kaggle/character_consistency/out/shot0_start.jpg \
    --output training/kaggle/jobs/face_consistent_processed.mp4 \
    --max-frames 30 \
    --report training/kaggle/jobs/face_report.json
```

### 9.5 Package as .wreel (Local)
```bash
python stages/generate/package_wreel_video.py \
    --video training/kaggle/jobs/face_consistent_processed.mp4 \
    --portrait runs/kaggle/character_consistency/out/shot0_start.jpg \
    --output runs/generated/S013_face_consistent.wreel \
    --title "Face-Consistent Photorealistic Video" \
    --identity-seed 42
```

### 9.6 Full Pipeline (Process + Package)
```bash
python stages/generate/face_consistent_pipeline.py full \
    --video <input.mp4> \
    --portrait <reference.jpg> \
    --video-output <processed.mp4> \
    --wreel-output <output.wreel> \
    --max-frames 30
```

### 9.7 Validate .wreel
```bash
python -m formats validate runs/generated/S013_face_consistent.wreel
```

---

## 10. Decisions Log (for this knowledge base)

| Decision | Date | Rationale |
|----------|------|-----------|
| Use face swap post-processing + embedding injection | 2026-11-10 | Embedding alone insufficient; swap guarantees pixel-level identity |
| Use Delaunay + seamlessClone (not InSwapper) | 2026-11-10 | Runs on CPU, no extra model download, good enough quality |
| 512x288 resolution | 2026-11-10 | Kaggle T4 15GB VRAM limit; 720p may OOM |
| 4 segments × 48 frames (12s total) | 2026-11-10 | Balance quality vs Kaggle time quota |
| Embed MP4 in .wreel (not native video layer) | 2026-11-10 | .wreel spec v1.1 doesn't have video texture layer yet |
| Kagke GPU (not local) for generation | 2026-11-10 | No CUDA on local machine (Intel Iris Xe only) |

---

## 11. Open Questions / Roadmap

1. **Can we get 720p on Kaggle T4?** — Need to test VAE tiling + CPU offload with larger latents
2. **Consistency models for 1-4 step video?** — Could reduce Kaggle time from ~9min to ~2min per job
3. **Native video texture layer in .wreel?** — Requires spec version bump + player changes
4. **Audio synchronization?** — Not yet implemented; need to add WAV stem to .wreel
5. **Multi-character face consistency?** — Currently single character (C01); need multi-face tracking
6. **Longer videos (>12s)?** — Need segment chaining with overlap blending

---

---

## 12. Video Generation AI — Technical Research Guide (10 Nov 2026)

Integrated from: `VIDEO_GENERATION_AI_Research_Guide.pdf` (ByteDance Seedance research guide)

### 12.1 Executive Overview & Mental Model

A video-generation model learns a **conditional distribution over sequences of visual frames**. Given text, image, reference clip, audio, or combinations, it generates a sequence satisfying the conditions. Modern systems operate in compressed latent space with diffusion/flow-matching objectives and a transformer backbone.

**The hard part** is not generating attractive individual frames — it's preserving identity, geometry, object permanence, physical plausibility, camera intent, timing, and narrative continuity across time.

**Quality is multi-dimensional:**
| Dimension | Requirement |
|-----------|-------------|
| Spatial quality | Each frame coherent, detailed, plausible |
| Temporal quality | Motion smooth; no flicker, melt, or teleport |
| Semantic alignment | Depicts requested subjects, actions, setting, style |
| Consistency | Characters, clothing, objects, scene layout persist |
| Control | Camera movement, shot changes, timing, references obeyed |
| Efficiency | Fits latency, memory, cost budget |

**Core mental model:** Video generation = appearance + motion + time + conditioning. Better frame quality alone does not guarantee better video.

### 12.2 What a Video Model Learns

**Tensor dimensions:** T × H × W × C (time × height × width × color). At 24 fps, a 5-second clip = 120 frames. Raw pixels are too expensive → compress with 3D VAE before the main transformer.

**Hierarchy of learned concepts:**
| Level | Examples |
|-------|----------|
| Low-level | Colour, texture, edges, lighting, local motion |
| Mid-level | Object parts, pose, scene layout, camera movement, interactions |
| High-level | Actions, events, causal sequences, shot structure, style, prompt semantics |

**Why temporal learning matters:** A frame-by-frame image generator can make each frame look convincing while producing a video that flickers or changes character identity. Video models need mechanisms and training data that connect frames across time.

### 12.3 Core Mathematical Foundations

#### 12.3.1 Latent Representations

Video VAE encodes video V (T × H × W × C) into lower-dimensional latent z. Encoder reduces spatial + temporal dimensions; decoder reconstructs pixels. Generator learns to create z, not every pixel.

**3D/Spatiotemporal VAE:** Uses temporal operations + spatial operations. Causal variants prevent a frame's latent from improperly depending on future frames (helps streaming, long sequences). Compression reduces token count and compute, but excessive compression erases fine detail or motion.

#### 12.3.2 Diffusion Intuition

| Stage | What happens |
|-------|-------------|
| Forward / training | Clean latent progressively corrupted with noise at sampled noise level |
| Prediction | Network predicts noise, denoised target, velocity, or other parameterization |
| Loss | Compare prediction with known training target |
| Reverse / generation | Start from noise; iteratively sample toward clean latent |

#### 12.3.3 Flow Matching

Trains model to predict a **velocity field** transporting samples along a path between noise distribution and data distribution. At inference, numerical solver integrates learned velocity field from noise toward data-like latent. Related to diffusion but different objective.

#### 12.3.4 Noise-Prediction Objective

```
L = E[ || ε − εθ(zt, t, c) ||² ]
```

- zt = noisy latent, t = noise level, c = conditioning, ε = known noise target, εθ = model prediction
- Real systems may use velocity prediction, flow matching, weighting schemes, or other objectives

#### 12.3.5 Transformers & Attention

Transformer turns video latents into tokens/patches. Attention lets tokens exchange information:
- **Spatial attention:** connects content within a frame
- **Temporal attention:** connects information across frames
- **Cross-attention / joint multimodal blocks:** connect visual tokens to text or other modalities

#### 12.3.6 Positional Information

Network needs to know where each token belongs: frame index, vertical/horizontal position, possibly shot/modality. Methods: RoPE (Rotary Positional Embeddings), 3D positional encodings, multimodal variants. Seedance 1.0 reports interleaved multimodal positional encoding for multi-modal and multi-shot inputs.

### 12.4 End-to-End Model Architecture

| Component | Role | Key Questions |
|-----------|------|---------------|
| Video VAE encoder | Compresses video into latent tokens | Compression ratio? Temporal causality? Reconstruction fidelity? |
| Text encoder | Maps prompt to embeddings | Tokenizer? Long prompt handling? Multilingual support? |
| Conditioning adapters | Inject images, video, audio, masks | How are references aligned and preserved? |
| Diffusion Transformer | Predicts denoising/noise/velocity | Spatial/temporal attention? Scaling? Memory? |
| Sampler | Turns noise into latent sequence | Steps? Solver? Guidance? Stability? |
| VAE decoder | Reconstructs frames | Tiling? Temporal chunking? Decode speed? |
| Post-processing | Muxes frames/audio, encodes output | FPS, bitrate, colour, interpolation, safety? |

#### 12.4.1 Spatial and Temporal Design

Separate spatial and temporal attention: spatial blocks model composition/detail inside frames; temporal blocks model changes across frames. More efficient than full attention over every space-time token at every layer. Seedance 1.0 reports decoupled spatial and temporal layers. Others use joint 3D attention or mixed attention patterns.

#### 12.4.2 Text-Video Fusion

Text embeddings injected through cross-attention, adaptive normalization, or joint token processing. Must connect language concepts to visual regions and actions. CogVideoX: expert transformer with expert adaptive LayerNorm. Wan 2.1: T5 encoder + cross-attention in transformer blocks.

#### 12.4.3 Multi-Shot and Long Video

Long videos create context and memory problems. Model needs shot boundaries, shot-specific captions, temporal ordering, persistent subject identity. Multi-shot training teaches shot transitions and narrative structure. Production system may combine shot planning, per-shot generation, reference conditioning, continuity checks.

### 12.5 Dataset Engineering (Foundation of Quality)

**Pipeline:** Collect → Inspect → Segment → Sample → Caption → Validate → Pack

#### 12.5.1 Data Acquisition & Rights
- Use data you own, licensed, or legally permitted for intended training purpose
- Keep provenance: source, creator, licence, collection date, consent status, permitted uses
- Separate permission to train from permission to publish/redistribute/commercially exploit
- Remove/protect personal data; apply privacy and biometric safeguards
- Create exclusion and takedown process for disputed/prohibited material

#### 12.5.2 Filtering & Normalization
- Decode validation: reject corrupt files, broken timestamps, unsupported codecs
- Quality filtering: resolution, sharpness, exposure, compression artefacts, motion blur
- Content filtering: safety, privacy, sensitive content, policy compliance
- Duplicate detection: exact hashes + perceptual/embedding-based near-duplicate detection
- Aspect-ratio bucketing: avoid damaging crops, reduce padding waste
- Frame sampling: choose target FPS or temporal sampling while preserving important actions
- Shot segmentation: split clips at scene cuts or major visual discontinuities
- Metadata: retain duration, dimensions, FPS, source, licence, preprocessing history

#### 12.5.3 Captioning (Static + Dynamic Facts)

**Weak caption:** "A car in a city."
**Useful dense caption:** "A red compact car drives from left to right through a rain-soaked city street at night; reflections shimmer on the road; the camera tracks alongside the car; pedestrians remain in the background."

Caption should describe: subject, appearance, action, environment, camera, temporal changes, style — without inventing facts. Seedance's report describes captions combining static info (subjects, objects, scenes) with dynamic info (actions, camera movements). CogVideoX also reports video-captioning pipeline for better text-video alignment.

#### 12.5.4 Dataset Quality Checks
- Human spot-checks on random samples and difficult edge cases
- Caption-video alignment checks (human review + vision-language models)
- Distribution reports by action, scene, camera movement, duration, style, aspect ratio
- Leakage checks between training/validation/test sets
- Reproducible manifests for traceability to exact data versions

### 12.6 Training Objectives & Optimization

#### 12.6.1 Pretraining
Exposes large model to broad video-text distribution. Learns general visual concepts, motion patterns, temporal relationships, language-to-video alignment. Requires large datasets, accelerator time, distributed training.

#### 12.6.2 Progressive Training
Begin with manageable frame counts and resolutions, then increase sequence length and resolution. CogVideoX describes progressive training + multi-resolution frame packing. Controls memory; lets model learn broad structure before harder high-resolution cases.

#### 12.6.3 Optimization Loop
```
1. Sample video clip + text/reference conditioning
2. Encode clip with video VAE
3. Sample noise level and noise (or flow-matching path position)
4. Corrupt/interpolate latent according to chosen objective
5. Run transformer with conditioning
6. Compute weighted loss against training target
7. Backpropagate gradients, update parameters
8. Log loss, gradient norms, throughput, memory, validation samples
9. Save checkpoints, run fixed validation prompts at intervals
```

#### 12.6.4 Training Stability & Scale
- Mixed precision (BF16/FP16) with numerically safe operations
- Gradient checkpointing: trades compute for lower activation memory
- Gradient accumulation: increases effective batch size without huge batch
- Distributed data parallelism or sharded strategies across GPUs
- Learning-rate warmup + tested scheduler; monitor instability and overfitting
- Save checkpoints, optimizer state, configuration, dataset manifests
- **Training loss alone does not prove video quality — inspect validation generations**

#### 12.6.5 Fine-tuning & LoRA
Fine-tuning adapts pretrained model to domain, style, subject, task. LoRA adds low-rank trainable matrices to selected layers while keeping most base weights frozen. More accessible than foundation pretraining, but does not create general-purpose model at Seedance scale.

### 12.7 Conditioning: Text, Image, Video, Audio

| Type | Description | Key Challenge |
|------|-------------|---------------|
| Text-to-video | Prompt encoded to guide generation | Prompt following depends on caption quality, text encoder, multimodal fusion, training diversity, post-training |
| Image-to-video | Image supplies appearance, composition, identity, initial scene | Infer plausible motion while preserving reference |
| Video-to-video / reference control | Reference video guides motion, camera, style, subject | Encode reference frames; decide which properties transfer; distinguish desired motion from background |
| Audio + joint audiovisual | Generated after, parallel, or jointly with video tokens | Align speech, sound effects, music, visible events |
| Multi-task | Single model trained on mixed task types | Balance so one task doesn't dominate |

### 12.8 Post-Training: Quality, Preferences, Consistency

| Method | Purpose | Notes |
|--------|---------|-------|
| SFT (Supervised Fine-Tuning) | Curated higher-quality prompt-video pairs | Improves prompt adherence, composition, motion, aesthetics, difficult scenarios |
| Human preference + reward models | Human evaluators compare outputs across dimensions | Must account for temporal failures single-frame image reward would miss |
| RLHF / preference optimization | Improve outputs per learned reward signals | Not magic: reward models can be exploited; labels inconsistent; optimizing one metric can harm diversity |
| Distillation | Faster student imitates slower teacher/trajectory | Consistency + adversarial approaches reduce sampling steps |

#### 12.8.5 Consistency Techniques
- Reference-image conditioning + subject embeddings
- Temporal attention + memory across frames
- Identity/appearance consistency objectives
- Shot-level planning + shared character references
- Optical-flow or motion-aware losses where appropriate
- Temporal discriminators or reward models
- Generate-and-rerank pipelines with continuity checks

### 12.9 Inference, Acceleration & Serving

#### 12.9.1 Inference Pipeline
1. Parse prompt + validate inputs
2. Encode text + any image/video/audio references
3. Create initial latent noise + apply masks/conditions
4. Run iterative sampler steps through transformer
5. Decode latent chunks into frames using VAE
6. Optionally upscale, interpolate, stabilize, colour process
7. Encode video/audio streams, mux to MP4/other format
8. Run safety checks, return output with metadata

#### 12.9.2 Speed & Memory Methods
- Lower resolution/fewer frames for preview generations
- Latent-space generation instead of pixel-space
- Fewer sampling steps (distilled models, faster solvers)
- Quantization (FP8 or lower-bit where supported)
- CPU offloading, layer-wise loading when VRAM constrained
- VAE slicing/tiling, chunked decode
- Memory-efficient attention, optimized CUDA kernels
- `torch.compile` after correctness testing
- Batching + queue scheduling for GPU utilization

#### 12.9.3 Production Serving Architecture
```
User/API → Job Queue (Redis/scheduler) → GPU Worker (model + sampler) → Object Storage (video files) → Database (status + metadata)
```
Real service separates user-facing API from GPU workers. Jobs enter queue, workers claim them, model processes, object storage holds videos, database tracks status/prompt/settings/seed/model version/output URL. Use retries, cancellation, quotas, observability, cost controls.

### 12.10 Evaluation & Failure Analysis

Video quality cannot be reduced to one scalar score. **VBench** defines 16 dimensions including subject consistency, motion smoothness, temporal flickering, spatial relationships. Human evaluation remains essential — automatic metrics don't fully match human judgement.

| Dimension | Example Question |
|-----------|-----------------|
| Prompt alignment | Did output depict requested subjects and actions? |
| Subject consistency | Does same character remain recognizable across frames? |
| Motion smoothness | Are movements continuous and plausible? |
| Temporal flicker | Do textures, shapes, lighting jump unexpectedly? |
| Spatial relationships | Do objects maintain sensible positions and interactions? |
| Camera control | Did camera follow requested movement? |
| Aesthetic quality | Is result visually coherent and appealing? |

#### 12.10.2 Failure Modes

| Failure | Possible Causes | Potential Remedies |
|---------|----------------|-------------------|
| Flicker | Weak temporal modelling; inconsistent conditioning | More temporal context, temporal losses, better data |
| Identity drift | Insufficient subject persistence/references | Identity conditioning, reference frames, consistency data |
| Bad physics | Sparse action examples; weak motion supervision | Curated action data, motion-aware evaluation |
| Prompt misses details | Caption mismatch; weak text fusion | Better captions, prompt suite, conditioning improvements |
| Blurry output | VAE bottleneck, low-res training, undercapacity | Improve VAE/data, staged resolution, model capacity |
| Long video breaks | Context limits, shot transitions, accumulated errors | Shot planning, memory, chunking with continuity references |

#### 12.10.3 Practical Evaluation Protocol
- Fixed prompt suite: people, animals, vehicles, nature, camera moves, interactions, physics, text rendering, multi-shot narratives
- Fixed seeds/settings when comparing checkpoints
- Score separately: prompt adherence, identity, motion smoothness, flicker, geometry, physics, camera control, aesthetics, diversity
- Evaluate short AND longer clips (failures grow with duration)
- Blind human pairwise comparisons with clear rubrics
- Track failure categories; curate targeted examples
- Report multiple samples; don't overinterpret one clip

### 12.11 Hardware & Scaling

Video tokens multiply quickly with frame count, resolution, temporal context. Model weights are only part of memory: activations, optimizer states, gradients, attention buffers, VAE operations, data pipelines also consume resources.

**Sensible compute progression:**
1. **Local CPU/modest machine:** learn concepts, preprocess small clips, write PyTorch experiments, train tiny toy networks, evaluation scripts, remote GPU services
2. **One rented GPU:** run small-scale inference, fine-tune adapter; select VRAM based on exact model/settings
3. **Multi-GPU node:** train larger models, heavier experiments with distributed strategies
4. **GPU cluster:** foundation-model pretraining, high-throughput data pipelines, long experiments

**Don't buy hardware based on model name alone** — check exact checkpoint, precision, resolution, frame count, attention implementation, VAE memory, documented minimum VRAM. Inference ≠ training requirements.

### 12.12 Practical Research Roadmap

**Stage 0 — Foundations:** Python, NumPy, PyTorch tensors/autograd; CNNs, transformers, attention, positional encodings; probability, Gaussian noise, optimization, gradient descent; image diffusion, latent autoencoders; video tensor handling, FFmpeg, OpenCV, dataset metadata.

**Stage 1 — Reproduce small image diffusion experiment:** Train small image model on permitted, narrow dataset. Learn forward corruption, timestep embeddings, denoising network, loss, sampler, checkpointing, validation. Isolate diffusion mechanics before temporal complexity.

**Stage 2 — Build toy video model:** Very low resolution, short clips. Small spatiotemporal U-Net or compact transformer. Educational experiment, not competitive generator. Track whether it learns motion vs memorizing colour palettes.

**Stage 3 — Fine-tune pretrained video model:** Supported open model, small licensed coherent dataset. Start with LoRA or parameter-efficient method. Fixed validation prompt set; compare before/after. Read model card + license before commercial/public use.

**Stage 4 — Modify architecture carefully:** Test spatial-only vs temporal attention at selected layers; compare temporal convolution vs temporal attention vs factorized attention; test frame sampling and sequence length; study VAE reconstruction quality before blaming transformer; ablate caption quality, reference conditioning, loss terms one at a time.

**Stage 5 — Research-grade model work:** Only after pipeline reproducible: large-scale pretraining, custom VAE training, multi-task training, preference optimization, distillation, distributed infrastructure.

**Suggested first project:** 16-frame motion learner — compact educational model predicting 16-frame 64×64 clip from narrow licensed dataset. Train without text first (learn video structure), then add small text encoder + conditioning. Evaluate motion smoothness and consistency. Keep expectations modest.

#### 12.12.1 Research Log Template
- **Hypothesis:** what single change are you testing?
- **Dataset version and licences:** which exact clips?
- **Model:** architecture, parameter count, pretrained components
- **Training:** steps, batch size, optimizer, learning rate, precision, GPU, wall time
- **Validation:** fixed prompts, seeds, samples, metric versions
- **Results:** examples, failure categories, quantitative metrics
- **Decision:** keep, revert, or run another controlled experiment

### 12.13 Open Models & Codebases to Study

| Project | What to Study | Notes |
|---------|--------------|-------|
| Wan 2.1 | 3D causal VAE, Flow Matching DiT, T5 conditioning, inference | Open implementation; multiple sizes/tasks. Check memory + licence |
| CogVideoX | Expert transformer, 3D causal VAE, progressive training, captioning | ICLR 2025 paper + open implementation; Diffusers has LoRA training docs |
| HunyuanVideo | Large-scale video generation + inference stack | Official Tencent repo: code, weights, model resources; check current licence |
| Hugging Face Diffusers | Pipelines, adapters, memory optimization, training scripts | Practical framework; APIs evolve |
| finetrainers | Memory-optimized fine-tuning workflows | Work in progress; pin versions |

**Suggested reading order:** CogVideoX paper (architecture + progressive training) → Wan 2.1 repo/report (practical open implementation) → Seedance 1.0 report (proprietary system public description) → VBench (evaluation)

### 12.14 Lessons from Reddit & Developer Communities

Reddit posts are anecdotal — not peer-reviewed evidence, not proof a recipe generalizes.

**Key takeaways from community reports:**
1. Start with narrow, measurable experiment rather than huge mixed dataset
2. Inspect samples and captions; noisy labels can undermine training
3. Keep validation prompts and seeds fixed to detect real progress
4. Small toy model is a learning instrument, not a substitute for foundation model
5. Reproduce results and inspect code before treating Reddit claim as best practice

**Notable community posts:**
- r/learnmachinelearning: ~20.8M-param spatiotemporal U-Net with flow matching, temporal convs, temporal attention (16-frame 64×64, 6,000 clips, batch 4, single gaming GPU). Plateauing loss, blurry results. Lesson: small dataset, limited capacity, low resolution make visual progress difficult. Falling loss ≠ perceptual quality.
- r/StableDiffusion: Klippbok open video dataset curation toolkit (scan → triage → caption → extract → validate → organize)
- r/MachineLearning: fine-tuning video diffusion models — VRAM limits, batch size, caption quality, clip-length sampling, adding temporal layers to image models

### 12.15 Safety, Rights & Governance
- Copyright/licence: establish rights to source clips; check code and weight licences separately
- Consent/likeness: consent-based data for identifiable people, faces, voices, biometric material; provide takedown processes
- Privacy: avoid unnecessary personal data; secure raw videos, captions, embeddings
- Misuse: safeguards against impersonation, non-consensual intimate imagery, fraud, deceptive media
- Provenance: retain generation metadata; consider watermarking or content credentials
- Evaluation bias: test across diverse subjects, environments, languages, camera styles
- Security: validate uploads, isolate media decoding, scan files, restrict worker permissions
- Transparency: label synthetic media where context/law requires; don't present generated footage as authentic evidence

### 12.16 Key References
| Reference | URL | Description |
|-----------|-----|-------------|
| Seedance 1.0 technical report | https://arxiv.org/abs/2506.09113 | Data curation, captioning, spatial/temporal layers, multimodal positional encoding, multi-task training, SFT, video-specific RLHF, distillation |
| Seedance 1.0 project page | https://seed.bytedance.com/en/seedance/ | Official product overview; vendor-reported performance claims |
| Wan 2.1 GitHub | https://github.com/Wan-Video/Wan2.1 | Open implementation: video VAE, Flow Matching DiT, T5 conditioning, model variants |
| CogVideoX paper (ICLR 2025) | https://proceedings.iclr.cc/paper_files/paper/2025/hash/ce31378e9f41d8907e97dab172b6c559-Abstract-Conference.html | Expert transformer, 3D VAE, progressive training, multi-resolution frame packing |
| Hugging Face Diffusers | https://huggingface.co/docs/diffusers/main/en/index | Pipelines, adapters, optimization, training, serving |
| HunyuanVideo GitHub | https://github.com/Tencent-Hunyuan/HunyuanVideo | Open model implementation, weights, resources |
| VBench (CVPR 2024) | https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html | Evaluation dimensions and benchmark methodology |

**Research snapshot: 10 November 2026.** Verify model releases, hardware requirements, repository instructions, licensing terms, and APIs at time of use.

---

## 13. Video Analyzer & Annotator Tools

### 13.1 Video Analyzer (`stages/generate/video_analyzer.py`)

Detects errors invisible to the human eye:

| Metric | Method | Threshold | Detects |
|--------|--------|-----------|---------|
| PSNR | Pixel MSE vs reference | N/A | Pixel-level degradation |
| SSIM | Structural similarity vs reference | N/A | Structural degradation |
| Laplacian variance | Blur/sharpness per frame | <50 = blur | Excessive blur |
| Optical flow (Farneback) | Motion magnitude/direction between frames | >30% high-motion frac | Temporal jitter, unnatural motion |
| Frame difference | Per-pixel abs change between consecutive frames | >50% changed pixels | Temporal jitter |
| Luminance flicker | Temporal brightness std/mean | >0.05 = flicker | Luminance flicker |
| Blocking artifacts | DCT 8x8 block boundary differences | >5.0 = blocking | Compression blocking |
| Ringing artifacts | Laplacian magnitude near edges | Variable | Edge ringing |
| Face consistency | SFace embedding cosine similarity between frames | <0.85 = drift | Face identity drift |
| Face detection rate | YuNet face detection per frame | <80% = failure | Face detection failures |
| Face region blur | Laplacian variance within face bbox | Variable | Face-specific blur |

**Issue auto-detection:** Flags temporal jitter, face drift, blur, blocking, flicker with severity classification (none/low/medium/high).

### 13.2 Video Annotator (`stages/generate/video_annotator.py`)

Creates annotated MP4 with visual overlays:
- **Red/orange/green boxes** on face regions (color-coded by blur level)
- **Face landmarks** (5 dots: eyes, nose, mouth corners)
- **Optical flow arrows** (green=low motion, red=high motion)
- **Frame-difference heatmap** (COLORMAP_JET overlay, 30% blend)
- **Per-frame text overlay:** motion, frame_diff, face_blur, face_sim
- **Frame index counter**

### 13.3 Usage Examples

```bash
# Analyze video for invisible errors
python stages/generate/video_analyzer.py \
    --video input.mp4 \
    --portrait reference.jpg \
    --output report.json \
    --sample-every 5 \
    --max-frames 30

# Create annotated visualization
python stages/generate/video_annotator.py \
    --video input.mp4 \
    --portrait reference.jpg \
    --output annotated.mp4 \
    --sample-every 5

# VMAF perceptual scoring (ffmpeg required)
ffmpeg -i reference.mp4 -i encoded.mp4 -lavfi "libvmaf=log_path=vmaf.json" -f null -
```

**VMAF gotcha on Windows:** `libvmaf` eats backslashes in `log_path`, so
`log_path=runs\analysis\x.json` is written as `runsanalysis\x.json` in the CWD. Use forward slashes.

### 13.4 The ghost-face QA tool (`tools/v2_check.py`)

An overlapping pasted face is easy to miss by eye and easy to *claim* fixed. `tools/v2_check.py`
makes it a number:

```bash
python tools/v2_check.py [video.mp4]     # defaults to the latest v2 run
```

- asks YuNet for **every** detection rather than only the most confident one, and reports
  `frames with > 1 face` — a pasted second face lands here
- inter-ocular distance and face-height spread (CV): two face populations ⇒ bimodal jitter
- identity cosine per sampled frame against the reference portrait (threshold 0.363)
- Laplacian variance inside the face oval (sharpness)

`tools/vmaf_read.py` parses libvmaf's XML log (the file is named `.json` but is XML).

### 13.5 Test Results (on `test_face_consistent.mp4`)

```
6 frames analyzed (every 5th of 30)
Face detection: 100% rate, 0.995 consecutive similarity (excellent)
Flicker index: 0.029 (under 0.05 — no flicker)
Blocking: mean 2.0 (under 5.0 — no blocking)
Issue: excessive_blur — 2 frames below blur threshold
Severity: MEDIUM
```

---

## 14. Kaggle Output-Path Trap (the silent failure mode)

**The rule: if an artefact is not inside `/kaggle/working/out`, it does not exist.**

### 14.1 What actually happens

Kaggle archives **only** `/kaggle/working/out` when it saves a script kernel's output.
`kaggle kernels output` then unpacks that archive, so a downloaded `out/` folder contains
exactly what was in `/kaggle/working/out` — nothing else, no matter where else the job wrote.

The failure is silent because:

1. The job genuinely succeeds (`rc=0`) — generation worked fine.
2. The job's own `log()` writes to `/kaggle/working/out/log.txt`, so **something** always
   arrives. The folder is never truly empty, so it looks like a partial download.
3. Kaggle serves a **stale cached output** for re-pushed versions. An old 50 KB MP4 can
   reappear for several fetches, masquerading as a fresh successful run.

### 14.2 The bug that caused it (WorldReel, 13 Nov 2026)

```python
OUT_DIR = Path("/kaggle/working/out")  # line 33  — correct
...
OUT_DIR = Path(os.environ.get("OUT", "/tmp/kaggle_out"))  # line 117 — SILENTLY OVERWRITES
```

The bootstrap defined `OUT` as a **Python variable**, not an environment variable, so
`os.environ.get("OUT", ...)` always fell through to `/tmp/kaggle_out`. The MP4, the JSON
report and the job log all landed there. Kaggle archived nothing but `log.txt`.

**Symptom:** `rc=0`, "Job completed successfully!", downloaded folder contains only `log.txt`.

### 14.3 Prevention (three layers, all implemented)

| Layer | Mechanism | Catches |
|-------|-----------|---------|
| Bootstrap | Exports `OUT`, `WORK`, `CODE` as **environment variables** | Job + subprocess can never disagree |
| Job | Exactly **one** `OUT_DIR` assignment, defaulting to `/kaggle/working/out` | No shadowing |
| Job end | Recovery sweep from `/tmp/kaggle_out` + **hard existence check** on the MP4 | Exits non-zero instead of silently passing |

### 14.4 The CPU smoke test

`training/kaggle/jobs/output_paths_test.py` — a **CPU-only** Kaggle kernel that guards the
plumbing without spending GPU quota. It asserts 10 conditions:

```
bootstrap exports OUT                PASS  OUT='/kaggle/working/out'
OUT is inside /kaggle/working/out    PASS  resolved OUT_DIR=/kaggle/working/out
no stray /tmp/kaggle_out in use      PASS  OUT_DIR=/kaggle/working/out
YuNet model mounted                  PASS  /kaggle/input/datasets/.../face_detection_yunet_2023mar.onnx
SFace model mounted                  PASS  /kaggle/input/datasets/.../face_recognition_sface_2021dec.onnx
portrait mounted                     PASS  /kaggle/input/datasets/.../C01_0.png
child process writes into OUT_DIR    PASS  rc=0 stdout='/kaggle/working/out/output_paths_test_child.txt'
marker readable after write          PASS  output_paths_test_marker.txt (27 bytes)
MP4 produced in OUT_DIR              PASS  cv2 mp4v 1892 bytes
no artefacts stranded in /tmp/...     PASS  0 stray file(s)
output path test: 10/10 checks passed
```

Run it with the new `--cpu` flag (zero GPU quota):

```bash
python -m tools.kaggle_runner run training/kaggle/jobs/output_paths_test.py \
    --scene data:facemodels --scene data:vacecast --cpu --requeue-after 30 --wait
```

### 14.5 General lessons for cloud GPU pipelines

- **Never trust `rc=0` alone.** Verify the *artifact*, not the exit code.
- **Cheap tests before expensive tests.** A plumbing bug should be caught by a 70-second CPU
  kernel, not an 19-minute T4 run. The `--cpu` flag exists for exactly this.
- **Stale-cache detection:** identical output size across re-pushes (e.g. exactly 50,641
  bytes, 2.25 s) means you are looking at a cached version, not your new code.
- **Write a sentinel.** The job now refuses to report success unless `face_consistency.mp4`
  exists inside the archived directory.
- **OpenCV version drift:** OpenCV 5 renamed `cv2.VideoWriter_fourcc` to
  `cv2.VideoWriter.fourcc`. Use `getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc`
  so the same code runs on Kaggle (OpenCV 4.14) and locally (OpenCV 5.0).
- **OpenCV version drift, part 2:** `FaceRecognizerSF.alignCrop` is broken in OpenCV 5.0.0
  (§5.5). Never call it; align with the canonical template instead.

---

## 15. Video Diffusion Transformers — the math, end to end

### 15.1 One-sentence definition

A **video diffusion transformer** generates video by learning to iteratively denoise a tensor of
latent frames, using a **transformer** (not a U-Net) as the denoiser.

### 15.2 The pipeline (Wan2.1 / VACE, the model used here)

1. **Encode to latent space.** A causal 3D VAE compresses `B×3×T×H×W` → `B×16×t×h×w`
   (8× spatial, 4× temporal). This is what makes the computation tractable: a 512×288×135 clip
   becomes a latent grid small enough to attend over.
2. **Patchify.** The latent is cut into 3D patches (Wan uses 2×2 spatial, 2 temporal). Each patch
   becomes one **token**. The video is now just a *sequence* — that is the "transformer" part.
3. **Denoise with a transformer.** Each block does roughly:
   - **self-attention** over all spatiotemporal tokens → this is what couples motion across time
     (every frame can attend to every other frame),
   - **cross-attention** to text embeddings (UMT5) so the prompt steers content,
   - **AdaLN modulation** from the timestep `t` and conditioning → this is how the model knows
     "how noisy am I right now".
4. **The objective.** Wan2.1 uses **flow matching** (rectified flow) rather than classic
   ε-prediction: the model learns a velocity field `v` pointing from pure noise `x₀` toward data
   `x₁` along a straight interpolation `xₜ = (1−t)x₀ + t·x₁`. Straight paths → fewer sampling
   steps (classifier-free guidance + ~a dozen solver steps).
5. **Decode.** The denoised latent goes back through the VAE to RGB pixels.

### 15.3 Why a transformer instead of a U-Net?

U-Nets bake in locality and down/upsampling; transformers scale better with data and compute,
handle long-range temporal dependencies more naturally, and the architecture is a clean drop-in
from image DiTs. The cost is that attention is `O(n²)` in token count — which is why **frame count
and resolution are the knobs that blow up memory**, and why trimming frames is the first lever
when a T4 runs out of VRAM.

### 15.4 Why this matters for face consistency

Identity is a *global, high-frequency* attribute that must survive both the VAE's lossy compression
and the model's temporal attention. Nothing in the objective says "this face is the same person
across 135 frames" — the model only minimises denoising error, so identity drift is a natural
failure mode.

There are two families of fix, and they must not be stacked (D-9.13):

1. **Condition the model on the identity** — VACE's native `reference_images` encodes each
   reference into the latent sequence and prepends it to the video tokens, so identity becomes part
   of the denoising problem itself. This is the right lever.
2. **Guarantee it afterwards** — a landmark-warp face swap. It is a hard pixel-space guarantee, but
   it only works when the destination face's geometry actually matches the reference's; otherwise it
   pastes a second face over the first.

The transformer is a denoiser over a video-shaped token sequence; the diffusion objective is how
it is trained to turn noise into that sequence.

### 15.5 Identity injection: what does and does not work

**Works — `reference_images` (VACE's native reference-to-video).** `WanVACEPipeline.__call__` takes
`reference_images: list[PIL.Image]`; `preprocess_conditions` builds a zeros canvas with an all-ones
mask when `video is None`, and `prepare_video_latents` encodes every reference and **prepends it to
the latent sequence**. So reference-to-video needs no control video at all — this is the supported
path, and the official CLI exposes it as `--src_ref_images`. Several views of *one* identity (we
pass four `C01_*` portraits) pin the character far harder than a single image.

**Does not work — the `FaceConditionedTransformer` wrapper (removed, D-9.13).** A zero-initialised
linear projector mapped the 128-d SFace embedding into the transformer's hidden size and
concatenated the resulting token onto `encoder_hidden_states`. Two independent reasons it failed:

- The token is **out-of-distribution**: the model was trained only on T5 token embeddings, and one
  extra vector scaled x10 does not read as "this person", it reads as noise in the text stream.
- It was **redundant** — `reference_images` was already active, so the injection added instability
  on top of an identity signal the model already had.

A soft hint inside the text stream also cannot be a "hard guarantee": §15.4's guarantee lives either
in the conditioning or in the pixels, not in a third place.

### 15.6 The overlapping-face artifact (why the swap was removed)

The v2 job passed `reference_images` **and** ran the face swap on every frame, producing
`runs/kaggle/character_consistency_v2/out/face_consistency.mp4` with a visible second face pasted
over the generated one while reporting `post_swap_sim` 0.93. Measured with `tools/_swap_debug.py` on
frame 45:

| Quantity | Destination (generated) | Reference |
|---|---|---|
| Inter-ocular / face-height ratio | 0.380 | 0.322 |
| Face bbox height | 129 px | 246 px |
| Nose-tip vs eye-midpoint X (yaw proxy) | -2.5 px | -0.5 px |

The swap aligns both faces to the canonical 112x112 template and warps back through a
**similarity transform** — rotation, scale and translation only. It cannot change face *shape*, so
the face-oval mask (`face_oval_mask`, a plain ellipse) becomes a mask-shaped patch of a longer,
sterner, differently-expressed face sitting on a rounder, softer one. The original jaw remains
visible around the edge. That is the artifact.

**Lesson:** a post-hoc swap is only safe when the generated face already has the reference's
geometry. With native reference conditioning producing `pre_swap_sim` 0.797, the swap had nothing to
gain and everything to lose. Identity is now measured on the **un-swapped** output, so the reported
`identity_sim` reflects what the model itself produced.

**Verified (S017, D-9.14):** with the swap removed, 135/135 frames contain exactly one face,
identity cosine mean **0.8230** / min **0.7540** against the 0.363 threshold, subject sharpness
641 (up from 375), and the analyzer reports 0 issues at severity NONE. Removing the second identity
mechanism improved the numbers rather than costing any.

### 15.7 Hardware ceiling: which model actually fits

- **No 500 B video model exists.** Open video models top out around 14 B dense params.
- **Wan2.1-VACE-14B**: 14.3 B params, ~28.6 GB fp16 weights, 75 GB repo. Kaggle's 2x Tesla T4 gives
  30.7 GB usable across both cards, and ~29 GB CPU RAM — so 14 B does not fit *with* activations and
  the text encoder. It also does not fit in CPU RAM for offloading.
- **Wan2.1-VACE-1.3B** is therefore the practical ceiling: ~8.19 GB for 81 frames at 480P, and it
  supports **480P only** (832x480). 14B adds 720P.

**Measured on the T4 (S017, D-9.14):** 832x480 x 49 frames x 50 steps with 4 reference images peaks
at **7.53 GB** — roughly half of the 15.36 GB usable. The OOM ladder never fired, which means the
attention-backend work (xformers / memory-efficient SDPA with the math kernel disabled) is what
keeps it there; without it the materialised `seq x seq` math-attention matrix is the thing that
overflows, not the weights.

Running at 832x480 (the trained 480P resolution) instead of 512x288 matters as much as the model:
generating far below the training resolution is a large part of why earlier output looked soft.

### 15.8 Chain drift: why a multi-segment video loses its face

Chaining segments (segment N conditioned on segment N-1's tail) buys continuity but costs identity,
and the cost **compounds**. Measured on S017 with `tools/identity_curve.py`:

| Segment | Conditioning head | Interior identity slope |
|---|---|---|
| 0 | reference portrait | **-0.00037**/frame |
| 1 | previous tail | **-0.00209**/frame (5.6x) |
| 2 | previous tail | **-0.00176**/frame (4.8x) |

The handoff itself is fine (steps of -0.023 and +0.010 at the cuts). What hurts is that each
segment inherits an already-degraded anchor and degrades it again: 0.93 -> 0.85 -> 0.82 -> 0.77 ->
0.76 -> 0.71.

**The fix is to keep the reference in the conditioning of every segment**, not just the first: put
the cover-cropped portrait at t=0 and let the previous tail follow it. The portrait supplies the
identity pull, the tail supplies motion continuity, and those conditioned frames are dropped from
the output so nothing is duplicated.

**The general lesson:** a chain of N stages that each preserve *most* of an attribute does not
preserve it N times over — the error compounds multiplicatively. Any attribute you care about must
be re-anchored on its source at every stage, not merely inherited from the previous stage. The same
argument applies to identity here, to colour/exposure across a shot, and to any metric reported as
"mean over the whole clip": a good mean can hide a monotonic decay, so always fit a slope per
segment rather than quoting an average.

**Verified in kernel v9:** re-anchoring took the interior slopes from -0.00037 / -0.00209 /
-0.00176 to -0.00029 / +0.00020 / +0.00146, mean 0.811 -> 0.831, min 0.687 -> 0.749, and frames
below 0.80 from 70/135 to 31/132. The trade is a one-time identity step at each cut (-0.04, -0.05
instead of -0.02, +0.01): anchoring on the reference buys a flat slope everywhere for a small
discontinuity exactly at the handoff. Segment starts then converge rather than compound
(0.9176 -> 0.8221 -> 0.8058, drops -0.096 then -0.016), so the chain settles into a band instead of
decaying without bound.

### 15.9 Per-cut tolls: why flat slopes still drift

The v10 run (5 segments) kept every interior slope flat (-0.0009 to +0.0010/frame — the §15.8
re-anchor works) yet still walked identity from 0.898 down to 0.702. Two effects that a per-segment
slope fit is blind to:

1. **A step toll at every cut** (~-0.04 identity and a luminance step per handoff). Steps
   compound exactly like slopes: N segments pay N-1 of them. Slope fits measure *within*-segment
   trends and skip the boundaries, so fit the **segment starts** too — here they fell
   0.928 -> 0.851 -> 0.789 -> 0.740 -> 0.709.
2. **A composition random walk.** corr(identity, inter-ocular distance) = **+0.874**: the framing
   drifted ~24% wider over the first segments (IOD 111.7 -> 85.6) and the identity metric fell
   with the face size. Before diagnosing "identity drift", check geometry — pose/scale often
   explains the drop, and no amount of identity anchoring fixes a framing walk.

Also: luminance flicker grew from clean (2 cuts) to 0.060 > 0.05 threshold (4 cuts) — every cut
is a brightness step between independently seeded generations, so flicker risk scales with cut
count; deflicker is the post-process of record.

**The structural lesson (corrected by v11):** a diffusion model does **not** "continue from the
last conditioned frame" — it bridges the *entire* conditioning window. D-9.17 predicted that moving
the portrait to the end of the head would reset the level; v11 measured the opposite: handoff steps
-0.12 to -0.38, min identity 0.317 (below the same-person threshold), 1 multi-face and 3 no-face
frames. A degraded tail anywhere in the window poisons the handoff, in either order — so the window
must contain **only** source-of-truth content. Segment 0 proves the recipe: head = [portrait]
alone, nothing else — start 0.92, flat slope, stable framing, zero defects in every single run. For
an attribute that must survive N stages, each stage must generate from the source alone; chaining,
even with the source present in the window, inherits the previous stage's error.

**Verified in kernel v12 (portrait-only segments, 5 x 49 frames = 15.0 s):** every segment starts
0.907-0.928 (re-pin), handoff steps **+0.02 to +0.08** (cuts now *gain* identity instead of paying
a toll), mean 0.8752 / min 0.7870 / std 0.0339, 0 frames below 0.75, 240/240 frames with exactly
one face, analyzer 0 issues (NONE, face sim 0.877), framing held at IOD 120.0 cv 0.099. The
between-take luminance steps never tripped the 0.05 threshold. Cost: a pose reset at each cut
(jump-cut look) — the accepted trade for a flat identity band.

---

*End of knowledge base. Update this file as new experiments confirm or deny methods.*
