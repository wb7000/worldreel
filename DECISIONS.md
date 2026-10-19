# DECISIONS.md

Running log of decisions, defaults chosen under ambiguity, assumptions and deviations.
Newest entries at the bottom of each phase. Format: `D-<phase>.<n>`.

---

## Phase 0 — Foundation (2026-10-15)

### D-0.1 Hardware of the primary dev machine (measured 2026-10-15)
| Item | Value |
|---|---|
| OS | Windows 11 Pro (build 26300) |
| CPU | 11th Gen Intel Core i5-1135G7 @ 2.40 GHz (4 cores / 8 threads) |
| RAM | ~15.7 GB usable |
| GPU | **Intel Iris Xe (integrated). No NVIDIA GPU; `nvidia-smi` not present.** |
| Disk | D: 321 GB, ~119 GB free |
| Toolchain | Python 3.12.10 (3.13 also installed), Node 24.9.0, npm 11.6.0, git 2.45.1 |
| Missing | ffmpeg (needed from Phase 2). Blender 3.6 is installed but not on PATH (see D-0.9) |

Re-run `python -m tools.hwcheck` to refresh; output goes to `hardware.local.json` (git-ignored).

**Consequence (important deviation from the brief's "4–8 GB VRAM" assumption):** there is no CUDA
device, so *no* GPU training or GPU inference of video/3D models is possible locally.
- Local machine: specs, formats, pipeline framework, mocks, player (WebGPU via the Iris Xe in the
  browser should work — to be verified in Phase 1), data tooling, CPU-only small models, tests.
- Anything needing CUDA (Gaussian splatting training, depth/pose models at speed, video diffusion,
  LoRA fine-tuning) is planned for **cloud GPU** and must stay behind config files
  (`/training/configs`) so it moves machines without code changes.
- Every stage must keep a CPU/mock path so CI and this laptop can run the full pipeline.

### D-0.2 Repository location
The brief targets "the repo". `D:\new_IDEA` already contains an unrelated project
(`freelancer-tax-kit`), so WorldReel lives in its own repository at `D:\new_IDEA\worldreel`.
`git init` done locally on branch `main`; remote `origin` = `https://github.com/wb7000/worldreel`
(see D-0.9).

### D-0.3 Languages
- Pipeline, stages, tools, eval, training: **Python 3.12** (ecosystem for CV/ML models).
- Player: **TypeScript (strict)**, built with plain `tsc` for now; bundler chosen in Phase 1.
- Tests: `pytest` (Python), `vitest` (TypeScript).
- Lint/types: `ruff`, `mypy --strict`.

### D-0.4 Python package layout
The brief's top-level folders (`pipeline`, `stages`, `tools`, `eval`, `training`) are used directly
as importable Python packages from the repo root (declared in `pyproject.toml`), rather than nesting
everything under one `worldreel` package. Keeps the brief's layout literal. Risk: generic names
(`pipeline`, `tools`) could clash with third-party packages of the same name; revisit if that
happens. Python tests live in a root `/tests` folder (not named in the brief).

### D-0.5 Large-file tracking
`.gitignore` blocks data/media/weight extensions and `data/`. CI also fails if any tracked file is
larger than 5 MB. Git LFS / DVC are **not** set up yet; decided when a real need appears (likely
Phase 2 for dataset-version lists that point at remote storage).

### D-0.6 CI
GitHub Actions (`.github/workflows/ci.yml`): Python job (ruff, mypy, pytest) on ubuntu + windows,
player job (typecheck + vitest), and a repo-hygiene job (no large files). CI first runs on the initial
push to GitHub.

### D-0.7 Stage package skeletons
All 11 stages from the brief exist as packages in `/stages` with a docstring describing their
responsibility and a `STAGE_NAME`. No stage logic or mocks in Phase 0; the stage framework is
built in the phase where it is first needed (pipeline framework + mocks are prerequisites of
Phase 2; format libraries come in Phase 1).

### D-0.8 External tools named in the brief — status: unverified
- **"Soup" CLI** for LLM fine-tuning (`soup train`, layer streaming): not verified to exist as
  described. Marked as an **assumption**; must be checked against official docs before use
  (Phase 7). Not a dependency now.
- No ML models chosen yet; `docs/MODELS.md` is an empty register.

### D-0.9 Remote, Blender, ffmpeg, cloud GPU (2026-10-15, owner input)
- **Remote:** `https://github.com/wb7000/worldreel` (owner-created, empty at first push).
- **Blender:** 3.6.23 installed at `C:\Program Files\Blender Foundation\Blender 3.6\blender.exe`,
  not on PATH. Tools will take the Blender executable from config/env (`WORLDREEL_BLENDER`)
  instead of assuming PATH. 3.6 is an LTS line; generator scripts target the 3.6 Python API.
- **ffmpeg:** not installed yet; owner installs a prebuilt Windows build (not source). `winget`
  is not available on this machine.
- **Cloud GPU (tentative):** Kaggle Notebooks free tier, quota as reported by the owner: 30 GPU
  h/week, 12 h max session. Treat the quota as an owner-reported value; check Kaggle's page
  before planning around it. Implications: sessions are not persistent, so all training must
  checkpoint often and resume (target: checkpoint at least every 30 min, finish or save before
  11 h). Backup: Google Colab free tier. Not needed before Phase 3.

---

## Phase 1 — Formats and player skeleton (2026-10-16)

### D-1.1 One agent instead of parallel worktree agents
The brief suggests format-agent and player-agent in parallel. The player depends directly on the
`.wreel` spec, and the phase was small enough for one agent working spec → library → player in
order on branch `phase-1-formats`. Parallel agents start in Phase 2 (data and pipeline are
independent).

### D-1.2 New top-level `formats/` Python package
The brief's layout has no home for format libraries. They live in `/formats` (`wreel.py`,
`wedit.py`, `wscript.py`, `common.py`), next to `/spec`. CLI: `python -m formats validate <file>`.

### D-1.3 Container and signing
- `.wreel` is a ZIP with `manifest.json` + `signature.json`; every other entry is listed with
  SHA-256 in the manifest. The Ed25519 signature covers the **exact bytes** of `manifest.json`,
  so no JSON canonicalisation is needed and the signature covers every file transitively.
- Python signing uses the `cryptography` package (Ed25519). The local key lives at
  `~/.worldreel/signing_key.pem` (or `$WORLDREEL_SIGNING_KEY`), created on first use, never in
  the repo. There is no key trust list yet: "signature valid" means "unchanged since signed by
  this key", not "trusted author". Recorded in the spec (§4.1).
- Writers are deterministic (sorted entries, fixed 1980 timestamps): same inputs + key + time →
  byte-identical file. Tested.
- Readers reject unsafe paths, encrypted entries, entries > 4 GiB and unlisted entries.

### D-1.4 Validation
JSON Schema 2020-12 files in `spec/schemas/` (validated with `jsonschema`), strict
`additionalProperties: false`, plus semantic checks (references, LOD sizes, monotonic times,
unit quaternions, confidence range, identity policy vs people). Files from a newer *minor*
version are read with unknown fields downgraded to warnings; another *major* is refused.
Writers refuse to write a file that does not validate.

### D-1.5 Geometry v1 = point layers only
`kind: "points"`, 20-byte little-endian records (xyz f32, rgb u8, provenance-class u8,
confidence f32), uploaded to the GPU unchanged. LODs are independent (each LOD replaces the
previous). Gaussian splats and meshes are reserved kinds for Phase 3.

### D-1.6 `.wscript` design
Header `wscript.json` (dictionary, scene index with summaries, continuity per scene) plus one
`.wsl` text file per scene in a small line-based language. Canonical files round-trip byte for
byte (tested). Tension inside `.wsl` is an integer 0–10. **No token-saving figure is claimed**:
the structure is designed to save tokens but has not been measured with a tokenizer. Measuring
it belongs in Phase 7.

### D-1.7 `.wedit` scope in Phase 1
Container (directory by default, ZIP optional), manifest schema, signature, auto-generated
`PROVENANCE.txt`, validation. Producing real video/stems/layers and an OpenTimelineIO timeline is
Phase 6; the OTIO file is only referenced by path today.

### D-1.8 Player stack (exact versions in `player/package-lock.json`)
TypeScript 5.6.3, Vite 5.4.21 (dev server and build), Vitest 2.1.8, fflate 0.8.3 (unzip),
@webgpu/types 0.1.74. Renderer: **WebGPU first, WebGL2 fallback** (`?renderer=webgl2|webgpu`
forces one). Points are drawn as instanced camera-facing discs straight from the `.pts` bytes.
Verified 2026-10-16 in the Claude desktop browser pane (Chromium) on this laptop's Intel Iris Xe:
WebGPU and WebGL2 both render the sample; phone width (375 px) checked. **Frame rate was not
measured**: the pane throttles animation while hidden, so the on-screen fps readout (1–4) is not
a real number. Measure in a real browser window and on a phone in Phase 4 (brief §12).

### D-1.9 Player validation is deliberately minimal
The player checks structure, version, provenance label, the audio slot, entry hashes, unlisted
entries and the signature (WebCrypto Ed25519; it reports "not checked" where a browser lacks
Ed25519). Full schema validation stays in Python.

### D-1.10 Confidence rendering
Points that are not `observed`, or have confidence < 0.5, blend toward a **violet vertical
gradient** by `1 - confidence`; guessed/retrieved/generated points are always ≥ 60 % blended. The
first attempt blended toward a pale blue close to the sky colour and made the guessed area behind
the wall almost invisible from overhead, so it was changed to violet. The viewer can choose "soft"
but cannot turn it off (spec §10.3).

### D-1.11 Not used by the player yet
Lighting (stored in the format, ignored by the point shader), audio (slot present and validated;
no playback until Phase 5), interaction rules beyond `time` → `set_tension` / `switch_pov`
(others are listed as skipped), camera collision (the follow camera can pass through pillars).

### D-1.12 Progressive loading is decode-level only
The whole file is downloaded, then the coarse LOD is decoded and drawn first and the fine LOD is
swapped in afterwards. Network streaming (HTTP range requests, spatial chunks) is future work.

### D-1.13 ffmpeg
ffmpeg 9.0.2 (gyan.dev "essentials" Windows build) at `D:\ffmpeg\bin`, on the user PATH.

### D-1.14 Sample scene
`tools/make_sample_wreel.py` builds a synthetic courtyard: observed floor/wall/pillars, inferred
pillar backs, a guessed floor behind the wall, two walking people with synthetic identities, a
director camera, six POV presets, footstep events and a mood curve. Output goes to
`player/public/samples/courtyard.wreel` (git-ignored); CI regenerates it.

---

## Phase 2: Data pipeline and synthetic generator (2026-10-16/17)

### D-2.1 One agent again
The pipeline framework and the data tools ended up sharing the `Item` type, the processing
command and the end-to-end acceptance test, so they were built in order on branch
`phase-2-data` rather than by two parallel agents.

### D-2.2 External tools via `tools/toolpaths.py`
ffmpeg, ffprobe and Blender are found via `WORLDREEL_<TOOL>` env var, then
`worldreel.local.toml` (git-ignored; template `worldreel.example.toml`), then PATH, then known
install folders. Reason: PATH changes do not reach already-running programs, and Blender is not
on PATH on this machine.

### D-2.3 Stage framework
- Stage = class with `name`, `version`, `impl`, `depends_on`, `run(ctx)`; outputs only through
  files (names in `pipeline/io.py`).
- Cache key includes the *content hash* of every input stage's output, so a re-run with
  identical output does not invalidate dependents (tested).
- Atomic temp folders; leftovers from crashes are deleted on the next run.
- Status values are the brief's four (`pending/running/done/failed`); a stage that cannot run
  because an upstream stage failed is `failed` with error `blocked: upstream X failed` instead
  of a fifth status.
- Parallelism is a thread pool inside one item. Stages that need real CPU parallelism can start
  subprocesses (ffmpeg already does). Process pools and multi-item scheduling are deferred.
- The runner refuses to start without a safety hook. Phase 2 hook: logging pass-through
  (`stages/safety/hook.py`), every decision in `processed/<id>/safety.jsonl`.
- Pipeline configuration is TOML (`pipeline/default.toml`): implementation per stage,
  per-stage options, seed, workers.

### D-2.4 Mocks
Every pipeline stage has a CPU mock (table in `docs/ARCHITECTURE.md`). The mocks are
deliberately simple and say so in their output (`"mock": true`): mock camera = static camera
over a flat floor, mock actors = nothing found, mock fill = grey `guessed` floor behind the
camera. **Their output is not a reconstruction**; the player shows it only to prove the plumbing.
`package` has no mock: the real one is cheap and deterministic.

### D-2.5 Blender depth semantics (measured)
Probe: a flat wall exactly 5 m in front of the camera, 160×120. Depth pass is constant across
the image: Cycles 5.0000, EEVEE 4.9961 (0.08 % low). So Blender 3.6.23's depth pass is
**planar** (along the camera axis), not distance along the ray. "No surface" pixels are 65504 in
EEVEE (float16 maximum) and huge in Cycles; anything at or beyond the far clip (200 m) is stored
as 0. The first version used a 1e5 threshold and the self-check caught it (13 % of points below
the floor).

### D-2.6 Blender ↔ WorldReel coordinates
World: (x, y, z)_Blender → (x, z, -y)_WorldReel (-90° about X). Cameras use the same local
convention in both, so only the world change applies. Generated actors face Blender local +Y and
get an extra +90° local-X rotation so they face WorldReel local -Z. Tests check both against
explicit vectors, and `tools/synth.py check` verifies on every render that actors face their
direction of motion (measured mean cosine 1.000 on seed 1).

### D-2.7 Ground-truth self-check (`tools/synth.py check`)
Back-projects depth through the ground-truth camera. Requires: < 0.1 % of points below the floor;
ground points from frame 1 reprojected into the last frame match its depth (median relative
error < 5 %; measured 0.25 % on seed 1, EEVEE, 320×240, 24 frames); actors face their motion.
A render takes about 10 s on this laptop (EEVEE, 320×240, 24 frames, measured once).

### D-2.8 Data rules beyond the brief
- File id = `f` + the first 12 hex digits of SHA-256 (collision risk negligible at our scale;
  the full hash is stored).
- No-Derivatives licences are held in `needs_info`: reconstruction makes a derivative work.
- Duplicates and unreadable files go to `data/rejected/` with a `.reason.txt` (never deleted
  silently).
- Excluded files stay in `raw/` so older frozen versions still reproduce.
- `manifest.csv`, `removals.csv` and `versions/` may be committed (brief §2: manifests and
  dataset-version lists are committed); everything else under `data/` is ignored. The manifest
  holds the `source` and `notes` text the owner writes, so do not put private details there.
- `tools.data` assumes one process at a time (no file locking yet).

### D-2.9 CI
The Linux Python job installs ffmpeg so the real `frames` stage and the acceptance test run in
CI; on Windows CI they skip. Blender tests skip in CI (Blender is not installed there).

---

## Phase 3a: Classical reconstruction baseline, CPU (2026-10-17)

Phase 3 is split. **3a** (this entry) runs on the laptop CPU: evaluation metrics, a classical
structure-from-motion + multi-view-stereo baseline, a benchmark on synthetic ground truth.
**3b** (cloud GPU) adds learned depth/pose, Gaussian splatting and moving actors. The brief's
Phase 3 acceptance is not complete until 3b and a real-video test are done (see end of entry).

### D-3a.1 Libraries
`opencv-python-headless` 5.0.0 (Apache-2.0) for SIFT, essential matrix, PnP, triangulation,
warping; `scipy` 1.18 (BSD-3) for bundle adjustment. No pretrained model weights are used in 3a,
so `docs/MODELS.md` is still empty. OpenCV 5.0 is newer than the agent's reference knowledge;
every API used was checked to exist and exercised by tests before relying on it.

### D-3a.2 Methods (all established, none invented)
- Camera: incremental SfM (SIFT, mutual ratio-test matching, 5-point essential matrix with
  RANSAC, EPnP + LM refinement, triangulation with angle/reprojection checks, global bundle
  adjustment with Huber loss). `stages/camera/sfm.py`, wrapped by `stages/camera/classical.py`.
- Reconstruct: plane-sweep stereo (ZNCC, best-2-of-N source aggregation, parabola sub-plane
  refinement, texture and geometric-consistency filters) and voxel fusion.
  `stages/reconstruct/planesweep.py`.
- Monocular scale and gravity are unknown: world = first camera's frame; scale set so the median
  sparse depth in frame 1 is 5 m (configurable, recorded in camera.json).
- Without calibration, a 60° horizontal field of view is assumed (configurable).

### D-3a.3 Synthetic scenes needed texture
Flat-coloured Blender materials gave ~34 SIFT features per frame and SfM could not start. Real
surfaces have texture, so the generator now adds procedural hard-edged noise (`--texture`,
default 2.0). Soft cloudy noise gave ~100 features; hard-edged blotches give ~2,300 at 480x360.

### D-3a.4 The orientation bias and its fix (measured)
First SfM version on seed 11: positions 0.87 % of path, but 3.26° mean orientation error with
only 0.08° frame-to-frame error. Investigation, in order:
1. Alignment on positions only is ill-conditioned on a short arc → alignment now also uses each
   camera's axes (eval/geometry.py). 5.7° → 3.26° (real remaining error).
2. Orientation-only alignment gives 0.55°: the estimate is internally consistent but sheared.
3. More BA iterations, a wider matching window, a scale-gauge constraint, periodic BA:
   3.26° → 3.0–3.2° only.
4. Re-triangulating the same tracks with the **true** cameras gives a *lower* reprojection error
   (0.231 px) than the estimate (0.261 px); BA started from the true cameras stays there
   (0.008°). So the model is right and SfM converged to a worse local minimum.
5. Cause: a short initial baseline (first frame with 3 % parallax). Choosing the **widest
   baseline initial pair** that still has ≥ 80 essential-matrix inliers (standard practice, e.g.
   COLMAP) gives 1.46° and halves every other error. Kept: widest-pair initialisation, scale
   gauge in BA (harmless, correct). Periodic BA is available (`ba_every`) but off: +40 % time for
   ~0.01°.

### D-3a.5 Ground-truth novel views
The generator renders `--eval-views N` extra cameras (10–25° around the scene, ±1.5 m height) at
frame 1 that are never used for reconstruction; `tools/synth.py check` verifies them
(reprojection error ≈ 0.24 %).

### D-3a.6 Thresholds: what was seen when (honesty record)
`eval/thresholds.toml` was first written after one SfM run on seed 11 had shown trajectory
numbers (ATE 0.87 %, rotation 3.26°), but before any depth or novel-view number. The rotation
threshold (2°) was deliberately set below that observed value. Depth thresholds were set blind.

The first novel-view thresholds (PSNR ≥ 20 dB, SSIM ≥ 0.6, absolute) were **infeasible**: an
oracle (true geometry, same reference frames and density, same point renderer) only reaches
~19–20 dB / SSIM ~0.4 on this texture. They were replaced, after seeing seed 11's results, by
oracle-relative thresholds (gap ≤ 2 dB, SSIM ratio ≥ 0.7). Rendering through the global Sim(3)
alignment also mixes in the residual pose bias, which is already scored by the trajectory
metrics, so the thresholds use the **localised** protocol: the eval camera is first located in
the reconstruction by SIFT + PnP (as BARF / NoPe-NeRF optimise test poses). Both `aligned` and
`localized` numbers are reported. Seeds 12 and 13 were rendered only after these choices.
The oracle is a reference, not a strict ceiling: fusing many frames can beat it (seed 11).

Bugs found while validating: the oracle used *estimated* intrinsics in uncalibrated runs (made
the gap pass trivially), fixed to always use the true intrinsics; pooling more frames for
localisation was tried after seeing seed 13 and **reverted** because it did not help.

### D-3a.7 Results (480x360, 30 frames, 7 textured boxes, no actors, 4 eval views)

| Scene | Intrinsics | ATE/len | Rot (deg) | RPE rot (deg) | Depth AbsRel | delta1 | Depth compl. | Novel PSNR loc / oracle | SSIM loc / oracle | Cover | All pass | camera + reconstruct (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | true | 0.0040 | 1.46 | 0.041 | 0.035 | 0.988 | 0.92 | 20.09 / 19.75 | 0.445 / 0.400 | 0.97 | yes | 56 + 10 |
| 11 | assumed 60 deg | 0.0056 | 1.89 | 0.076 | 0.056 | 0.986 | 0.93 | 20.12 / 19.75 | 0.451 / 0.400 | 0.98 | yes | 58 + 10 |
| 12 | true | 0.0014 | 0.29 | 0.026 | 0.021 | 0.988 | 0.93 | 19.43 / 20.07 | 0.409 / 0.438 | 0.90 | yes | 65 + 11 |
| 12 | assumed 60 deg | 0.0036 | 1.12 | 0.065 | 0.049 | 0.988 | 0.94 | 19.49 / 20.07 | 0.411 / 0.438 | 0.90 | yes | 66 + 10 |
| 13 | true | 0.0022 | 0.75 | 0.030 | 0.020 | 0.985 | 0.91 | 16.78 / 20.45 | 0.271 / 0.442 | 0.76 | **NO** | 40 + 10 |
| 13 | assumed 60 deg | 0.0038 | 1.45 | 0.066 | 0.031 | 0.985 | 0.91 | 16.76 / 20.45 | 0.270 / 0.442 | 0.74 | **NO** | 42 + 9 |

(True horizontal FOV is 61.9°, so "assumed 60°" is a mild 3 % focal error.) Trajectory and depth
pass on all six runs. Novel views fail on seed 13: one eval view looks at the boxes from an
angle the camera path barely covers; localisation finds only 45 matches and the pose is wrong.
Kept as a recorded failure: it is a limitation of sparse classical matching and of point
rendering, targeted by 3b (learned features, splats).

Sensitivity: at 320x240 with 16 frames the orientation error is 4.0° (fails), so the method
needs reasonable resolution and sequence length. Timing is SfM-dominated (~1 min per 30 frames
at 480x360 on this laptop); matching is brute force.

### D-3a.8 Defaults
`pipeline/default.toml` now uses the classical `camera` and `reconstruct` stages for real
videos (CPU), frames at 6 fps. The Phase 2 acceptance test uses an explicit frames-only config
because its ffmpeg test pattern has no camera motion.

### D-3a.9 Not done in 3a (→ 3b / later)
Moving actors (segmentation, tracking, per-actor geometry), learned depth/pose, Gaussian
splats, LPIPS (needs a network), gravity/scale estimation, loop closure, a speed-up of
matching. **A real (non-synthetic) video** has not been processed yet: needs a clip from the
owner (Phase 3 acceptance: "a short real video becomes a free-camera .wreel"). A synthetic MP4
through the full video path (ffmpeg → SfM with assumed intrinsics → plane sweep → package)
works and opens in the player with a free camera.

### D-3a.10 First real video (owner's phone, 2026-10-17)
- A first submitted clip was an edited third-party video (cuts, captions, stock footage) whose
  info note claimed own recording. It was not processed (stopped on inspection) and is waiting
  for the owner's decision (remove, or correct its source). Lesson: ingest trusts the sidecar,
  so the agent looks at new media before processing it.
- The second clip is the owner's own phone recording of their room: 23 s, 480x854 portrait,
  hand-held, fast panning (100–260 px median feature motion in 0.5 s), plain walls, motion blur.
  The agent created its info file on the owner's behalf (source "my own recording").
- With frame 0 as the only possible start, SfM could not start (door + mirror, rotation only).
  Changes, both standard for real footage and neutral on the benchmark (frame 0 is still tried
  first; all 137 tests incl. the seed-12 benchmark unchanged):
  1. if frame 0 cannot start, search initial pairs across the whole video (stride 2, span 12);
  2. register outward from the initial pair in both directions; bundle adjustment fixes the
     gauge at the initial frame; the package's camera path uses tracked frames only.
- Config `data/real_video.toml` (local): 8 fps, assumed 60° FOV, 1.5 m median depth,
  `min_registered_fraction = 0.2`.
- **Result:** 62 of 183 frames tracked (frames 12–73, ≈ 1.5–9 s, the textured corner: bed,
  books, laptop table), reprojection RMSE 0.75 px, 12,113 sparse / 244,752 fused points, about
  2 minutes on the laptop. Rendered from tracked cameras the reconstruction covers 3–49 % of each
  frame (PSNR 15–19 dB on covered pixels; no ground truth, so no accuracy number). Plain walls
  and the fast second half are missing. The `.wreel` validates, is signed, labelled
  AI-reconstructed, and plays with director, free and overhead cameras.
- Phase 3 acceptance "a short real video becomes a free-camera .wreel": **met for the trackable
  part of the clip**. Better results need slower camera motion with sideways movement, or the
  learned components of Phase 3b.

---

## Phase 3b: GPU reconstruction with Gaussian splats (2026-10-17, in progress)

### D-3b.1 Kaggle runner and environment (measured)
`tools/kaggle_runner.py` uploads tracked code only (never `data/`, never credentials) and
scenes as PRIVATE datasets of the owner's account, pushes a job script as a private GPU kernel,
waits and downloads outputs to `runs/kaggle/` (git-ignored). Credentials: Kaggle's new
`KGAT_` access token in `C:\Users\li\.kaggle\access_token`, never in the repository.
Environment check on Kaggle: 2x Tesla T4 (15 GB), Python 3.13.15, PyTorch 2.11.0+cu128,
gsplat 1.5.3 (Apache-2.0) installs in 6 s; its CUDA code compiles on first use in ~5.6 min per
job. Kaggle unpacks uploaded zip files and mounts datasets under
`/kaggle/input/datasets/<user>/<slug>/`; `training/kaggle/bootstrap.py` handles both forms.

### D-3b.2 gsplat camera convention (measured, not assumed)
A Gaussian at camera-space +z is visible and one at -z is not; one at +y renders below the image
centre. So gsplat uses OpenCV axes (x right, y down, z forward) with world-to-camera matrices,
which is what `stages/reconstruct/splat_train.py` assumes.

### D-3b.3 Kaggle queue behaviour (observed 4 times)
Freshly pushed GPU jobs often stayed QUEUED for 20–60+ minutes, while deleting and pushing the
same job again usually started it within seconds (once it stayed queued). The runner now does
this automatically: `--requeue-after 20` (owner's rule) is the default.

### D-3b.4 Splat trainer and the fog investigation
`stages/reconstruct/splat_train.py`: 3D Gaussian Splatting (Kerbl et al. 2023) on gsplat,
initialised from the classical SfM points and poses (so it inherits their accuracy), plain RGB
colour, loss 0.8·L1 + 0.2·(1-SSIM), 7,000 steps, gsplat DefaultStrategy densification.
History, in order (single runs, so see D-3b.6 for what holds up):
1. No background: seed 11 novel views 26.9 dB avg, but sky-coloured "floaters" in the higher
   eval views.
2. A far "background sphere" of Gaussians (common for unbounded scenes) with gsplat's 3D prune
   limit raised to 1.0 x extent so they survive: seed 11 → 29.0 dB, but seed 13 got a fog veil.
3. Background Gaussians started small (≤ 0.05 x extent, 100k of them) with the default 0.1 limit.
4. A diagnosis job proved (2): the raised limit let ~240 huge Gaussians (median size 3.2 units)
   sit in front of the scene. It also showed that the same configuration scores 15.2 dB in one
   run and 24.2 dB in another on the same view: **single training runs are not comparable**.

### D-3b.5 Robust scene extent
The extent was the maximum point distance; a few stray SfM points made it 15.9 units on
seed 13 while the cameras span ~5. It now uses the 95th percentile (x1.1). It scales the prune
limit, the position learning rate and the background sphere.

### D-3b.6 Variance experiment (protocol fixed before results)
Protocol written before the run: 2 variants x 3 training seeds x 3 scenes; headline = eval
views localised with ≥ 1000 PnP inliers (fair test of the splats, not of localisation); all
views reported too. Results (PSNR / SSIM, mean over 3 seeds, non-sky pixels):

| Scene (well-localised views) | Classical (Kaggle run) | Splats, no background | Splats, small background |
|---|---|---|---|
| 11 (4 of 4) | 20.13 / 0.447 | 25.05 / 0.742 (runs 25.9, 30.4, 18.8) | **28.54 / 0.873** (24.9, 30.8, 29.9) |
| 12 (2 of 4) | 20.75 / 0.497 | 31.14 / 0.941 (30.7, 31.1, 31.6) | **32.78 / 0.956** (32.8, 32.6, 33.0) |
| 13 (1 of 4) | 20.78 / 0.543 | 22.09 / 0.791 (22.5, 20.9, 22.8) | **22.24 / 0.787** (23.1, 22.5, 21.1) |

Decisions: splats beat the classical reconstruction on every scene (1.4–12 dB, SSIM 0.79–0.96 vs
0.45–0.54) → **adopted** as the Phase 3b reconstruction method. The small background sphere is
better or equal on every scene and steadier on seed 11 → **adopted** as default. Training takes
~100–420 s per scene on one T4. Open problems: run-to-run spread is still ~6 dB on seed 11;
5 of 12 eval views cannot be localised reliably with SIFT + PnP (≤ 230 inliers), which limits
evaluation on seeds 12 and 13. Classical numbers on Kaggle differ slightly from the laptop
(different OpenCV build); comparisons above use the Kaggle run only.

### D-3b.7 Gaussian layers in `.wreel` 1.1 and the player (2026-10-17)
- Spec §6.3: `"kind": "gaussians"` layers with 52-byte records (centre, std-dev scales, unit
  quaternion, sRGB colour, opacity, provenance class, confidence). Files with Gaussian layers are
  1.1; point-only files stay 1.0 so older players can open them. View-dependent colour (SH) is
  reserved for a later minor version.
- Conversion (`stages/package/gaussians.py`): OpenCV → WorldReel axes as R' = F·R·F (scales
  unchanged), tested to preserve every covariance. Gaussians with opacity < 1/255 are dropped.
  Background-sphere Gaussians are found by distance (densification reorders them) and stay
  `observed` (their colour is seen) with confidence 0.6 (their distance is assumed); the
  rest get 0.9. LOD 0 = the 20 % with the largest opacity x cross-section.
- Signing stays local: Kaggle returns only parameters + an unsigned-for-us template;
  `python -m tools.splat_wreel` builds and signs with the owner's key.
- Player: EWA projection of each Gaussian to a screen ellipse (±3σ quad, 0.3 px² low-pass),
  back-to-front "over" blending from a CPU 16-bit counting sort, re-sorted only when the view
  changes and at most every 120 ms; splats are depth-tested against points but do not write
  depth; far plane raised to 1000 m. Same maths in WGSL and GLSL.
- Verified 2026-10-17 in the Claude browser pane on the Iris Xe: seed 11 splats (414,358
  Gaussians, 16.6 MB) render correctly in WebGPU and WebGL2, original and free cameras. Frame
  rate not measured (pane throttles). Regions no training camera saw show streaks — expected.

### D-3b.8 Owner's room video with splats (2026-10-17)
- With the owner's explicit consent, the room video was uploaded to the owner's Kaggle account
  as a PRIVATE dataset (`worldreel-video-room`: the video, its source/licence note and the
  pipeline config only). `python -m tools.kaggle_runner push-video <id>` does this.
- Kaggle has ffmpeg/ffprobe at /usr/bin. Pipeline (`data/real_video.toml`) ran end to end; the
  camera stage registered 49 of 183 frames there (62 on the laptop; different OpenCV build).
- Splats trained on the 49 registered frames from 8,644 SfM points: 997,822 Gaussians after
  densification, 988,430 kept (opacity ≥ 1/255) → `room_splats.wreel` 47.8 MB, signed locally.
- In the player (WebGPU, Iris Xe) the tracked part of the room is photo-like: door and mirror,
  window ledge with cable, wall, bed cover, laptop with its screen content, mouse, blue cloth.
  Untracked parts and views far from the camera path smear. No numeric quality score: real video
  has no ground truth; a held-out-frame evaluation is the next step. Frame rate not measured.
- Phase 3 acceptance ("a short real video becomes a free-camera .wreel") is now met with the
  GPU method as well as the classical one, for the trackable part of the clip.

### D-3b.9 Moving people: masks, lifting, synthetic avatars (2026-10-17)
- Mask format shared by all actor backends: `masks.npz` (uint8 labels per frame, 0 = nobody,
  k = track k) plus `actors.json` (track ids, frames, pixel counts). Camera (SfM features) and
  plane-sweep depth ignore people's pixels (masks dilated 4 px / 2 px) so walkers don't corrupt
  the static scene.
- Backends: `mock` (no people), `oracle` (Blender ground truth: depth back-projected and tested
  against each actor's body cylinder + head sphere; for testing lifting in isolation). The real
  backend (DETR person boxes → SAM 2.1 video propagation, both Apache-2.0) comes next; its
  interfaces are being checked on Kaggle first, not assumed. torchvision Mask R-CNN weights
  were rejected: their licence is unclear.
- Lifting (invented, `stages/actors/lift.py`): RANSAC floor plane from static points (normal
  within 35° of +Y), foot pixel ray ∩ floor, height from the head pixel ray at the foot's depth,
  median height per track, symmetric median smoothing, facing from velocity.
- Body-depth correction: the visible foot contour is the near edge of the body, not its centre.
  With true cameras and floor the range bias was −0.20 m (= the 0.22 m test body radius). Fix:
  push each foot 0.075 × height away from the camera along the floor — a human-proportion
  value (≈13 cm for 1.75 m), deliberately not tuned to the test cylinders.
- Measured on synthetic scene seed 21 (2 walkers, 30 frames, 480×360, full pipeline, oracle
  masks), errors in each frame's own camera frame scaled by the trajectory Sim(3):
  p1 median 0.385 m (range bias −0.380 m), p2 0.361 m (−0.358 m); estimated heights 1.77 m
  for both (true 1.75). The remaining error is mostly along the viewing ray at ~12 m range and
  grazing angles, where a small floor/camera error moves the foot intersection a lot (camera
  height above the fitted floor was within 12 cm). Not good enough for close first-person
  views; acceptable for follow/overhead views. A bigger fix (e.g. joint floor+track bundle
  adjustment) goes in the roadmap, not this phase.
- Heights stay in reconstruction units (monocular scale is unknown), so there is no metric
  clamp; an early 1.2–2.1 m clamp made every avatar 1.2 units and was removed.
- Identity: every person becomes a seeded, faceless, neutral-coloured avatar scaled to the
  estimated height (`identity.policy = "synthetic-replacement"`, seed stored); only motion comes
  from the video. POV presets per person: "Follow person N" (third person) and
  "Person N's eyes" (first person, comfort mode).
- Verified in the player (WebGPU, Iris Xe): both presets play and follow the avatar.

### D-3b.10 Real actors stage: DETR + SAM 2.1, and the static-track guard (2026-10-17)
- `stages/actors/sam2_stage.py` (impl `real`): DETR person boxes on every 5th frame → boxes
  linked into tracks by IoU (Hungarian, max gap 3 keyframes, ≥ 2 hits) → each track's first box
  prompts SAM 2.1 hiera_small → video propagation → highest positive logit per pixel.
- Interfaces checked on Kaggle before use (`actors_env` job): transformers 5.16.1 rejects the
  `no_timm` DETR config's `"dilation": null`, so it is set to False (plain ResNet-50); "person"
  is id 1; SAM 2.1 signatures recorded; from a ground-truth box SAM 2.1 reached IoU 0.92–0.94
  vs ground truth on 8 frames. The predictor must run under bf16 autocast, as in the official
  README; without it the full run crashed (mat1/mat2 dtype BFloat16 vs Float).
- Full pipeline on synthetic scene seed 21 ran end to end (125 s on Kaggle) but detection
  FAILED the test: DETR does not see the cylinder-and-sphere test people (23×67 and 15×49 px)
  as people. With "vase" allowed (test-only) it tracked 4 scene props instead: mask IoU 0.0,
  3.9 false tracks per frame, "heights" 3.2–3.6 m. Synthetic scenes therefore cannot score
  detection; that needs real footage (asked the owner for a short walking clip).
- Guard added from what that run showed: a track whose 3D path stays within 0.3 × its height
  of its median position gets no avatar (`static` in the package report). Measured separation:
  falsely detected props 0.08–0.16 × height, true walkers 0.53–0.57. Static tracks stay masked
  out of the static scene, so a real person standing still is never shown with their real
  appearance — they are simply absent (accepted loss, privacy first). On the Kaggle output the
  guard drops all 4 false tracks; on the oracle run it keeps both walkers.
- Rejected guard: person height vs camera height above the floor. The test camera is 4.96 m
  up, so true people scored 0.35 and props ~0.7 — it would have kept the props.

### D-3b.11 First real-people clip: owner's friends in a room (2026-10-18)
- Footage: owner's own 30.7 s phone clip of friends in a room (`f29ecb3c1553a`), owner states
  the friends approved; uploaded only as a PRIVATE Kaggle dataset. Three other clips the owner
  offered were NOT used: one showed only the owner in a mirror, two showed crowds/events with
  many third parties (some likely minors) who had not consented. Real people are shown only as
  synthetic avatars.
- Pipeline: frames 8 fps (245), camera registered only 82 frames (≈10–22 s; phone barely
  moved, 1,283 SfM points). Kaggle 2×T4: pipeline 26 min, splat training 7 min.
- Run 1: 19 tracks for a handful of people — tracks broke during fast pans and SAM 2.1 kept
  following the person with the old object, so people had several labels.
- Run 2 (merge on intersection / smaller area ≥ 0.6): 3 tracks, but two friends were chained
  into one label (frames 16–81, 211) and a wardrobe-mirror reflection became a person
  (height 1.93 vs 0.66/0.80). Checked by overlaying masks on 16 frames.
- Run 3 (current rule): objects merge only if median IoU ≥ 0.5 over ≥ 3 co-visible frames, and
  never if any member pair is co-visible with median IoU < 0.2 (cannot-link). Height guard:
  from 3 people up, a track outside 0.55–1.8 × the median height gets no avatar. Result:
  5 tracks → 3 avatars (one per friend, heights 0.66/0.76/0.80 units); rejected: the mirror
  reflection and a track that leaked onto the bedsheet / an extreme face close-up. A motion-
  blurred friend in one pan was missed. No ground truth, so no numeric score.
- Ghosts: a static splat model kept a faint, blurry silhouette of a real friend where masks
  missed frames — a privacy issue, not just cosmetics. Fix: `carve_people` deletes every
  Gaussian inside each tracked person's swept body volume (radius 0.35 × height = body
  half-width ~0.13 + measured lifting range bias ~0.22 from D-3b.9; from 0.04 to 1.1 × height
  above the floor). 46,019 of 461,750 Gaussians removed; silhouette gone in the player. A
  person who is never tracked at all is not carved — limit stated, not solved.
- Player/presets: follow camera aimed at a fixed 1.3 units and sat 2.2 × height (+0.6) back;
  scene units are arbitrary, so it looked at the ceiling from outside the room. Now aim at
  0.8 × eye height, offset [0, 1.15 h, 1.0 h]. Even so, follow/eyes views on this clip show
  mostly guessed geometry: the camera path is ~0.36 units long and only 82 frames registered.
  Better capture (walk around while filming) fixes that, not code.
- Verified in the player (WebGPU): original-camera view photo-like with avatars standing where
  the friends stood (15 s, 18.5 s); no console errors.

### D-3.end Phase 3 close-out (2026-10-18)
Final checks on main (b4e69ec + this entry): pytest 162 passed; ruff and ruff format clean;
mypy clean (86 files); `formats validate` sample OK; player vitest 28 passed; player build OK.

Acceptance (brief §Phase 3):
- "On synthetic scenes, depth/pose and novel-view metrics meet configured thresholds":
  trajectory and depth pass on seeds 11/12/13 (true and assumed intrinsics, D-3a.6). Novel
  views: classical passes 11 and 12, **fails 13** (recorded, thresholds not loosened); splats
  beat classical on all three (D-3b.6). Partially met — the seed-13 failure stands.
- "A short real video becomes a free-camera .wreel": **met** (room video, classical D-3a.10 and
  splats D-3b.8), for the part of the clip the camera stage can track.
- Dynamic (actors): mask format, lifting (synthetic local error 0.36–0.39 m, heights 1.77 vs
  1.75 m), DETR + SAM 2.1 stage, merge and guards, synthetic avatars, person carving; real
  clip gives one avatar per friend (D-3b.9–11). Detection is unscored on synthetic data
  (test bodies are not human-like) and real footage has no ground truth.

Known limits carried forward: camera tracking registers only part of handheld clips (82/245
on the walking clip); views away from the camera path show guessed geometry (blur/smears);
lifting range bias ~0.2 x height; untracked people are not carved; seed-13 novel views.

Owner's manual steps:
1. For better results, film slowly while walking through the space, each area from 2–3
   positions, good light (no code needed for the improvement).
2. Real people: only footage of yourself or adults who agreed; it goes to your PRIVATE Kaggle.
3. Unused clips are still in `data/inbox/` (vide.mp4, approved.mp4, aprove.mp4) — delete them
   when you like; nothing was uploaded from them.
4. Kaggle: datasets/kernels are private; GPU quota is weekly — the runner resubmits queued jobs
   after 20 min.


## Phase 4: POV and unseen-region filling (2026-10-18, in progress)

### D-4.0 Plan and order
Brief acceptance: viewer can follow any tracked actor; guessed regions are visibly marked;
retrieved images carry source/licence records. Order chosen (each builds on the previous):
1. **4.1 Honest confidence map for splats** (algorithm A-1, docs/ALGORITHMS.md) + the brief's
   "fill honesty" metric, measured on synthetic ground truth. Without it guessed regions are
   not marked at all for splat worlds (everything is `observed` 0.9).
2. **4.2 POV in seen space**: follow/first-person cameras prefer positions with high
   confidence; the follow camera stays on the observed side of the actor.
3. **4.3 Time-boxed progressive fill** (brief §6.2): gradient at once, improved fill within a
   configurable budget, background refinement swapped in.
4. **4.4 Retrieval** of open-licensed reference images with geometric verification and
   source/licence records — public outdoor scenes only; never private homes or people (so not
   the owner's room clips).
5. **4.5 Generative fill** for the remainder, tagged `generated`; model licence verified first.
Branch: `phase-4-pov-fill` from main + Phase 3 close-out.

### D-4.1 Confidence map A-1: measurements and selection protocol (2026-10-18)
Seed 11 (export from D-3b, 173,767 foreground Gaussians with opacity ≥ 0.05, 30 views; "bad" =
centre > 0.58 m (5 % of extent) from the true surface; 11 % are bad):

| Labelling | observed | bad flagged (recall) | observed that are good (precision) | Spearman |
|---|---|---|---|---|
| baseline: all observed 0.9 | 1.000 | 0.000 | 0.890 | – |
| count only | 0.928 | 0.215 | 0.907 | −0.21 |
| count + spread | 0.895 | 0.349 | 0.920 | −0.29 |
| + far from stereo (1 % extent) | 0.782 | 0.879 | 0.983 | −0.43 |
| + far from stereo (2 %) | 0.836 | 0.800 | 0.974 | −0.43 |
| + far from stereo (4 %) | 0.865 | 0.617 | 0.951 | −0.37 |
| + free-space violation | 0.874 | 0.534 | 0.942 | −0.35 |
| + far from stereo (2 %), only where stereo covers ≥ 3 views | 0.837 | 0.791 | 0.973 | −0.42 |

Walking clip (no ground truth): stereo is gappy (phone barely moved), so plain "far from stereo"
flags 42 % of Gaussians vs 11 % on seed 11 — it cannot tell a floater from a stereo hole. The
"covered" variant flags 27 %, free-space 12 %. Decision deferred to the protocol below.

**Protocol fixed before seeing seeds 12/13:** candidates = count+spread, +free-space,
+far-2 %-covered. On each of seeds 11, 12, 13 compute recall and precision. Choose the
candidate with the highest mean recall among those whose precision beats the baseline on every
seed and whose observed fraction is ≥ 0.75 on every seed. Ties (< 0.02 recall): prefer the one
that flags less on the walking clip (fewer false alarms on gappy stereo). Adopt A-1 only if the
chosen candidate beats the baseline on both honesty numbers on all three seeds.

Seeds 12/13 (exports made after the protocol was written; 177,677 / 140,324 Gaussians; bad =
6 % / 6.3 %):

| Candidate | recall 11 / 12 / 13 (mean) | precision 11 / 12 / 13 | observed 11 / 12 / 13 |
|---|---|---|---|
| baseline | 0 / 0 / 0 | 0.890 / 0.940 / 0.937 | 1 / 1 / 1 |
| count + spread | 0.349 / 0.470 / 0.275 (0.365) | 0.920 / 0.963 / 0.949 | 0.895 / 0.866 / 0.901 |
| + free-space | 0.534 / 0.677 / 0.362 (0.524) | 0.942 / 0.977 / 0.955 | 0.874 / 0.852 / 0.895 |
| + far 2 % where covered | **0.791 / 0.967 / 0.961 (0.906)** | 0.973 / 0.998 / 0.997 | 0.837 / 0.827 / 0.834 |

By the protocol: all three beat baseline precision on every seed and keep ≥ 0.75 observed; the
highest mean recall is "far 2 % where covered" → **adopted** as the A-1 default
(`relabel_gaussians`), used by `tools/splat_wreel.py --camera camera.json [--masks masks.npz]`.
Without a camera.json the builder keeps the old constant labels and says so.


### D-4.2 Follow views planned in seen space (algorithm A-2) (2026-10-18)
- "Follow person N" was a third-person offset behind the actor's facing. On the walking clip the
  people often walk towards the phone, so "behind" looks at parts of the room the video never
  saw. A-2 plans the follow path offline: per track sample, 36 candidates (18 yaws x distances
  0.9h / 1.3h, eye 1.15h, aimed at 0.55h), score = mean A-1 confidence of the front-most Gaussian
  per pixel of a 48x27 buffer (empty = 0), zero if the person is not in clear line of sight;
  Viterbi for smoothness. Stored as an `alternative` camera path + `directors-cut` preset: **no
  format change**. Built by `tools/splat_wreel.py --camera ...`.
- Walking clip, 60k Gaussians scored (opacity ≥ 0.2):

| Person (samples) | seen share: behind → planned | view-direction gap to nearest real camera, mean / p90 (deg): behind → planned |
|---|---|---|
| p1 (40) | 0.377 → 0.533 | 94.0 / 137.5 → 26.7 / 40.9 |
| p2 (31) | 0.324 → 0.513 | 81.1 / 130.7 → 35.0 / 45.5 |
| p3 (69) | 0.289 → 0.523 | 87.2 / 137.9 → 34.1 / 35.4 |

  The seen share is the planner's own objective; the view gap is independent of it and drops by
  ~55° on average. Clear line of sight for the baseline was 0.87–0.97. → **adopted** for splat
  worlds. Not yet measured on a synthetic scene with ground truth (needs a synthetic splat world
  with actors) — open item.
- Player: directors-cut presets that play an `alternative` path now say "Planned camera path
  that keeps to filmed areas." Verified in the player (WebGPU): person framed mid-shot at 15 s and
  17 s; background soft but tinted where weakly seen; no console errors.
- Planning cost: ~1.5 min for 140 samples on the laptop (numpy).


### D-4.3 Progressive plane fill (A-3): built, measured, NOT adopted (2026-10-18)
- Built: quick tier within a 5 s budget (measured 0.6–3.7 s incl. plane detection) + refined tier
  (1.3–15.6 s), stored as a separate Gaussian layer (fill LOD 0 = quick, LOD 1 = refined; the
  player shows LOD 0 then swaps — spec §6.1, no format change) + an `inferred` region record.
  Fill is `inferred`, confidence ≤ 0.45, so the player keeps marking it.
- Player bug found and fixed on the way: each Gaussian layer was sorted and drawn separately;
  splats do not write depth, so a later layer painted over everything (fill covered the TV on
  the walking clip). All scene Gaussian layers are now merged into one sorted set.
- Generator 0.2.0 records exact static geometry (`objects`: ground plane, axis-aligned boxes) in
  gt.json. Old seed 11–13 scenes could not be regenerated (camera paths differ by up to 13.8 m
  after generator changes), so seeds 11/12/13 were re-rendered as "v2" and re-exported on Kaggle
  (camera ATE 0.2–0.7 % of path length, rotation 0.7–2.8°).
- Exact-geometry results (r = 2 % of extent): see docs/ALGORITHMS.md A-3. Coverage of what the
  eval views see rises only 0.4–6.5 points. Fill inside objects ≈ 0.
- Root cause on seeds 11v2/12v2 is the **reconstruction**, not the fill: on seed 12v2, 89 % of
  observed splats lie below the true ground (median −0.29 m; camera 4 m up, oblique view); seed
  11v2 ground +0.19 m and only 46 % of observed splats within 2 % of the true surface. This is a
  new finding about splat accuracy that the earlier image-based metrics (PSNR/SSIM) did not show;
  it is an open problem for the reconstruction stage, recorded here, not fixed.
- Decision: fill is opt-in (`--fill`) until the reconstruction bias is fixed and fill passes ≥ 80 %
  precision on all three exact-geometry scenes. Brief §6.2 (gradient → improved fill → refined
  swap) is implemented and demonstrable, but not switched on by default because it is not yet
  honest enough.

### D-4.3b Reconstruction ground bias: cause and partial fix (2026-10-18)
- Where: classical stereo points showed the same offset as the splats (+0.28 m seed 11v2,
  −0.32 m seed 12v2), true intrinsics in use → upstream of splat training.
- What: the reconstructed ground is tilted 3.2° / 1.4° and the offset grows with distance
  (seed 11v2: +0.13 m at 0–5 m, +0.73 m at 20–60 m). Per-camera orientation errors after
  alignment are one constant rotation (sd 0.06–0.23°, axis consistency 0.79–0.99) while
  frame-to-frame rotation error is tiny (0.08°) and per-view depth is good (AbsRel 3.7 %): the
  solve misjudges the direction of travel relative to where the cameras point on these short,
  straight, high (3–4 m) camera paths. Aligning by orientations instead of positions raises
  stereo-on-surface from 37 % → 61 % (11v2) and 27 % → 58 % (12v2); a smaller offset remains.
- Tried: more bundle adjustment (ba_max_nfev 60 → 400, periodic BA every 5): 2.55° → 2.28°,
  5x slower → rejected. Longer feature tracks (`match_window`, frames each frame is matched
  against): full pipeline on the three v2 scenes, stereo on true surface / benchmark failures:

| match_window | 11v2 | 12v2 | 13v2 | mean | threshold failures |
|---|---|---|---|---|---|
| 3 (old default) | 0.432 | 0.364 | 0.973 | 0.590 | rot (11v2), ssim ratio (13v2) |
| **8** | 0.523 | 0.925 | 0.876 | **0.775** | **none** |
| 12 | 0.482 | 0.645 | 0.932 | 0.686 | ssim ratio (12v2, 13v2) |

  Original seeds 11/12/13 with 8 vs 3: identical pass/fail (seed 13's novel-view failure was
  already there), rotation 1.46→1.29, 0.29→0.39, 0.75→0.96°. → **default 8 adopted**.
- Not solved: seed 11v2 keeps a +0.20 m ground offset; seed 13v2 got slightly worse. Results
  vary a lot between scenes with three samples; a stronger fix (e.g. global rotation averaging
  before bundle adjustment, or a level-ground / gravity prior from phone sensors) is a roadmap
  item, not done here.
- Re-export on Kaggle with `match_window` 8 (seeds 11v2/12v2/13v2): splats on the true surface
  0.39 → 0.75 / 0.31 → 0.55 / 0.90 → 0.86; eval-view surface covered by observed splats
  0.58 → 0.85 / 0.48 → 0.78 / 0.75 → 0.72; ground tilt 3.24 → 1.10° / 1.44 → 1.36° / 0.89°.
  Seed 12v2 keeps −0.23 m on Kaggle although the laptop run gave −0.07 m (different OpenCV
  build, as in D-3b.6).
- Fill (A-3) re-test, share on the true surface quick / refined: 0.24 / 0.29, 0.22 / 0.33,
  0.996 / 0.993 → still below the 0.8 bar on two of three scenes. A-3 stays experimental and
  opt-in. Where the reconstruction is accurate (13v2) the fill is accurate; the fill does not add
  error of its own, it inherits the reconstruction's.


### D-4.4 Licensed reference images with geometric verification (A-4) (2026-10-18)
- APIs verified from official docs before use: Openverse `GET /v1/images/` (`q`, `license`,
  `category`, `mature`; results `url`, `license`, `license_version`, `foreign_landing_url`,
  `attribution`), MediaWiki geosearch generator + `imageinfo/extmetadata` on Commons
  (`LicenseShortName`, `LicenseUrl`, `Categories`), Wikimedia User-Agent policy (contact
  required). Network retrieval is implemented but **off**: it refuses to run without a contact
  string, which the owner has not given; it is only for public outdoor places, never homes. The
  clients are tested offline against responses built from the documented fields.
- Licences: CC0, PDM, CC BY, CC BY-SA only. ND refused (a reconstruction is a derivative), NC
  refused (non-commercial use cannot be guaranteed), unknown / multi-licensed refused.
- People filter: metadata words (people, portrait, selfie, crowd, interior, home, ...) and the
  YuNet face detector (MIT, OpenCV model zoo; OpenCV 5 dropped the Haar cascades and HOG people
  detector from the main module, so they were not available). Missing model = hard error.
- Verification: SIFT vs every registered frame → 2D-3D through the observed geometry's depth in
  the best frame → PnP RANSAC over a focal grid → >= 40 inliers, ratio >= 0.25, median <= 2 px,
  confirmed in a second frame (>= 20), >= 5 % of inliers off the dominant plane, colour
  agreement >= 0.3. Results in docs/ALGORITHMS.md A-4: 12/12 accepted, 24/24 rejected.
- Use: accepted photos recolour low-confidence Gaussians they see (colour averaged over each
  element's projected footprint — single pixels carried texture noise, 23 vs 15 error), class
  `retrieved`, confidence 0.5; one region per photo with `sources: [{kind: user|retrieval, url,
  license, verified: geometric}]`; attribution in `extensions["org.worldreel.credits"]`.
  Colour vs plane-mean baseline on fill elements: better on 12v2 (13.2 vs 26.7) and 13v2
  (14.4 vs 15.1), worse on 11v2 (19.5 vs 16.6, where the fill itself is off-surface).
- Acceptance check (brief Phase 4: "retrieved images carry source/licence records"): synthetic
  seed 13v2 world built with 4 true + 1 wrong CC0 reference → 4 verified, 1 rejected, 18,469
  Gaussians recoloured, 4 `retrieved` regions with url + licence + `verified: geometric`, file
  valid. `tools/splat_wreel.py --camera ... --frames ... --references DIR`.
- Open: no real-photo test yet (needs a public outdoor clip and photos of it); thresholds are
  in-sample; face detector not yet measured on real faces.


### D-4.5 Generative fill (A-5, LaMa): built, measured, opt-in (2026-10-18)
- Model chosen for a verifiable licence and CPU use: LaMa (Apache-2.0 code and weights), ONNX port
  `Carve/LaMa-ONNX` lama_fp32 (Apache-2.0) via onnxruntime (MIT); I/O ranges measured, not
  assumed (docs/MODELS.md). Stored in git-ignored `data/models/`; `fill` optional extra.
- Use: textures of A-3 planes; classical pre-fill of every unknown cell, LaMa only for real holes;
  the refined fill tier takes the generated texture (class `generated`, confidence 0.2), the quick
  tier keeps the plane mean; region `r_fill_generated` with `sources: [{kind: model, ...}]`.
- Results in docs/ALGORITHMS.md A-5: texture realism matches the real surface (spread 0.91–1.15)
  but colour accuracy is worse than the plane mean (local error 38–43 vs 15–27). Off by default,
  like A-3, which it builds on. 18,582 elements textured in 42 s on seed 13v2; verified in the
  player (WebGPU, refined LOD loaded, purple "generated" tint, no console errors).
- A generative model invents: in the first test it erased half of a box inside a hole. This is
  why generated content is always labelled and fogged, never shown as observed.

### D-4.end Phase 4 close-out (2026-10-18)
Final checks on `phase-4-pov-fill`: pytest 200 passed; ruff + ruff format clean (124 files);
mypy clean (101 files); `formats validate` sample OK; player vitest 28 passed; player build OK.

Acceptance (brief §Phase 4):
- **"Viewer can follow any tracked actor"** — met. Every tracked person has "Follow" (A-2, planned
  in seen space: view gap to the real camera 81–94° → 27–35° on the walking clip) and "eyes"
  views; the follow camera scales with the person's height.
- **"Guessed regions are visibly marked"** — met. A-1 confidence map adopted (bad splats flagged
  0.79–0.97 vs 0 before, observed-precision 0.97–0.998 on seeds 11–13); fill, retrieved and
  generated content carry their classes and the player fogs everything not observed.
- **"Retrieved images carry source/licence records"** — met. A-4: licence allow-list, people
  filters, geometric verification (12/12 true references accepted, 24/24 wrong ones rejected on
  exact-geometry scenes); each used image has a region with url + licence + `verified:
  geometric`, attribution in `extensions["org.worldreel.credits"]`.
- Time-boxed progressive fill (A-3) and generative fill (A-5) are built and demonstrable but
  **experimental, opt-in**: fill is only as accurate as the reconstruction under it (24 % / 29 % /
  99 % on the true surface, bar 80 %); generated texture is realistic (spread 0.9–1.15) but less
  accurate than a flat colour.

Also in this phase: player fix (all Gaussian layers sorted together); generator 0.2.0 records
exact geometry; reconstruction tilt found and reduced (`match_window` 3 → 8: splats on the true
surface 0.39 → 0.75 and 0.31 → 0.55 on the two hard scenes).

Known limits carried forward: residual reconstruction tilt on straight high camera paths
(seed 11v2 +0.15 m); real-photo retrieval untested (no public outdoor clip; network off without a
contact); verification and filter thresholds chosen in-sample; YuNet not measured on real faces;
follow planner not yet checked on synthetic ground truth.

Owner's manual steps:
1. To try web retrieval: film a public outdoor place (no identifiable people), give its name or
   coordinates, and a contact (email or web page) for the Wikimedia/Openverse User-Agent.
2. Models live in `data/models/` (git-ignored): YuNet (MIT, 232 KB) and LaMa (Apache-2.0,
   208 MB). Delete them to free space; the code says what to re-download.
3. Unused clips are still in `data/inbox/` (vide.mp4, approved.mp4, aprove.mp4).

### D-4.4b First real-world retrieval test: Vasai Fort (2026-10-18)
- Footage: a drone montage of Vasai Fort (a public historic site) provided by the owner, who
  states they have the creator's permission (creator name pending; watermark "MUMBAI ONE").
  Only one continuous piece was usable: the gate arch, 53.0–55.07 s (31 frames at 15 fps after
  removing a cut), cropped to 1135x548 to remove letterbox bars and the watermark. Processed on
  the laptop only; nothing uploaded. Intrinsics unknown: horizontal FOV assumed 64° (≈70° drone
  lens before the crop) — an assumption.
- Classical reconstruction: 31/31 frames registered, reprojection 0.46 px, 5,078 SfM points.
- Search (live, User-Agent with the owner's contact from the git-ignored local config): Commons
  "Vasai Fort" + "Bassein Fort" → 46 licence-compatible files (CC BY 2.0, CC BY-SA 3.0/4.0) after
  the metadata filter; downloaded at 1,280 px to scratch only (not the project).
- Result: 4 skipped by the face detector; 42 rejected by geometric verification (≤ 32 SIFT
  matches, 2 with no consistent PnP pose); **0 accepted**. Safe (no wrong image used), but it does
  not demonstrate a real positive: a 2 s top-down drone view of one gate shares almost no
  viewpoint with ground-level photos of other parts of the fort. A real positive needs
  ground-level footage of a spot people photograph (main gate, a long wall).
- Fix: Commons URLs carried `utm_*` tracking parameters; records now keep the bare file URL.

## Phase 5: Audio (2026-10-18, in progress)

### D-5.0 Plan and order
Brief acceptance: footsteps change with surface and weather; first-person audio differs from
third-person; stems export separately. Order (each builds on the previous):
1. **5.1 Sound events + procedural sound engine.** Footstep events from actor tracks (step times
   from speed / gait), attributes from the spec's open vocabulary (`surface`, `wet`, `footwear`,
   `weight`, `speed`, `weather`). Sounds are *generated* (numpy DSP, seeded) — no sample
   libraries, so no licensing risk (brief §7). Each distinct sound is rendered once into a small
   clip referenced by `attributes.clip` (the engine's vocabulary; no schema change).
   Measured: spectral features (centroid, decay, noise share) differ by surface and weather.
2. **5.2 Layers and stems:** foley (footsteps), ambience (weather + tension), music (procedural
   mood bed driven by the tension curve), effects; dialogue empty (no voices yet). Each layer is
   its own track; `tools` export them as separate WAV stems plus the mix.
3. **5.3 Mixing:** loudness measurement and target per spec (`mix.loudness_target_lufs`);
   ducking hook for dialogue. Library or own implementation decided after checking licences.
4. **5.4 Player audio:** Web Audio playback of tracks, positional event clips (panner),
   listener at the camera in third person / free, at the actor's head in first person; mood live.
   Measured: offline render of the same scene from both listener positions differs (level and
   left/right balance), plus player unit tests of listener placement.
Not in this phase: voices, dubbing, diarization (no dialogue source yet); subtitles stay a slot.
Branch: `phase-5-audio` from main eee6fab.

### D-5.1 Procedural footsteps (A-6): measured (2026-10-18)
- `stages/audio/synth.py`: generated footsteps (heel + toe transients x surface response,
  footwear shaping, rain/wet splash), deterministic per seed, numpy + scipy only — no sample
  library, nothing to license. WAV via the standard-library `wave` module.
- Measurement `eval/footsteps.py`: 7 surfaces x {dry, rain} x 3 seeded variants. The first
  measure (4 scalar features: centroid, decay, share > 3 kHz, onset count) could not separate the
  conditions — too coarse; replaced by the spectral-temporal profile (energy in 9 octave bands
  63 Hz–16 kHz + decay to −10/−20 dB). Criterion unchanged from the design note: a condition is
  distinguishable when its nearest other condition is farther than twice its own variant spread.
- Iterations (all measured): splash scaled to the impact energy (a fixed splash made every wet
  surface alike); stone given a ringing resonance, concrete a crisp slap (they were one design);
  hard surfaces splash, soft ones (sand, dirt, grass) squelch; variant randomness narrowed from
  ±20–50 % to a realistic ±3–10 % (it was blurring every condition); sand made fine-grained
  (1–4 kHz) instead of low (it overlapped dirt).
- Result: **11 / 14 conditions distinguishable** (wood, stone, gravel, grass, dirt in both
  weathers; concrete dry; gravel rain). Not met: concrete/rain (margin 2.0, bar > 2), sand/dry
  (1.0), sand/rain (1.3). Rain lengthens the tail on 7/7 surfaces and raises high-band energy on
  6/7 (sand, already bright, gains a low squelch instead). Sand is the weak spot; recorded, tuning
  stopped to avoid fitting the test.
- Clips for listening are written by `python -m eval.footsteps --wav-dir DIR`.

### D-5.2 Footstep events from actor tracks (2026-10-18)
- `stages/audio/events.py`: a step each time the horizontal distance walked reaches one stride;
  cadence 1.8 steps/s at 1.4 m/s, rising to 3/s when running (approximate human gait — an
  assumption, not measured here); scene units → metres via the actor's height (1.75 m assumed,
  as in packaging). Standing still (< 0.2 m/s) makes no steps.
- Events use the spec's open attribute map: `surface`, `weather`, `wet`, `footwear`, `weight`,
  `speed` (bucketed to 0.5 m/s, min 0.5, and 10 kg so similar steps share clips) and `clip` (path
  of the generated WAV in the `.wreel`, `audio/fx/step_<hash>_<variant>.wav`, 3 variants used in
  turn). No schema change. Scene-wide surface/weather from package config `audio`.
- Bug found by the validator: when speed dropped between track samples, a step could land before
  its segment (t = −0.235 s); positions are now clamped into the segment (+ test).
- End to end (synthetic 2-walker scene, gravel + rain): 9 footsteps (4 + 5 over 2.5 s, as the
  cadence predicts), 4 clips (159 kB), file valid.

### D-5.3 Layers, stems and mood music (2026-10-19)
- `stages/audio/beds.py`: generated ambience (wind with tension-deeper gusts; rain = droplets +
  hiss) and mood music (own procedural scheme: major pad crossfading to minor, drone and high
  shimmer growing with tension, a pulse from tension 3 whose tempo rises 60 → 140 bpm).
  `stages/audio/stems.py`: foley = footstep clips placed at event times; each non-empty layer is
  a separate mono WAV track (`audio/stems/<layer>.wav`, licence `synthetic`, non-positional);
  dialogue/effects have no source yet, so no track. Mood can be set per scene in package config.
- `tools/export_stems.py`: one WAV per layer + `mix.wav` (fixed gains) + `stems.json`. Walker
  scene: ambience, foley, music, mix — **"stems export separately" met**.
- Mood measurement `eval/mood.py` (20 s per tension, measured from the audio; tempo from the
  onset envelope's autocorrelation, independent of the generator):

| tension | 0 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| level (dBFS) | −20.7 | −20.7 | −21.0 | −20.4 | −19.0 | −17.7 |
| centroid (Hz) | 316 | 316 | 308 | 336 | 411 | 447 |
| tempo (bpm), measured / intended | – / none | – / none | – / 71 | 94.0 / 94.3 | 117.5 / 117.1 | 139.7 / 140 |

  Louder, brighter and faster with tension; weak spot at 4 (pulse too quiet to detect, 0.3 dB dip).
- Fixed along the way (all measured): per-render peak normalisation made the music *quieter* as
  tension added layers → fixed gains; a 55 Hz drone made it *darker* → drone an octave up plus a
  high shimmer; linear major→minor crossfade lost ~3 dB halfway → equal-power; the tension
  smoothing was an O(n·k) convolution (minutes for 20 s) → running sum (0.02 s); the tempo
  detector reported 144 bpm for pulse-free pads (window aliasing on the 55 Hz chord pattern) →
  energy low-passed at 8 Hz before sampling.

### D-4.4c Owner's own fort photos; unused clips deleted (2026-10-19)
- At the owner's request all five unused clips were deleted from `data/inbox/`, including the
  drone montage: its creator is no longer known, so it cannot be credited and was not used.
  Everything derived from it (gate frames, the 2 s reconstruction, previews) was deleted too;
  D-4.4b's numbers remain as a record of what was run.
- The owner added 14 of their own photos of Vasai Fort (gate wall, arch courtyard, church
  façade, nave, tombstones; several show other visitors — kept on the laptop only). As a set they
  cannot be reconstructed (1–3 near-identical shots per spot, strong sun flare).
- 2D check instead (fundamental-matrix RANSAC, no 3D): 4 distinct spots vs the 46 Commons
  candidates. Best: 17 inliers (façade vs Bassein-Fort-11.jpg), 12–17 for the others — below
  what a reliable match needs (~40); **no real positive match**. Likely: different parts / angles
  of the fort on Commons, and flare / haze in the owner's photos.
- Still needed for a real-world positive: a continuous ground-level clip of one spot (for 3D),
  ideally a spot that open-licensed photos also show.

### D-5.4 Mixing, loudness and player audio; first vs third person (2026-10-19)
- Loudness: ITU-R BS.1770-4 integrated loudness via `pyloudnorm` 0.2.0 (MIT, licence read from
  the installed package; reference check: a 1 kHz sine at −6 dBFS peak reads −9.07 LUFS). The mix
  is normalised to the scene's `audio.mix.loudness_target_lufs` (default −16, the spec's example
  value — not a claim about any platform's recommendation) and peak-limited to 0.99; tested at −23
  and −16 LUFS within 0.5 LU. `tools/export_stems.py --target` writes other targets (phone,
  headphones, speakers) on demand.
- Ducking: music and ambience drop ~10 dB under dialogue (50 ms attack, 500 ms release; tested with
  a synthetic speech-like burst — there is no dialogue source yet).
- Player (`player/src/audio.ts`, Web Audio): stems play in sync with the clock (the pre-mixed foley
  stem is skipped when positional footsteps exist); footstep events play through HRTF panners at
  their positions, scheduled 0.25 s ahead; the listener is the current view's eye — the
  character's head in first person, the camera otherwise. A "Sound" button starts audio (browsers
  require a user gesture). Verified in the browser pane: audio decoded and running, first-person
  view, seek and pause/resume with no console errors (listening itself was not possible there).
- Acceptance "first-person audio differs from third-person": player unit tests (own step > 3 dB
  louder through the walker's ears; another person heard > 10° from a different direction) and
  `eval/pov_audio.py` on the synthetic two-walker world, mirroring the player's listener rules and
  distance model:

| listening as | own steps: level diff (first − third) | other person's steps: level / direction diff |
|---|---|---|
| p1 (4 own, 5 other steps) | +3.14 dB | 1.64 dB / 12.7° |
| p2 (5 own, 4 other steps) | +3.17 dB | 1.71 dB / 14.9° |

  (The own-step direction differs by 180°: through one's own eyes the step is just behind / below
  the head, from the follow camera it is ahead. With a source almost directly below, azimuth is
  not very meaningful; the level difference is the robust number.)

### D-5.5 "Like a thousand bees": owner's listening test failed; sound redesigned (2026-10-19)
- The owner listened in the player and described the result as "a thousand bees screaming". Not
  pushed. Measured per stem (walker scene, rain, tension 1→9): ambience 35 % of energy above
  4 kHz (rain: ~400 bright 2–12 kHz clicks/s + hiss to 9 kHz), footsteps 44 % above 4 kHz (bright
  gravel grains + spray), mix 25 % above 4 kHz; the music was pure-sine chords whose crossfade
  sounded a major and a minor third together.
- Targets fixed before the redesign: energy above 4 kHz ≤ 5 % music, ≤ 10 % ambience, ≤ 20 %
  footsteps, ≤ 10 % mix; music roughness ≤ 0.15.
- Roughness measure (`synth.harshness`), corrected twice on reference sounds: a share-of-flutter
  version scored a steady sine 0.999; a full-band depth version rated a smooth low chord as rough
  (its waveform repeats at 27.5–55 Hz — heard as pitch). Final: depth of 15–100 Hz loudness flutter
  within 1–8 kHz (steady 2 kHz sine 0.001, 80 % 30 Hz tremolo 0.56, noise 0.074).
- Redesign: soft rain (low-passed wash + ~60 dull drops/s); footsteps low-passed at 6 kHz, gravel
  grains and spray at 1.5–5 kHz; music = warm detuned pad (band-limited saw voices, low-passed
  900 Hz) with a third that *glides* major→minor, a low drone and a soft heartbeat-like pulse
  (60 → 120 bpm) — no pure sines, no high beeps.
- Result (20 s, rain + gravel, tension 1→9): above 4 kHz — ambience 0.3 %, footsteps 13 %, music
  0 %, mix 3 % (was 25 %); music roughness 0. All targets met. Trade-offs: footstep conditions
  distinguishable 10/14 (was 11/14; gravel in rain dropped); music is now *darker* with tension
  (heavier low pulse) instead of brighter — tempo still tracks tension (86 / 103 / 120 bpm at
  tension 6 / 8 / 10, as intended).
- Objective checks are not a listening test: 20 s clips in `data/listen/` (git-ignored) go to the
  owner; Phase 5 is not closed or pushed until the owner is happy with how it sounds.

### D-5.end Phase 5 close-out (2026-10-19)
Final checks on `phase-5-audio`: pytest 218 passed (1 skipped: LaMa model not downloaded); ruff +
format clean; mypy clean (113 files); `formats validate` sample OK; player vitest 35 passed;
player type-check and build OK. **Owner's listening test of the redesigned mix: "sounds fine".**

Acceptance (brief §Phase 5):
- **Footsteps change with surface and weather** — met: generated footsteps for 7 surfaces x
  dry/rain, 10/14 conditions measurably distinct by the pre-set criterion (sand and wet concrete /
  gravel are the weak spots); rain lengthens every step and changes its spectrum; steps are
  triggered from actor tracks at a speed-dependent cadence.
- **First-person audio differs from third-person** — met: the player's listener is the view's
  eye (character's head vs camera); own steps +3.1 dB louder through the character's ears,
  others' steps 1.6–1.7 dB / 13–15° different (walker scene, D-5.4); unit-tested.
- **Stems export separately** — met: ambience, music, foley as separate tracks; export tool
  writes one WAV per layer + a loudness-normalised mix (BS.1770, pyloudnorm MIT).
Also: mood music follows tension (louder, heavier, tempo 86 → 120 bpm); dialogue ducking
(tested with a synthetic burst); harshness checks guard against the "bees" regression (D-5.5).

Not done (no source yet): dialogue / voices, dubbing, diarization, subtitles content; live
tension changes in the player only affect visuals (music is pre-rendered from the mood curve).

Owner's manual steps:
1. Listen with headphones in the player ("Sound" button) to hear first- vs third-person
   positioning; short looping test scenes restart their sound at each loop.
2. Scene surface / weather / mood are set in the package config, e.g.
   `[stage.package] audio = {surface = "gravel", weather = "rain", mood = [...]}`.
3. Listening clips are in `data/listen/` (git-ignored) — delete when done.
4. Still open from Phase 4: a ground-level Vasai Fort clip for a real retrieval match; your 14
   fort photos are in `data/inbox/` (they show other visitors — keep them local).

## Phase 6: `.wedit` export and flat MP4 (2026-10-19)

### D-6.0 Plan, tools and order
Brief acceptance: a `.wedit` opens in at least one common editor with stems, layers and timeline
intact. Facts checked before planning:
- Editor: the owner's laptop (Iris Xe) is too weak for DaVinci Resolve; the owner installed
  **Shotcut 26.9.27** (official download page: current version, Windows 10 build 1809+, no
  hardware requirement listed) at `D:\Shotcut`. It ships `melt.exe` 7.41.0 (MLT command-line
  renderer) and `ffmpeg.exe`. Shotcut's feature page: it creates / loads / renders **MLT XML**
  projects and *exports* CMX3600 EDL; OpenTimelineIO import is not listed.
- OpenTimelineIO 0.18.1 (PyPI JSON: Apache-2.0, Python 3.9–3.13). Installed adapters:
  `otio_json`, `otioz`, `otiod` only — EDL / FCPXML / MLT adapters are not built in (not used).
Decisions:
1. **6.1 Spec first:** `.wedit` 1.1 adds timeline format `"mlt"` (MLT XML, what Shotcut opens).
   OTIO stays the reference timeline; both are written.
2. **6.2 Offline renderer** (Python, CPU, numpy): points, Gaussian splats and actor avatars along
   a camera path → RGB frames + depth + per-actor masks (the layers the spec asks for). The
   browser player cannot write files, and a CPU renderer is testable and runs anywhere.
3. **6.3 Flat MP4:** frames + loudness-normalised mix → H.264/AAC via ffmpeg; the AI label is
   written to the file metadata and burned in small in a corner (spec §3: SHOULD).
4. **6.4 `.wedit` bundle:** video, stems (from the `.wreel`), layers (16-bit PNG depth with its
   scale in the manifest — monocular scenes have no metric unit; PNG masks), `main.otio`,
   `main.mlt`, signed `wedit.json`, `PROVENANCE.txt`.
5. **6.5 Verification:** schema + signature validation; OTIO read-back; `melt` renders the MLT
   timeline headless and the result is checked (duration, every audio stem present, frames);
   then the owner opens it once in Shotcut.

### D-6.1 – D-6.4 Built (2026-10-19)
- **6.1** `spec/wedit.md` + schema → 1.1 (timeline format `"mlt"`, media type
  `application/xml`); 1.0 bundles stay readable.
- **6.2** `stages/export/render.py`: CPU renderer (Numba 0.68.0, BSD) mirroring the player — EWA
  splats (+0.3 px² low-pass, 3σ), front-to-back compositing, same fog rule and gradient; depth is
  the median depth (where transmittance crosses 0.5), actors drawn as z-buffered points with ids.
  Measured: ~0.7 s per 640×360 frame on this laptop. 4 unit tests.
- **6.3** `stages/export/flat.py`: rgb24 piped to ffmpeg libx264 (yuv420p, CRF 20) + AAC mix at the
  world's loudness target; label "AI-RECONSTRUCTED - not a recording" burned in top-left and
  written to `comment` / `description` metadata. Walker world: 60 frames, h264 + aac, 8 s.
- **6.4** `stages/export/wedit_export.py`: one render pass writes the video, `layers/depth`
  (16-bit PNG, scale in `encoding`), `layers/mask_<actor>`, `stems/<layer>.wav`,
  `timeline/main.otio` and `timeline/main.mlt` (layer tracks present but hidden).

### D-6.5 Verification with Shotcut's `melt` 7.41.0 (2026-10-19, measured)
Walker world (`walkers_audio.wreel`, camera `director`, 60 frames 854×480):
- Bundle validates: 0 errors, signature valid, 186 files.
- Bug found: MLT resolves paths relative to the XML file → all timeline paths now `../…`
  (OTIO too). Bug found: the main clip's own mix plus the stems would play the sound twice →
  the main track has `hide="audio"`; the stems are the timeline's sound.
- `melt main.mlt` render: 60/60 frames; PSNR vs `video/main.mp4` 33.9 dB (hidden layers do not
  leak into the picture). Depth sequence loads as a producer (60 frames).
- Audio rendered at 44.1 kHz mono: least-squares gains per stem ambience 1.00, foley 1.00,
  music 1.00, correlation 1.00 with the plain stem sum — every stem present exactly once.
  (A first check at melt's default 48 kHz looked wrong, 0.56, only because of the rate change.)
- `tests/test_wedit_export.py`: validity, OTIO read-back, MLT paths, melt stem gains (skips
  without melt). Remaining: the owner opens `timeline/main.mlt` once in Shotcut.

### D-6.6 Shotcut opened the timeline as one clip; MLT now in Shotcut's project layout (2026-10-19)
Owner's check: Shotcut opened `main.mlt` in the Source viewer as a single clip with an empty
Timeline. Shotcut's source (`src/mltcontroller.cpp`, `src/shotcut_mlt_properties.h`,
`src/models/multitrackmodel.cpp`, master) shows why: a file is a multitrack project only when the
tractor has the property `shotcut`; tracks carry `shotcut:video` / `shotcut:audio` and
`shotcut:name`; track 0 is the `background` playlist (black `color` producer); video tracks are
blended with `qtblend`, every track has a `mix` transition. The export now writes that layout.
Layer tracks sit *under* the main picture instead of being hidden (Shotcut classifies tracks
partly by their `hide` bits, and I did not confirm how a hidden video track is classified); main
covers them, so hide or move main to see a layer.
Measured with melt (walker world): 60/60 frames, PSNR vs `video/main.mp4` mean 39.4 dB, min 38.9
(melt output at CRF 12); stem gains 1.00 / 1.00 / 1.00. Not yet confirmed in the Shotcut window.

### D-6.7 Shotcut showed "INVALID": layer sequences failed to load (2026-10-19)
Owner's second check: the timeline opened (D-6.6 worked) but the viewer showed MLT's "INVALID"
placeholder. Shotcut's log: `producer_xml failed to load producer ".../layers/depth/%06d.png?begin=0"`
(and both masks). The `?begin=0` query hides the `.png` extension, so the loader cannot pick a
service. Fix: `mlt_service=qimage` (Shotcut's `qt6/producer_qimage.yml`: "%" in the filename = image
sequence, `begin` = starting number) with `begin=0` as a property. Measured with melt, each layer
rendered alone (others hidden): depth, mask_p1 and mask_p2 load, no load errors; main picture and
stem gains unchanged (PSNR 39.4 dB, gains 1.00 / 1.00 / 1.00).

### D-6.8 Layers covered the main picture in Shotcut; main is now V1, layers hidden above (2026-10-19)
Owner's third check: the depth layer showed instead of the main picture, without the owner hiding
anything. Evidence: melt rendering Shotcut's own autosave of the project gave PSNR 4.6 dB against
`main.mp4` (my file: 39.4 dB). The one change that mattered: Shotcut sets `disable=1` on the bottom
video track's blend (its V1 is the opaque base); re-enabling it restored main (25.9 dB, single PNG
frame). In that state blends that all composite onto track 0 do not show either.
Fix: main is V1; depth and masks sit above it, hidden (`hide="video"`, the eye button in
Shotcut); each video blend composites onto the video track below (`a_track` = previous video
track). Measured with melt, both as written and with Shotcut's V1 `disable` applied: layers hidden →
output identical to main; depth un-hidden → depth on top (26.3 dB vs the 8-bit depth, 6.0 vs
main). Full render unchanged: PSNR 39.4 dB, stem gains 1.00 / 1.00 / 1.00.

### D-6.end Phase 6 close-out (2026-10-19)
Final checks on `phase-6-wedit`: pytest 227 passed (1 skipped: LaMa model not downloaded); ruff +
format clean; mypy clean (119 files); player vitest 35 passed; player build OK.
**Owner's Shotcut 26.9.27 check of `walkers_v4.wedit`: the timeline opens with its tracks, the
main picture (with the AI label) plays, and switching on `mask_p2` shows the walker's mask.**

Acceptance (brief §Phase 6):
- **A `.wedit` opens in at least one common editor with stems, layers and timeline intact** —
  met in Shotcut (owner, three fix rounds D-6.6 – D-6.8) and checked headless with its `melt`:
  picture PSNR 39.4 dB vs the exported video, every stem mixed once (gains 1.00), every layer
  loads and shows when un-hidden. OTIO timeline written and read back; not tested in an
  OTIO-importing editor (none available on this laptop).
- **Flat MP4 of one camera path** — met: H.264 + AAC, loudness-normalised mix, AI label burned in
  and in the metadata. `python -m tools.export_video world.wreel --out clip.mp4`.

Limits: CPU rendering ~0.7 s per 640×360 frame (a 1-minute 854×480 export takes roughly 20 min);
the export is "flat" level only (no re-import of edits into a `.wreel`).

Owner's manual steps:
1. Export your own: `python -m tools.export_video world.wreel --out my.wedit --wedit`, then open
   `my.wedit/timeline/main.mlt` in Shotcut. Layers are hidden tracks: click a track's eye to show
   it. Shotcut may ask to save changes on close — it only adds its own bookkeeping.
2. `data/exports/walkers.wedit`, `walkers_v2`, `walkers_v3` are broken earlier attempts and
   `walkers_v4` is the good one (all git-ignored); delete the old ones when you like.

---

## Phase 7: Script brain and character bible (2026-10-19, in progress)

### D-7.0 Plan and order
Brief acceptance: a 10-minute script loads scene by scene with a continuity state; character
distinctness checks pass. Plus the brief's "fine-tuned script model". Starting point (Phase 1):
`.wscript` 1.0 container, lossless `.wsl` scene language, validator, `load_context` (current scene +
one-line summaries of the others). Gaps: continuity is stored per scene but nothing can *check* it,
because what happens inside a scene is free text; no long script; no character bible; token savings
never measured (D-1.6).
1. **7.1 Continuity (A-7):** spec 1.1 adds a state-change line `= <ID> key=value …` to `.wsl`.
   A tracker replays scenes: end state = start state + changes; the next scene's start must match
   unless it lists the difference as `offscreen` (time jumps). Also checked: speakers are present
   in the scene's location, a prop is in one place at a time. 1.0 files stay readable and valid.
2. **7.2 A 10-minute script:** an original script written for this project (no copyrighted text).
   "10 minutes" needs a measurable definition: an estimator (dialogue words at a speaking rate +
   fixed time per action / sound / POV line), recorded as an assumption, checked against the
   1-page-per-minute convention on the rendered screenplay.
3. **7.3 Token measurement:** scene context vs whole script vs a plain screenplay (IDs expanded),
   with a real tokenizer (verified library, exact version), per scene and in total.
4. **7.4 Character bible (A-8):** identity code = seed → appearance vector + voice parameters,
   stored in the file; distinctness = minimum pairwise distance ≥ threshold; a sampler that keeps
   characters apart; a collapse counter; a style-request parser that accepts attribute words only,
   so a named real person cannot be requested. Face / voice *embeddings of rendered output* need a
   generator (Phase 8); Phase 7 checks distinctness in the identity-parameter space and says so.
5. **7.5 Script model:** no local GPU (D-0.1), so fine-tuning runs on Kaggle (T4) behind a config.
   "Soup" is still unverified (D-0.8); it is checked against its docs first, otherwise a verified
   library is used. Model, licence and training data are chosen and recorded before any training.
6. **7.6 Close-out.**

### D-7.1 `.wscript` 1.1 and the continuity tracker (A-7) (2026-10-19)
- Spec 1.1: `.wsl` line `= <ID|world> key=value|key+=value|key-=value …` (keys and operators in
  spec §4.2, validated); continuity keys `time` and `offscreen`. 1.0 files stay valid.
- `stages/generate/continuity.py` (the script feeds the `generate` stage): `replay` (end state
  from start + changes; `speaker_absent`, `prop_in_two_places`, `prop_not_held`), `check`
  (scene n end vs scene n+1 start, `continuity_break` unless listed in `offscreen`), `propagate`
  (authors write the first state and the changes; later start states are filled in).
- Tests: every check fires on a hand-made violation; no issue on the clean script; round trip
  of `=` lines is byte-exact. Detection rate on a long script: next, with the 10-minute script.

### D-7.2 The 10-minute script and continuity measured (2026-10-19)
- **Script:** "The Lantern Keeper", original, written for this project
  (`tools/make_sample_wscript.py` → `eval/scripts/lantern_keeper.wscript`, directory form, committed
  as text). 10 scenes, 4 characters, 5 locations, 7 props, dusk → night → dawn → two days later
  (two declared `offscreen` time jumps). Only scene 1's state is hand-written; the rest is
  filled in by `propagate`. Validates with 0 errors, 0 warnings.
- **Runtime definition (set before measuring):** 1 screenplay page ≈ 1 minute, on a rendered
  page in the common US layout (55 lines, action 60 / dialogue 35 / parenthetical 25 characters
  wide; `stages/generate/screenplay.py`); acceptance band 9–11 pages, and dialogue alone ≥ 4 min
  at 150 words per minute. First draft measured 11.69 pages (over the band) → 6 non-plot lines
  cut. Final: **10.91 pages; 1,073 dialogue words = 7.15 min of speech.** The band was not changed.
- **Scene-by-scene loading:** every scene loads with its full continuity state (all 4 characters)
  and only its own text plus 9 one-line summaries (tested; token sizes in D-7.3).
- **Continuity (A-7), seeded errors, 200 trials per kind, seed 0** (full checker / in-scene checks
  only / scene-boundary check only):

  | Corruption | Full | In-scene only | Boundary only |
  |---|---|---|---|
  | wrong location in a move | 1.000 | 0.055 | 1.000 |
  | edited start state | 1.000 | 0.810 | 0.880 |
  | phantom prop pick-up | 1.000 | 0.860 | 0.760 |
  | dropped `=` line | 0.885 | 0.230 | 0.795 |

  First run: phantom 0.815, dropped 0.835. Every miss was inspected: most were "picks up what he
  already holds", then only a warning → made an **error** (a pick-up that cannot happen). Remaining
  misses (11.5 % of dropped lines) are undetectable by design: changes in the last scene (nothing
  follows) and weather / lighting changes just before a declared time jump (`offscreen` covers
  them). Clean script: 0 issues (no false alarm). Both halves are needed (ablation).
  A-7 status: **adopted** (beats the no-check baseline, which catches nothing).

### D-7.3 Token measurement (2026-10-19, measured)
Tokenizers: tiktoken 0.14.0 `o200k_base` (MIT) and the tokenizer of `Qwen/Qwen2.5-0.5B-Instruct`
(Apache-2.0 per model card and LICENSE, revision 7ae557604adf67be50417f59c2c2f167def9a775,
`tokenizer.json` 7,031,645 bytes, SHA-256 c0382117…e87539, git-ignored `data/models/`; read with
`tokenizers` 0.23.3, Apache-2.0). Optional extra `.[script]`. `python -m eval.tokens`.

| o200k_base (Qwen2.5 in brackets) | 10 scenes (~11 min) | 60 scenes (~1 h, sample x6) |
|---|---|---|
| whole `.wscript` (header + all scenes) | 7,413 (8,185) | 41,443 (45,955) |
| whole script as plain screenplay | 3,306 (3,388) | 19,836 (20,328) |
| **scene context** (`load_context`), mean / max | **919 / 1,115** (1,012 / 1,221) | **2,159 / 2,355** (2,387 / 2,596) |
| one scene's `.wsl` alone, mean | 365 (397) | 365 (397) |

Findings:
- Scene loading works as designed: the context is 28 % of the plain whole script at 10 scenes
  and 11 % at 60; it grows only by one summary line (~21 tokens) per extra scene, while the whole
  script grows 6x. (The plain screenplay leaves out state, sound and POV, so it holds less.)
- **IDs do not save tokens here** — the brief's assumption fails for short names: scenes written
  with `{C01}` take 3,653 tokens vs 3,382 with names spelled out (+8 %; Qwen +12 %). IDs remain
  for losslessness and unambiguous references; they would save tokens only for long reusable
  `T…` descriptions. No token saving is claimed for IDs.
- Feeding the whole `.wscript` costs 2.2x the plain script, mostly the per-scene state in the
  JSON header — never feed it whole.
- Prompt change: the continuity block is now one line per character + one for the world instead
  of indented JSON (`continuity_lines`): mean context 1,020 → 919 tokens (−10 %), same facts.
- Script fix found while reading prompts: lighting stayed `oil_lamp` in the chapel after leaving
  the hut — the tracker cannot know lighting belongs to a place (A-7 limit, added below); fixed
  in the script (`phone_torches` in S005, `oil_lamp` again in S007). Runtime unchanged (10.91 p).
- At ~60 scenes the summaries are half the context; for longer films they could be windowed
  (nearest scenes in full, distant ones grouped) — ROADMAP, not needed for the 1-hour rung.

### D-7.4 Character bible, style filter and distinctness (A-8) (2026-10-19, measured)
- **Identity code** (`stages/generate/bible.py`): `seed` + style request → 17 appearance
  dimensions (age 18–85, presentation, height, build, face proportions, skin, hair, facial hair)
  and 5 voice dimensions (pitch, formant, rate, breathiness, roughness), each 0–1. Stored per
  character in `.wscript` 1.1 as `identity` (spec §2, schema); `verify` recomputes it from the
  seed, so a hand-edited identity is detected (tested). The vectors stay out of model prompts.
- **Style filter** (`stages/safety/likeness.py`, the brief's likeness-request hook): allowlist of
  ~60 style words / phrases. Any unknown word rejects the request, so names cannot be requested;
  "like / resembling / lookalike / famous …" are refused as likeness requests; contradictions
  ("tall short") are refused. Adults only (age scale starts at 18). Default, not rule: a feminine
  presentation is clean-shaven unless facial hair is requested. Limit: a name made only of style
  words would pass (none known); Phase 9 hardens this hook.
- **Distinctness thresholds (set before measuring):** RMS difference ≥ 0.20 for appearance and
  for voice, between every pair of characters in a script.
- **Measured** (`python -m eval.identity`, 300 casts per size with no style requests; failure =
  any pair under a threshold):

  | Cast size | independent seeds (baseline) | best of 8 | best of 32 (A-8) |
  |---|---|---|---|
  | 2 | 5.0 % | 0 % | 0 % |
  | 4 | 26.3 % | 0 % | 0 % |
  | 8 | 71.3 % | 0 % | 0 % |
  | 12 | 95.7 % | 0 % | 0 % |
  | 20 | 100 % | 4.7 % | 0 % |
  | 40 | 100 % | 93.7 % | 1.3 % |

  Hard case, every character requested as "older bearded man, deep voice" (100 casts): best of
  32 passes up to 12 characters (0 %), 6 % fail at 20 and 100 % at 40 — the request itself leaves
  too little room, and the check names the pairs (intended: the writer must vary the requests).
  Collapse counter: 20 copies of one identity → 1 unique; a sampled cast of 20 → 20.
  A-8 status: **adopted** (beats the baseline at every size; K ablation above).
- **Sample script cast** (4 characters, requests in `tools/make_sample_wscript.py`, cast seed
  2026): distinct — min appearance 0.405, min voice 0.418; no tampering.
- Still open, by design: faces and voices are not rendered yet, so the distance is in parameter
  space, not perceptual; Phase 8 must repeat the check on face / speaker embeddings of the output.

### D-7.5 Script model: Soup verified, data, task and pass marks (2026-10-19, set before training)
- **Soup is real** (closes D-0.8): PyPI `soup-cli` 0.76.1, Apache-2.0, repo
  github.com/MakazhanAlpamys/Soup. README: `pip install "soup-cli[train]"`, `soup train --config
  soup.yaml`, Python 3.10–3.12 only, Qwen 2.5 supported, layer streaming opt-in and BETA. Every
  config key used is checked in `src/soup_cli/config/schema.py` at commit 64c12ca (base, task,
  data.train/format/val_split/max_length/train_on_responses_only, training.epochs/lr/batch_size/
  gradient_accumulation_steps/seed/quantization/gradient_checkpointing, lora.r/alpha). Kaggle
  runs Python 3.13 (D-3b), so the job builds a Python 3.12 environment with `uv` for Soup.
  Layer streaming is not needed (0.5 B model, 16 GB T4) and is not used or claimed.
- **Model:** Qwen/Qwen2.5-0.5B-Instruct, revision 7ae5576…, Apache-2.0 (model card + LICENSE).
- **Data:** 13 Project Gutenberg plays, public domain in the US (authors died 1900–1950;
  translations by William Archer, R. Farquharson Sharp and others, published before 1929); raw text
  and conversions stay in git-ignored `data/plays/`, uploaded only as a private Kaggle dataset,
  Gutenberg header / licence stripped, never redistributed. `tools/plays_to_wscript.py`
  converts 12 (An Enemy of the People uses title-case speaker names → skipped; Pygmalion's text
  labels only Acts I–II, so II–V are one scene; Hedda Gabler has one false speaker "Mrs"):
  ~12,200 dialogue lines, ~3,000 directions; every converted play validates and passes the
  continuity tracker. Converting found a **tracker bug**: a character entering in a later scene
  had `None` vs empty prop lists → false `continuity_break`; fixed (missing list = empty).
- **Task:** scene continuation — the `load_context` prompt (dictionary, continuity, summaries,
  last ≤ 24 lines of the scene) → the next 8 `.wsl` lines (`stages/generate/script_model.py`).
  Split by play: test = Ghosts + Arms and the Man (97 prompts), val = Uncle Vanya (39), train =
  the other 9 (1,417); out-of-domain test = The Lantern Keeper (20 prompts, modern, with sound /
  POV / state lines the plays never have). Data version `manifest.json` (seed 7).
- **Metrics** (greedy decoding, ≤ 320 new tokens, same system prompt for both models):
  syntax validity (every line parses, ≥ 1 line), reference validity (only the script's IDs),
  and teacher-forced loss on the true next lines. Scorer checked: gold lines score 1.0 / 1.0.
- **Pass marks (set now):** on the held-out plays the fine-tuned model beats the base model on
  all three metrics, and reaches syntax ≥ 0.90 and reference validity ≥ 0.90. The Lantern Keeper
  is reported without a pass mark (out of domain).
