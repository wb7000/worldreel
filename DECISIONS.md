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
