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
