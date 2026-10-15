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
