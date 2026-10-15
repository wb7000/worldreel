# CLAUDE.md — WorldReel

Instructions for any agent (human or AI) working in this repository. The full brief lives in
`docs/MASTER_PROMPT.md`; this file is the short, binding version.

## Vision (one paragraph)
WorldReel turns an ordinary flat video into an explorable world. Instead of storing pixels from one
camera, a `.wreel` file stores the **scene** (geometry, actors, motion, lighting, sound events) and
the player renders whatever view the viewer picks: director's cut, first/third person as any
character, free camera, overhead, interactive. Later it generates such worlds from `.wscript`
scripts, climbing a test ladder (10 s → 1 min → 10 min → 1 hr film).

## Architecture rules
- The system is a set of **stages** (`/stages/<name>`) run by a job runner (`/pipeline`).
- Stages communicate **only through files** in a documented layout. No hidden global state.
- Every stage has: an input/output manifest, a content-hash cache key, a version number, a status
  (`pending/running/done/failed`), and a **mock implementation** that runs on tiny synthetic
  input with no GPU. The whole pipeline must run end to end on mocks.
- Every algorithm sits behind a stage interface so it can be swapped.

## Formats
- Specs live in `/spec` and are written **before** code. Every format carries `format_version`;
  old versions must stay readable.
- `.wreel` (interactive world), `.wedit` (edit bundle of standard files), `.wscript` (script).
- `.wreel` has an audio section and a signed "AI-reconstructed / AI-generated" provenance header
  from v1.

## Hard rules
1. **Never commit data, videos, images, audio, model weights or outputs.** Commit code, specs,
   manifests, dataset-version lists and configs only. `data/` is git-ignored. Use Git LFS / DVC
   pointers if something large must be tracked.
2. **Do not remove or bypass safety hooks** (`/stages/safety`, provenance label, identity
   replacement + dissimilarity check, likeness-request filter, upload-rights/takedown fields,
   sandboxed interaction scripts) to simplify code.
3. **Do not invent** APIs, library functions, model names, licenses or benchmark numbers. Check
   official docs, or mark the item as an assumption in `DECISIONS.md`. Inventing *algorithms* is
   allowed — follow the process in `docs/ALGORITHMS.md`.
4. Never report a number you did not measure.
5. Do not add features outside the current phase; put them in `docs/ROADMAP.md`.
6. Ambiguity → pick a sensible default, record it in `DECISIONS.md`, continue.
7. Every model used is listed in `docs/MODELS.md` with version and license.

## Coding standards
- Python 3.12, fully type-annotated, `ruff` for lint/format, `mypy --strict` on library code.
- TypeScript (strict) for the player.
- Small, replaceable modules; clear names; comment only where the *why* is not obvious.
- Training is driven entirely by config files in `/training/configs` + a frozen dataset version.
  Seeds are explicit and recorded.
- Design for low VRAM / no GPU first (see hardware in `DECISIONS.md`).

## How to run tests
```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest
ruff check .
mypy
cd player && npm install && npm test && npm run typecheck
```
Hardware report: `python -m tools.hwcheck` (writes `hardware.local.json`, git-ignored).

## Multi-agent workflow
- One agent per git worktree/branch; at most 2–3 at once; review and merge before the next batch.
- Agents touching the same files run sequentially.
- End of every phase: run all tests, summarise, record deviations in `DECISIONS.md`, list manual
  steps the owner must do.
