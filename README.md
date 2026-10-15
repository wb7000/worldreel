# WorldReel

Turn an ordinary flat video into a world you can step into, and later generate such worlds from
scripts.

A normal MP4 stores pixels from one fixed camera. A **`.wreel`** stores the *scene* (geometry,
actors, motion, lighting, sound events) and the WorldReel player renders whatever view you choose:
director's cut, first or third person as any character, free camera, overhead, interactive.

> **Status:** Phase 0 (foundation). Nothing generates or reconstructs anything yet.
> See `docs/ROADMAP.md` for the phase plan and `DECISIONS.md` for every decision so far.

## File types
| Extension | What it is | Opens in |
|---|---|---|
| `.wreel` | Interactive world (scene layers, actors, audio, provenance, rules) | WorldReel player |
| `.wedit` | Edit bundle of standard files (video, stems, render layers, timeline) | Common editors |
| `.wscript` | Structured, token-efficient long-form script | WorldReel tools |
| `.mp4` | Flat export of one camera path | Anything |

Specs: `/spec` (written before code, versioned, open).

## Repository layout
```
spec/       format specs        pipeline/  stage framework, job runner, cache
stages/     one package/stage   player/    web player (TypeScript, WebGPU)
tools/      CLIs, generators    training/  configs + logs (no weights)
eval/       metrics, benchmarks docs/      architecture, models, data, roadmap
tests/      Python tests
```

## Quick start
```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (bash: source .venv/Scripts/activate)
pip install -e ".[dev]"
pytest && ruff check . && mypy
python -m tools.hwcheck       # record this machine's hardware
cd player && npm install && npm test && npm run typecheck
```

## Principles
- Every reconstructed or generated file is labelled as AI-made, in the format itself.
- People from uploaded footage are replaced with synthetic, verified-dissimilar identities by
  default. Voices are synthetic only.
- Unseen regions are marked honestly (observed / inferred / guessed) and shown as such.
- No data, media or weights in Git.
