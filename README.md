# WorldReel (Prototype 1)

#Future ideas
Turn an ordinary flat video into a world you can step into, and later generate such worlds from
scripts.

A normal MP4 stores pixels from one fixed camera. A **`.wreel`** stores the *scene* (geometry,
actors, motion, lighting, sound events) and the WorldReel player renders whatever view you choose:
director's cut, first or third person as any character, free camera, overhead, interactive.

> **Status:** Phase 3a. A classical CPU reconstruction (structure-from-motion + multi-view
> stereo) turns a video into a free-camera `.wreel` of the static scene, benchmarked against
> Blender ground truth. Learned models, splats and moving people come in Phase 3b (GPU).
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
spec/       format specs + JSON schemas    formats/   read/write libraries + validators
pipeline/   stage framework, job runner    stages/    one package per stage
player/     web player (TypeScript)        tools/     CLIs, generators
training/   configs + logs (no weights)    eval/      metrics, benchmarks
docs/       architecture, models, data     tests/     Python tests
```

## Quick start
```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (bash: source .venv/Scripts/activate)
pip install -e ".[dev]"
pytest && ruff check . && mypy
python -m tools.hwcheck       # record this machine's hardware
cp worldreel.example.toml worldreel.local.toml   # paths to ffmpeg / Blender
python -m tools.data ingest   # after dropping a video + .info.json into data/inbox
python -m tools.data process
python -m tools.make_sample_wreel     # writes player/public/samples/courtyard.wreel
cd player && npm ci && npm test
npm run dev                           # open http://localhost:5173
```

## Principles
- Every reconstructed or generated file is labelled as AI-made, in the format itself.
- People from uploaded footage are replaced with synthetic, verified-dissimilar identities by
  default. Voices are synthetic only.
- Unseen regions are marked honestly (observed / inferred / guessed) and shown as such.
- No data, media or weights in Git.
