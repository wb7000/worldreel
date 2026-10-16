# Architecture

Status: Phase 0 skeleton. Sections are filled in as each phase lands.

## Overview
```
 video / script
      │
      ▼
 ┌────────┐  ┌────────┐  ┌────────┐  ┌─────────────┐  ┌────────┐  ┌──────┐
 │ ingest │─▶│ frames │─▶│ camera │─▶│ reconstruct │─▶│ actors │─▶│ fill │──┐
 └────────┘  └────────┘  └────────┘  └─────────────┘  └────────┘  └──────┘  │
                  │                                                          ▼
                  └──────────────▶ ┌───────┐ ─────────────────────────▶ ┌─────────┐
                                   │ audio │                            │ package │─▶ .wreel
                                   └───────┘                            └─────────┘
                                                                             │
                                                                        ┌────────┐
                                                                        │ export │─▶ .wedit / .mp4 / glTF / USD
                                                                        └────────┘
 safety: every stage's inputs and outputs pass through the safety interface.
 generate (later): .wscript ─▶ scene generation ─▶ package
```

## Stage contract (Phase 2, `pipeline/stage.py`)
Each stage class declares `name`, `version`, `impl` (`mock` or `real`) and `depends_on`
(other stage names, or `source` for the raw file), and implements `run(ctx)`. The context gives
the item (id, hash, licence, rights), finished input folders, an empty output folder, the stage's
config and the seed. File names shared between stages are constants in `pipeline/io.py`.

| Stage | Depends on | Mock (CPU, tiny) | Real |
|---|---|---|---|
| frames | source | synthetic 64×48 frames | ffmpeg: PNG frames, proxy.npy, audio.wav, scene cuts; `images`: a folder of frames |
| camera | frames | static camera over a flat floor | classical SfM (SIFT, PnP, bundle adjustment), CPU |
| reconstruct | frames, camera | back-projects frame 0 | plane-sweep multi-view stereo + voxel fusion, CPU |
| actors | frames, camera | finds nothing | Phase 3b |
| fill | reconstruct, camera | grey "guessed" floor behind the camera | Phase 4 |
| audio | frames | empty audio section | Phase 5 |
| package | frames … audio | — (the real one is cheap) | signed `.wreel` |
| export | package | records what would be exported | Phase 6 |
| generate | source (script) | returns the canned sample scene | Phase 8 |

`ingest` (`stages/ingest/store.py`) runs before the pipeline; `safety` is the hook below.

## Job runner (`pipeline/runner.py`)
- Runs stages in dependency order, independent ones in parallel (thread pool, `max_workers`).
- **Cache key** = SHA-256 of (stage, version, impl, stage config, seed, source hash, the
  *content* hashes of every input stage's output). A finished stage is reused when the key
  matches and its files are unchanged; a re-run that produces identical bytes therefore does
  not invalidate anything downstream.
- **Atomic, resumable:** a stage writes into `.<stage>.tmp-*` and is moved into place only on
  success; leftovers from a crash are deleted on the next run.
- **Failure isolation:** a failing stage records `failed` with the error; its dependents are
  marked `failed (blocked)`; independent stages still finish.
- **Safety hook:** required. Every stage's inputs and outputs go through it, and every
  decision is logged to `safety.jsonl`. Phase 2 uses a logging pass-through.

## Evaluation (`/eval`, Phase 3a)
| Module | Measures |
|---|---|
| `geometry.py` | Sim(3) alignment (Umeyama, on positions + camera axes), ATE, orientation error, RPE |
| `depth.py` | AbsRel, RMSE, δ<1.25, completeness |
| `image.py` | PSNR, SSIM (LPIPS not yet: needs a network) |
| `render.py` | reference z-buffered point renderer for novel views |
| `localize.py` | locate a test image in a reconstruction (SIFT + PnP) |
| `benchmark.py` | runs the real pipeline on a Blender scene and checks `thresholds.toml` |

`python -m eval.benchmark data/synthetic/<scene> [--calibrated]`; scenes come from
`python -m tools.synth render --seed N --actors 0 --eval-views 4 --width 480 --height 360
--frames 30`. Results and how thresholds were chosen: DECISIONS D-3a.6 / D-3a.7.

## Work-directory layout
```
data/processed/<item id>/
  <stage>/              finished outputs + _stage.json (status, cache key, config, inputs,
                        output hashes, timings)
  .<stage>.tmp-*        in-progress output
  safety.jsonl          safety-hook decisions
```
Configuration: `pipeline/default.toml` (implementation per stage, per-stage options, seed,
workers). External programs are located by `tools/toolpaths.py` (env var →
`worldreel.local.toml` → PATH → known locations).

## Formats (Phase 1)
Specs in `/spec`, JSON Schemas in `/spec/schemas`, Python libraries in `/formats`:

| Module | Reads/writes | Notes |
|---|---|---|
| `formats.common` | containers, hashing, schema checks, Ed25519 signing | shared by all formats |
| `formats.wreel` | `.wreel` | `write_wreel`, `read_wreel`, `validate_wreel`, `PointCloud` |
| `formats.wedit` | `.wedit` (directory or ZIP) | container + manifest only until Phase 6 |
| `formats.wscript` | `.wscript` + `.wsl` scenes | `parse_scene`, `serialize_scene`, `load_context` |

`python -m formats validate <file>` validates any of them.

## Player (Phase 1)
`/player`: TypeScript + Vite. Plays `.wreel` only.

```
main.ts            app shell: load file, viewer menu, timeline, render loop
wreel/reader.ts    unzip (fflate), hash + Ed25519 checks, coarse-first LOD loading
wreel/track.ts     track / camera-path sampling (lerp + slerp)
cameras.ts         director's cut, free, third-person, first-person (comfort), overhead
rules.ts           declarative interaction rules (time triggers only so far)
render/webgpu.ts   instanced point discs + uncertainty fog (WGSL)
render/webgl2.ts   same, GLSL ES 3.0 fallback
```

Run `npm --prefix player run dev` after `python -m tools.make_sample_wreel`.

## Safety
See brief §11. The `safety` stage starts as a logging pass-through but is wired in from the start.
