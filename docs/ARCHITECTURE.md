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

## Stage contract (to be specified in Phase 1/2)
Each stage declares:
- `name`, `version`
- input and output manifests (paths in the work directory layout)
- cache key = hash(stage name, stage version, config, input content hashes)
- status: `pending | running | done | failed`
- a real implementation and a **mock** implementation (CPU-only, tiny synthetic data)

Stages only talk through files. The job runner executes them in dependency order, in parallel where
independent, and resumes after a crash by skipping stages whose cache key is already `done`.

## Work-directory layout
TBD in Phase 2 (`docs/DATA.md` covers the `data/` drop-folder layout).

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
