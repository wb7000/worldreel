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

## Player
Web-first (TypeScript + WebGPU with fallback). Reads `.wreel` only. Details in Phase 1.

## Safety
See brief §11. The `safety` stage starts as a logging pass-through but is wired in from the start.
