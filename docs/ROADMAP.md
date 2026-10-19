# Roadmap

Target: first public release ~18 months from Phase 0 (2026-10-15 → ~2028-04). Each phase stops for
verification before the next. Full acceptance criteria are in `docs/MASTER_PROMPT.md` §15.

| Phase | Name | Status |
|---|---|---|
| 0 | Foundation: repo, CLAUDE.md, structure, DECISIONS, hardware, MODELS skeleton, CI | **done** |
| 1 | Format specs + libraries (`.wreel`, `.wedit`, `.wscript`) and player skeleton | **done** |
| 2 | Data pipeline (ingest, manifest, dataset versions) + Blender synthetic generator | **done** |
| 3 | Reconstruction v1 (static, then dynamic) | **3a done** (classical, CPU, branch `phase-3a-reconstruction`); 3b next (GPU); real-video test needs a clip |
| 4 | POV + honest unseen-region filling | |
| 5 | Audio (stems, sound events, mixing, positional, mood) | |
| 6 | `.wedit` export + flat MP4 | |
| 7 | Script brain + character bible | |
| 8 | Generation + test ladder (10 s, 1 min, 10 min) | |
| 9 | Interaction, branching, round-trip edits, safety hardening | |
| 10 | 1-hour film attempt + public benchmark | |

## Phase 3b plan (draft, needs cloud GPU)
1. Pick open models (licences verified into `docs/MODELS.md`) for: learned features / depth,
   Gaussian-splatting training, person/object segmentation and tracking.
2. Kaggle notebook runner driven by the same TOML configs; checkpoints every ≤ 30 min.
3. Splat layer kind in `.wreel` (spec bump to 1.1) and a splat renderer in the player.
4. Moving actors: per-actor geometry and tracks; people get the synthetic-replacement policy.
5. Benchmark with actors (`--actors 2`); fix seed 13's novel-view failure.
6. Real-video acceptance on an owner-provided clip.

## Deferred from Phase 3a (see DECISIONS D-3a.9)
- LPIPS metric; gravity and metric scale; loop closure; faster matching (approximate NN).
- Seed-13-style views poorly covered by the camera path.

## Deferred from Phase 2 (see DECISIONS D-2.x)
- Process-level parallelism across items; file locking for concurrent `tools.data` runs.
- Image inputs are ingested but the video pipeline skips them.
- Ground-truth `.wreel` export from Blender scenes (exact geometry for the player and eval).

## Deferred from Phase 1 (see DECISIONS D-1.8 to D-1.12)
- Measure player frame rate and load time in a real window and on a mid-range phone.
- Point lighting, camera collision for follow cameras, audio playback, more rule types.
- Network streaming (HTTP range requests / spatial chunks) for large `.wreel` files.
- Signing-key trust list. (`.wscript` tokens measured in D-7.3.)
- `.wscript` for films longer than ~1 h: window the scene summaries (near scenes in full, far
  ones grouped); location-bound world state (lighting per place) in the continuity tracker.

## Parked ideas (outside current phase)
- AI camera director, watch-together, branching story editor, creator analytics, VR/AR.
- Desktop player.
- Own-likeness feature behind consent/liveness check.
- "AI edits" in `.wedit` round-trip.
