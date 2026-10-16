# Roadmap

Target: first public release ~18 months from Phase 0 (2026-10-15 → ~2028-04). Each phase stops for
verification before the next. Full acceptance criteria are in `docs/MASTER_PROMPT.md` §15.

| Phase | Name | Status |
|---|---|---|
| 0 | Foundation: repo, CLAUDE.md, structure, DECISIONS, hardware, MODELS skeleton, CI | **done** |
| 1 | Format specs + libraries (`.wreel`, `.wedit`, `.wscript`) and player skeleton | **done** |
| 2 | Data pipeline (ingest, manifest, dataset versions) + Blender synthetic generator | **done (branch `phase-2-data`)** |
| 3 | Reconstruction v1 (static, then dynamic) | next (needs cloud GPU partway) |
| 4 | POV + honest unseen-region filling | |
| 5 | Audio (stems, sound events, mixing, positional, mood) | |
| 6 | `.wedit` export + flat MP4 | |
| 7 | Script brain + character bible | |
| 8 | Generation + test ladder (10 s, 1 min, 10 min) | |
| 9 | Interaction, branching, round-trip edits, safety hardening | |
| 10 | 1-hour film attempt + public benchmark | |

## Phase 3 plan (draft)
1. CPU-runnable baselines first: classical structure-from-motion / depth on the synthetic scenes,
   scored against `gt.json` with eval metrics (depth error, pose error).
2. Pick open models for depth/pose and reconstruction (licences into `docs/MODELS.md`), run them
   on Kaggle through config files; the laptop keeps the mocks.
3. Real `camera`, `reconstruct`, `actors` stages behind the existing interfaces; packaging of
   tracked actors (people get synthetic-replacement identity policy).

## Deferred from Phase 2 (see DECISIONS D-2.x)
- Process-level parallelism across items; file locking for concurrent `tools.data` runs.
- Image inputs are ingested but the video pipeline skips them.
- Ground-truth `.wreel` export from Blender scenes (exact geometry for the player and eval).

## Deferred from Phase 1 (see DECISIONS D-1.8 to D-1.12)
- Measure player frame rate and load time in a real window and on a mid-range phone.
- Point lighting, camera collision for follow cameras, audio playback, more rule types.
- Network streaming (HTTP range requests / spatial chunks) for large `.wreel` files.
- Signing-key trust list; measure `.wscript` token savings with a real tokenizer.

## Parked ideas (outside current phase)
- AI camera director, watch-together, branching story editor, creator analytics, VR/AR.
- Desktop player.
- Own-likeness feature behind consent/liveness check.
- "AI edits" in `.wedit` round-trip.
