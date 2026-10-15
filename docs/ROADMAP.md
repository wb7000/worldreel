# Roadmap

Target: first public release ~18 months from Phase 0 (2026-10-15 → ~2028-04). Each phase stops for
verification before the next. Full acceptance criteria are in `docs/MASTER_PROMPT.md` §15.

| Phase | Name | Status |
|---|---|---|
| 0 | Foundation: repo, CLAUDE.md, structure, DECISIONS, hardware, MODELS skeleton, CI | **done (local; CI not yet run on GitHub)** |
| 1 | Format specs + libraries (`.wreel`, `.wedit`, `.wscript`) and player skeleton | next |
| 2 | Data pipeline (ingest, manifest, dataset versions) + Blender synthetic generator | |
| 3 | Reconstruction v1 (static, then dynamic) | |
| 4 | POV + honest unseen-region filling | |
| 5 | Audio (stems, sound events, mixing, positional, mood) | |
| 6 | `.wedit` export + flat MP4 | |
| 7 | Script brain + character bible | |
| 8 | Generation + test ladder (10 s, 1 min, 10 min) | |
| 9 | Interaction, branching, round-trip edits, safety hardening | |
| 10 | 1-hour film attempt + public benchmark | |

## Proposed Phase 1 agent batch
`format-agent` (specs + read/write libs) and `pipeline-agent` (stage framework, job runner, cache,
mocks) in parallel worktrees; then `player-agent` once the `.wreel` spec is merged.

## Parked ideas (outside current phase)
- AI camera director, watch-together, branching story editor, creator analytics, VR/AR.
- Desktop player.
- Own-likeness feature behind consent/liveness check.
- "AI edits" in `.wedit` round-trip.
