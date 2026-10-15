# Algorithms

Register of algorithms WorldReel creates or adopts where no ready-made method fits (brief §16).

## Process (required for every new algorithm)
1. **Prior-art check** — record what was found, or "nothing suitable found".
2. **Design note before code** — problem, why existing methods fail, idea, pseudocode, time/memory
   complexity (low-VRAM / CPU limits), known failure cases.
3. **Baseline** — simplest existing or naive approach.
4. **Implement behind a stage interface**, with a mock and unit tests.
5. **Measure** on synthetic ground truth with `/eval` metrics, plus at least one ablation.
6. **Status** — `experimental` until it beats the baseline, then `adopted`. Record in `DECISIONS.md`.
7. **Honesty** — no unmeasured claims. Failures are recorded; the baseline stays.

## Register
| ID | Problem | Status | Baseline | Design note |
|---|---|---|---|---|
| — | *(none yet)* | | | |

## Known open problems (candidates)
- Confidence-aware honest filling of unseen regions
- Sound-event → sound selection/generation
- Identity-distinctness sampling (faces and voices)
- World consistency across a 1-hour film
- Token-efficient script chunking for `.wscript`
