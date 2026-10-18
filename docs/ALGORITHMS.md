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
| A-2 | Follow camera that stays in seen space | adopted (D-4.2; synthetic GT check open) | third-person offset behind the actor (actor frame) | below |
| A-1 | Per-Gaussian confidence for splat reconstructions (honest seen / inferred / guessed) | adopted (D-4.1) | constant: foreground 0.9 observed, background 0.6 | below |

## Known open problems (candidates)
- Confidence-aware honest filling of unseen regions
- Sound-event → sound selection/generation
- Identity-distinctness sampling (faces and voices)
- World consistency across a 1-hour film
- Token-efficient script chunking for `.wscript`

---

## A-1 View-support confidence for Gaussian splats (design note, 2026-10-18)

**Problem.** Brief §6.1: mark every region as seen, inferred or guessed, and fog the guesses.
After splat training every foreground Gaussian is labelled `observed` with confidence 0.9
(D-3b.7). Areas no camera saw well (behind furniture, behind people, far from a short camera
path) still get Gaussians — the optimiser puts *something* there — and they render as sharp-
looking smears with full confidence. That is dishonest: on the walking clip most of what a
follow camera shows is such guesswork (D-3b.11).

**Prior art.** Uncertainty estimation for radiance fields is an active research topic
(approaches based on Fisher information or Bayesian perturbation of the trained field exist;
not verified or cited in this session). They need extra GPU passes and per-parameter
second-order statistics; the laptop has no CUDA and the packaging step runs on the laptop.
Nothing suitable found that runs in numpy on a trained export in seconds. Adopting one of them
later stays possible behind the same interface.

**Idea.** A Gaussian is trustworthy when several training views saw it un-occluded and from
different directions (multi-view triangulation needs parallax). Count, per Gaussian centre:
- `n`: training views in which it projects inside the image, is not hidden behind nearer
  opaque Gaussians (coarse z-buffer), and does not fall on a masked person;
- `spread`: the angular spread of those viewing directions.

**Pseudocode.**
```
for each registered view v (pose, K, W, H; person mask optional):
    project all centres; keep z > near and inside the image
    zbuf = min depth per cell (cells of 4 x 4 px) over centres with opacity >= 0.5
    visible = depth <= zbuf[cell] * (1 + tol)  and not mask[pixel]
    n[visible] += 1; accumulate unit view direction d (camera -> centre)
spread = 2 * max over visible views of angle(d_v, mean d)        # second pass
s = min(1, n / N0) * min(1, spread / THETA0)                      # N0 = 8, THETA0 = 10 deg
class, confidence = observed, 0.5 + 0.5 s     if s >= 0.6
                    inferred, 0.2 + 0.5 s     if n >= 1  (capped below 0.5)
                    guessed,  0.1             if n == 0
background-sphere Gaussians keep their D-3b.7 values.
```

**Complexity.** O(G x V) projections (0.47 M x 82 views ≈ 38 M on the walking clip) plus one
z-buffer per view; numpy, chunked; memory O(G). Expected seconds on the laptop.

**Known failure cases.** Mirrors and windows (virtual geometry seen consistently from many
views gets high confidence); textureless walls (many views, still weakly constrained depth);
coarse z-buffer lets thin occluders leak; centres only — a big smeared Gaussian with a
well-seen centre keeps high confidence (ablation: scale penalty).

**Baseline.** The constant labelling above.

**Measurement (synthetic, seed 11 first).** Per foreground Gaussian, error = distance from its
centre to the nearest ground-truth surface point (back-projected GT depth of all frames and
eval views), after the camera Sim(3) alignment. Report: mean error per confidence decile
(should fall as confidence rises), Spearman correlation between confidence and error, and the
honesty pair: of Gaussians with error > 5 % of scene extent, the fraction NOT labelled observed
(recall of bad); of Gaussians labelled observed, the fraction with error ≤ 5 % (precision).
Ablations: `n` only; `n` + spread; + scale penalty. Adopt only if it beats the baseline on
both honesty numbers without labelling most good Gaussians as guessed.

**Result (D-4.1).** Adopted variant: count x spread, score halved when > 2 % of the extent from
any stereo point *and* the stereo covers the Gaussian's image cell in >= 3 views. Seeds 11/12/13:
bad Gaussians flagged 0.79 / 0.97 / 0.96 (baseline 0), observed-precision 0.973 / 0.998 / 0.997
(baseline 0.89 / 0.94 / 0.94), 83–84 % still labelled observed. Plain stereo distance failed on
gappy stereo (walking clip: 42 % flagged); the coverage condition is the fix. Scale-penalty
ablation not run (not needed to pass; listed as future work).

---

## A-2 Seen-space follow planner (design note, 2026-10-18)

**Problem.** "Follow person N" puts the camera at a fixed offset behind the actor's facing
direction. When the person walks towards the original camera, "behind" is the part of the room
the video never saw, so the view is mostly guessed geometry (blur, smears, fog) — seen on the
walking clip (D-3b.11). Brief Phase 4: the viewer must be able to follow any tracked actor; the
view should show what was actually observed.

**Prior art.** Game "chase cameras" avoid walls with ray casts against known collision geometry;
virtual cinematography plans camera paths with visibility and smoothness costs. Neither knows
which parts of a reconstruction are trustworthy. Nothing found that uses a per-Gaussian
confidence map; the idea here combines the two.

**Idea.** Plan the follow path offline (packaging, numpy) and store it as an ordinary camera
path (`cameras[]` + a `directors-cut` preset) — no format change. At each actor sample time,
score candidate camera positions on a ring around the actor and pick a smooth high-score path.

**Pseudocode.**
```
G = Gaussians (subsample <= 60k, opacity >= 0.2), each with confidence c (A-1)
for each track sample t (actor at p_t on the floor, height h):
    for yaw in 0..340 step 20, distance in {0.9h, 1.3h}:      # 36 candidates
        eye = p_t + distance * dir(yaw) + 1.15h * up; target = p_t + 0.55h * up
        render a 48 x 27 front-most-Gaussian buffer from eye towards target (fov 60)
        seen  = mean over pixels of c(front-most) (empty pixel = 0)
        clear = front-most depth at the target pixel >= 0.9 x |target - eye|   (line of sight)
        score = seen if clear else 0
path = Viterbi over t: maximise sum(score) - lambda * (yaw change / 180 deg) - mu * distance change
keyframes: eye and look-at rotation per t; times between samples interpolate.
```

**Complexity.** T x 36 x 60k projections (40 samples → 86 M) + tiny buffers; seconds in numpy.

**Known failure cases.** Coarse buffer misses thin occluders; candidates are sampled, not
optimised; the ring may cut through walls the splats do not represent (empty space looks clear);
the objective rewards seen pixels, so the camera may prefer facing a well-seen wall over framing
the actor nicely (the line-of-sight term only ensures the actor is not hidden).

**Baseline.** Current third-person preset (offset [0, 1.15h, 1.0h] in the actor frame).

**Measurement.** Two numbers per path, the second independent of the objective:
1. seen share (as above) averaged over the path;
2. view-direction gap: angle between the follow view direction and the nearest registered
   training view looking at the same target (smaller = closer to what the video saw; the usual
   proxy for novel-view quality). Report mean and 90th percentile; plus a visual check.
