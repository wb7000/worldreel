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
| A-12 | Conversational delivery: spoken text + manner, whole-exchange rendering, per-turn voice conversion | adopted (D-8.7c, owner) | isolated read-aloud lines | DECISIONS D-8.5–8.7 |
| A-11 | Per-emotion voice-quality shaping after synthesis | experimental — not adopted (hoarse v1, phasey; PSOLA v3 hoarser) | unshaped TTS output | below |
| A-10 | Script-driven emotional delivery for synthetic voices | superseded by A-12 | one fixed (flat) description per character | below |
| A-9 | Procedural places from location descriptions | planned | — | (8.4) |
| A-8 | Identity-distinctness sampling for the character bible | adopted (D-7.4) | independent random seeds per character | below |
| A-7 | Continuity tracking across `.wscript` scenes | adopted (D-7.2) | none (continuity stored, never checked) | below |
| A-6 | Procedural, event-driven footstep sounds | experimental | one fixed footstep sound for every event | below |
| A-5 | Generative texture for filled planes (LaMa) | experimental | plane-mean colour (A-3) | below |
| A-4 | Licensed reference images, geometrically verified, to colour unseen regions | adopted, provisional (D-4.4) | no retrieval (fill colour = plane mean, A-3) | below |
| A-3 | Time-boxed progressive fill of holes in planar surfaces | experimental — failed adoption bar (D-4.3) | gradient only (mock fill: flat grey `guessed` floor behind the first camera) | below |
| A-2 | Follow camera that stays in seen space | adopted (D-4.2; synthetic GT check open) | third-person offset behind the actor (actor frame) | below |
| A-1 | Per-Gaussian confidence for splat reconstructions (honest seen / inferred / guessed) | adopted (D-4.1) | constant: foreground 0.9 observed, background 0.6 | below |

## Known open problems (candidates)
- Confidence-aware honest filling of unseen regions
- Sound-event → sound selection/generation
- Identity-distinctness sampling on rendered faces and voices (parameter space: A-8)
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

---

## A-3 Progressive plane-completion fill (design note, 2026-10-18)

**Problem.** Brief §6.2: show a gradient instantly, try an improved fill within a time budget
(e.g. 5 s), refine in the background and swap in. Today unseen parts are either empty (fog /
gradient) or smeared splats. Most unseen area in our scenes (rooms, courtyards) lies on large
planes — floor, walls, ground — where the observed part already tells us a lot: the plane
itself and the colours around the hole.

**Prior art.** Planar-region completion and image inpainting by diffusion / Laplace
interpolation are classical; RANSAC plane extraction is standard. Learned 3D completion needs a
GPU and training data. Nothing to adopt directly for "fill only what is implied by
observations, label it, and do it within a budget"; this combines the classical parts.

**Idea.** Holes *inside* a detected plane's observed footprint are filled on that plane, with
colours diffused in from the observed border. Nothing is extended beyond the footprint (that
would be guessing). Fill is `inferred` with confidence < 0.5 that falls with distance from
observed evidence, so the player still marks it (spec §10.3).

**Pseudocode.**
```
P = observed points (class observed, confidence >= 0.5): positions, colours
planes = sequential RANSAC on P (<= 6 planes, inlier band 1 % of extent, >= 3 % of P each)
for each plane (tier 1: cell = extent / 60, <= 30 diffusion iterations, stop at budget_s;
                tier 2: cell = extent / 120, until converged or refine_budget_s):
    grid on the plane over the inliers' 2D bounding box
    occupied cells = cells with inliers (mean colour); footprint = 2D convex hull of inliers
    holes = footprint cells that are not occupied
    colour(holes) = Laplace diffusion from occupied cells (Jacobi iterations)
    d = cell distance to nearest occupied cell; confidence = 0.45 * exp(-d / 4)
    emit one flat Gaussian / point per hole cell (normal = plane normal, size = cell)
tier 1 -> fill layer LOD 0, tier 2 -> LOD 1 (player shows LOD 0, then swaps in LOD 1)
```

**Complexity.** RANSAC O(iterations x |P|) on a subsample; grids O(cells x iterations). Seconds.

**Known failure cases.** Non-planar holes (behind furniture, under a sofa) are not filled;
planes crossing the footprint of a doorway would fill the opening (convex hull) — a doorway is a
hole in the wall that should stay open; textured surfaces get blurry colour; a plane under an
observed object gets filled underneath it (harmless, hidden).

**Baseline.** Gradient only (current mock fill / no fill).

**Measurement (synthetic, seeds 11–13).** Ground-truth surface seen from the held-out eval
views (their GT depth + RGB). r = 2 % of extent.
- coverage: fraction of those GT points within r of scene geometry, observed only vs observed +
  fill;
- fill precision: fraction of fill elements within r of the GT surface;
- fill colour error: mean |RGB| difference between fill and the nearest GT point's colour;
- time to tier 1 and tier 2.
Adopt if coverage rises, precision ≥ 0.8, and tier 1 meets its budget.

**Result (D-4.3): not adopted, stays experimental (`tools/splat_wreel.py --fill`).** Changes during
development (all measured): enclosed-gap footprint (morphological closing, ≤ 15 % of extent)
instead of the convex hull, which joined separate box tops through the air; cells under observed
objects are hidden, not filled; colour = the plane's opacity-weighted mean (diffusion from noisy
border splats was worse than a flat colour: 35–40 vs 16–20 mean |RGB|); only level or upright
planes (a 37° plane through clutter on the walking clip); cells that cameras saw through are
dropped. Exact-geometry scenes (generator 0.2.0): fill on the true surface 24 % / 5 % / 98 %
(seeds 11/12/13 v2) against a ≥ 80 % bar. The fill follows the reconstruction it is built from:
observed splats on the true surface were 46 % / 32 % / 98 % on the same scenes. Fix the
reconstruction bias first (D-4.3), then re-test.

---

## A-4 Licensed reference retrieval with geometric verification (design note, 2026-10-18)

**Problem.** Brief §6.3: improve unseen regions with reference images from open-licensed sources
(Wikimedia Commons, Openverse, ...) or the user, *only* after verifying them geometrically against
what the video shows ("a wrong image is worse than a gradient"), never images of private homes or
people, and with source URL + licence stored in the region's provenance record.

**Prior art.** Image retrieval + feature matching + PnP is the standard visual-localisation
pipeline (it is also how our own eval views are localised, D-3a.6). Verified APIs (2026-10-18):
Openverse `GET /v1/images/` (`q`, `license`, `category`, `mature`; results carry `url`,
`license`, `license_version`, `foreign_landing_url`, `attribution`); MediaWiki geosearch on
Commons (`generator=geosearch`, `ggscoord=lat|lon`, `ggsradius`) with `prop=imageinfo&
iiprop=url|extmetadata` (`LicenseShortName`, `LicenseUrl`, `AttributionRequired`, `Categories`).
Wikimedia requires a User-Agent with contact information; WorldReel sends none until the owner
configures one (`worldreel.local.toml`), so network retrieval is off by default.

**Eligibility (before any request).** The item must be marked by the owner as a public outdoor
place with a place name or coordinates. Rooms / homes are never eligible.

**Candidate filters.** Licence in {CC0, PDM, CC BY, CC BY-SA} (ND refused — reconstruction is a
derivative; NC refused — use cannot be guaranteed non-commercial); photographs only; mature off;
reject when metadata (categories / tags) mention people, portraits, selfies, crowds, interiors or
homes, and when a face detector (OpenCV Haar frontal face, shipped with opencv-python) fires.

**Geometric verification.**
```
for each candidate image I (resized to <= 800 px):
    SIFT match I against every registered frame (ratio 0.75); keep the 3 best frames
    for the best frame F: 3D for F's matched keypoints = observed geometry depth at those pixels
        (z-buffer of observed points from F's camera)
    solvePnPRansac(2D in I, 3D) for focal in a grid (0.6..1.6 x image width), keep the best
    accept iff inliers >= 40, inlier ratio >= 0.25, median reprojection <= 2 px, and the pose is
        consistent with a second frame (>= 20 inliers of the same model in another frame)
```
**Use.** Project unseen / inferred elements into each accepted image; where visible (z-test
against observed geometry) take the image colour; class `retrieved`, confidence 0.5; one region
per image with `sources: [{kind: retrieval, url, license, verified: geometric}]`; attribution
text in `extensions["org.worldreel.credits"]`.

**Complexity.** SIFT on (frames + candidates); PnP per candidate; seconds to a minute on CPU.

**Known failure cases.** Repetitive facades (wrong matches with a consistent pose); images
from a very different season / time of day (few matches → rejected, which is safe); a person too
small for the face detector; licence metadata errors upstream (Commons warns that multi-licensed
values are unreliable — such files are skipped).

**Baseline.** No retrieval (plane-mean fill colour / fog).

**Measurement (synthetic, no network).** User-provided references = the held-out eval views of
the same scene (positives) and of other scenes (negatives). Report: acceptance of positives,
rejection of negatives, pose error of accepted images vs ground truth, and colour error of
recoloured elements vs the plane-mean baseline.

**Result (D-4.4).** Two checks were added after the first run accepted 2 of 24 wrong-scene images
(identical procedural ground texture across synthetic seeds — a stand-in for repeating real
pavement): >= 5 % of PnP inliers off the dominant plane, and colour agreement >= 0.3 between the
photo and projected observed points. With them: 12 / 12 true references accepted, 24 / 24 wrong
ones rejected on seeds 11v2/12v2/13v2; pose error 0.14–0.86 m, 0.7–1.9°. Thresholds were chosen
on these same scenes (in-sample) — provisional until validated on real photos.

---

## A-5 Generative plane texture (design note, 2026-10-18)

**Problem.** Brief §6.4: a generative model produces the remainder, tagged `generated`. A-3
fills holes in planes with one flat colour per plane — honest but visibly fake on textured ground.

**Model (verified).** LaMa image inpainting: code Apache-2.0 (advimman/lama LICENSE), Big-LaMa
weights model card apache-2.0 (huggingface smartywu/big-lama); ONNX port `Carve/LaMa-ONNX`
`lama_fp32.onnx` (model card: Apache-2.0, opset 17, fixed 512x512), 208,044,816 bytes, SHA-256
1faef530...eb68d6. Inputs read from the file: `image` [b,3,512,512] float, `mask` [b,1,512,512];
measured on a synthetic frame: image in [0,1], mask 1 = hole, output in [0,255]; ~12 s per
512x512 inference on the laptop CPU with onnxruntime 1.31.0 (MIT).

**Idea.** For each A-3 plane: rasterise an orthographic texture of the plane (cell colours =
opacity-weighted mean of observed Gaussians in the cell), mask the hole cells, inpaint with LaMa
in 512x512 tiles (resampled), and colour each fill element from the inpainted texture. Class
`generated`, confidence 0.2; a region record with `sources: [{kind: model, model, version}]`.
Progressive (§6.2): plane-mean colour is the quick tier, the generated texture the refined tier.

**Complexity.** One 512x512 inference per plane tile, ~12 s each on CPU; usually 1–6 planes.

**Known failure cases.** Invents content (erased half a box in the first test — the model does
not know what was there); texture scale must match the plane's cell size; tiles can show seams;
inherits A-3's geometry errors (D-4.3).

**Baseline.** A-3 plane-mean colour.

**Measurement.** On seeds 11v2/12v2/13v2, for fill elements on the true surface: (1) error vs
the local mean GT colour (as eval.fill) and (2) error vs the single nearest GT pixel (texture
level), plus (3) the spread (std) of fill colours vs the spread of GT colours around them
(a flat fill has spread 0; real ground does not). Report all three honestly: a generated texture
can be more realistic (3) while not more accurate (1).

**Result (D-4.5): experimental, opt-in (`--fill --generative`).** First run: generated colours were
far too dark (error 140–170) — the hole cells were zero and resampling to 512x512 bled them into
the known context (known texture 117 vs ~185). Fixed by pre-filling every unknown cell classically
(OpenCV TELEA) and sending only real holes (> 2 cells from observed cells) to LaMa. Then, on
seeds 11v2/12v2/13v2: texture spread vs the real surface 0.91 / 1.15 / 0.96 (plane-mean 0.12 /
0.34 / 0.00) — realistic texture; but error vs the local true colour 37.9 / 39.9 / 43.1 (plane-mean
16.9 / 27.1 / 14.9) and vs the nearest true pixel 43–47 (plane-mean 23–32): invented texture is
not where the real texture was. More realistic, less accurate — hence `generated`, confidence 0.2.

---

## A-6 Procedural event-driven footsteps (design note, 2026-10-18)

**Problem.** Brief §7: every action emits an event with attributes (`surface`, `footwear`,
`weight`, `speed`, `weather`, ...) and the engine selects or generates the matching sound;
"walking on wet concrete in rain sounds different from dry gravel". Recorded libraries need
licences per sound; the brief prefers generated sounds.

**Prior art.** Procedural / physically informed footstep synthesis exists in the sound-design
literature (modal resonators for hard surfaces, granular synthesis for crunchy ones); not
verified or cited in this session. Nothing to adopt as a library; the method below is a small
own design in that spirit, in numpy, deterministic per seed.

**Model.** A step = heel impact + toe roll (second, weaker transient 60–120 ms later, shorter at
higher speed). Each transient = excitation x surface response:
- hard (concrete, stone): 15 ms bright noise burst + low "thump" (decaying ~100 Hz sine,
  amplitude ~ weight);
- wood: damped modes at 200–800 Hz, ~40 ms;
- gravel: granular — 20–60 tiny bright clicks spread over ~120 ms (Poisson);
- grass: soft low-passed swish, ~150 ms, no sharp click;
- sand / dirt: muffled low-passed burst, ~60 ms.
Footwear shapes the attack (boot: stiff + more thump; sneaker: softened attack; heels: short
high click; barefoot: soft, low). `wet` adds a splash (3–8 kHz band-passed fizz, 80–200 ms) and a
small squelch; `weather: rain` implies wet unless the surface is covered (indoor). Loudness grows
with weight and speed. Three seeded variants per attribute combination avoid machine-gun repeats.

**Measurement.** Per clip: spectral centroid, decay time (RMS to −20 dB), share of energy above
3 kHz, onset count (granularity). Acceptance check: for every pair of conditions in the matrix
surfaces x {dry, wet}, the feature distance between conditions exceeds the largest distance
between variants of one condition (so each condition is distinguishable), and wet vs dry differs
in the expected direction (more high-band energy, longer tail).

**Known limits.** Synthetic, not recorded: plausible rather than realistic; no room acoustics
yet; surface per step comes from scene/config, not from vision (no material recognition).

---

## A-7 Continuity tracking across scenes (design note, 2026-10-19)

**Problem.** Brief §4.3: "a continuity state per scene ... so scene 40 matches scene 4". `.wscript`
1.0 stores the state at the start of each scene, but what happens in a scene is free text, so a
tool cannot tell whether scene n+1's state follows from scene n.

**Prior art.** Screenplay software tracks props / wardrobe in breakdown sheets (manual lists);
story-state tracking for language models is a research topic; nothing found that fits a
line-based, lossless script format. Own design, deliberately simple.

**Idea.** Make changes explicit and replay them.
- New `.wsl` line `= <ID> key=value …`, `<ID>` a character or `world`. Character keys:
  `location=<L…>`, `wearing+=<P…>`, `wearing-=<P…>`, `holding+=<P…>`, `holding-=<P…>`,
  `state=<text_with_underscores>`. World keys: `lighting=`, `weather=`, `time=`.
- `replay(start, statements)` applies changes in order and returns the end state; at every
  dialogue line the speaker must be at the scene's location (else `speaker_absent`); after every
  change no prop is held or worn by two characters (`prop_in_two_places`); a `-=` of something not
  held is an error (`prop_not_held`).
- Between scenes: `diff(end(n), start(n+1))` must be empty, except paths listed in scene n+1's
  `continuity.offscreen` (e.g. `["C01.wearing", "world.time"]`) — explicit time jumps.
- Authoring help: `propagate(script)` fills each scene's start state from the previous end state
  (keeping declared offscreen changes), so writers only write changes.

**Complexity.** Linear in the number of lines; state is small (characters x props).

**Measurement.** Unit tests on hand-made violations (each check fires, no false alarm on the clean
script); on the 10-minute script: 0 errors, and seeded corruptions (move a prop, delete a change,
swap a location) are all detected — detection rate reported.

**Known limits.** Only what is written as `=` lines is tracked; actions described in prose alone
are invisible to it. No physical plausibility (travel time between locations).
World state is global: it cannot know that lighting belongs to a place (found in D-7.3: a
hut's oil lamp carried into the chapel). Location-bound world state is a possible extension.

---

## A-8 Identity-distinctness sampling (design note, 2026-10-19)

**Problem.** Brief §5: identity must be an explicit input; characters in one film must be farther
apart than a threshold (faces and voices); collapse must be detectable. Independent random seeds
can, by chance, give two similar characters, and the chance grows with the cast size.

**Prior art.** Farthest-point / Poisson-disk sampling (classic, for spreading points in a space);
rejection sampling. Used here as the method; the novel part is only the identity space and the
check.

**Idea.**
- Identity code: `seed` → deterministic draw of an appearance vector (face shape, skin tone, hair
  colour / length, eye / nose / mouth proportions, age, build, height) and voice parameters
  (median pitch, formant scale, rate, breathiness), each dimension normalised to [0, 1]; any
  user-specified attributes (from the style parser) are fixed and the rest drawn.
- Distance: weighted Euclidean over normalised dimensions, separately for face and voice.
- Sampler: for each new character draw K candidate seeds and keep the one whose nearest existing
  character is farthest (best-candidate sampling); deterministic given a cast seed.
- Check: minimum pairwise face and voice distance ≥ thresholds (set before measuring); violations
  name the pair. Collapse counter: number of distinct identities (after quantising) in a batch.

**Baseline.** Independent seeds. **Measurement.** Over many random casts of size 2–40: rate of
casts that fail the threshold, baseline vs sampler; ablation K=1 (= baseline).

**Known limits.** Distance in parameter space is not perceptual distance; once Phase 8 renders
faces and voices, the check must also run on face / speaker embeddings of the output.

---

## A-10 Script-driven emotional delivery for synthetic voices (design note, 2026-10-20)

**Problem.** The first generated voices (D-8.1) were stable and distinct but, in the owner's
words, "sounds like robot, doesn't have emotions". Brief §7 asks for an `emotion` attribute on
sound events. A description-controlled TTS (Parler-TTS) reads delivery from its text description,
so a fixed description per character gives the same flat delivery for every line.

**Prior art.** Expressive / emotional TTS conditions on emotion labels or reference audio; Parler
descriptions include expressiveness ("expressive and animated", "monotone") per its model card.
Reference audio would mean cloning — not allowed. Nothing ready-made maps a screenplay to
per-line descriptions; own design.

**Idea.** Description = identity part (fixed per character: gender, pitch, speed, timbre) +
delivery part (per line), from what the script says:
- tension 0–10 at that line (`%` lines / scene tension) → expressiveness: ≤ 2 "calm, slightly
  expressive", 3–5 "expressive and animated", 6–8 "very expressive, tense", ≥ 9 "urgent, very
  expressive and animated" (never "monotone");
- manner note → emotion words (whisper / quietly / under the breath / barely audible → "speaks
  softly, almost whispering"; dry → "dry, deadpan"; gently / careful → "gentle, warm";
  smiling / laughs → "cheerful"; angry / shouting → "angry, loud"; ...);
- line cues: "!" → "excited"; questions → "questioning"; a small lexicon (fear: "no, no", "please",
  "fast", "come on"; grief: "dead", "died", "fifty years"; warmth: "thank you", "family");
- emotion also goes into the dialogue sound event's `emotion` attribute (brief §7).
Seed stays fixed per character so the identity part dominates.

**Measurement (thresholds set before generating).** All 101 lines of the sample script, twice:
flat (D-8.1 descriptions) and A-10. Expressiveness = per-line pitch spread (standard deviation of
f0 in semitones over voiced frames). Pass: mean spread ≥ 1.25x flat, **and** voice identity still
meets D-8.0 (nearest-centroid ECAPA accuracy ≥ 0.95, centroid cosine between characters < 0.60),
**and** the owner's listening test. Pitch spread is a proxy, not "emotion"; the listening test
decides.

**Known limits.** The lexicon is crude and English-only; the TTS may ignore some words; emotion
words may pull the voice away from its identity (that is why identity is re-measured).

---

## A-11 Per-emotion voice-quality shaping (design note, 2026-10-21)

**Problem.** After three voice rounds (D-8.2–8.3e) the owner still hears "robotic": emotion must
change the *sound* of the voice, not only how much the pitch moves — "when humans are happy
their vocal changes". Open TTS models give some acting (Dia > Chatterbox, D-8.3d) but none
controls voice quality per emotion while keeping a designed voice.

**Prior art.** Emotional voice conversion in research changes pitch contour, duration and
spectrum (voice quality) per emotion; acoustic correlates are well described in the speech
literature (joy: higher, wider pitch, raised formants in "smiled" speech, brighter spectrum;
sadness: lower, narrower, slower, darker; anger: wider range, tense, strong high frequencies;
fear: higher, faster, breathy, unsteady; crying: tremor, breathy, breaks). Cited from general
knowledge, not re-checked in this session. Learned EVC models need emotional training data we
do not have; a rule-based transform on a vocoder is cheap, transparent and testable.

**Idea.** WORLD (pyworld, MIT; WORLD 3-clause BSD) splits speech into pitch (f0), spectral
envelope (voice timbre) and aperiodicity (breathiness), so each can be edited and resynthesised:
- pitch: shift (semitones), range (scale of log-f0 around its mean), tremor (sinusoidal,
  Hz and depth), jitter (random per-frame);
- timbre: formant warp (envelope frequency axis x alpha; alpha > 1 = "smiling", brighter),
  spectral tilt (dB per octave above 1 kHz: + harsh / bright, − dark);
- breath: aperiodicity pulled toward 1 (0 = as is, 1 = whisper);
- tempo: time-scale of the analysis frames (> 1 slower).
Each emotion has a preset (below); the line's intensity interpolates from no change to the
preset. Loudness is matched to the source. Applied after TTS (and after voice conversion), so
it works with any source.

| emotion | shift st | range x | tremor Hz / st | jitter st | warp | tilt dB/oct | breath | tempo |
|---|---|---|---|---|---|---|---|---|
| excited / joyful | +2.0 | 1.5 | – | – | 1.06 | +2.5 | 0.10 | 0.93 |
| comic / sarcastic | +0.5 | 1.7 | – | – | 1.03 | +1.0 | – | 1.05 |
| despair / sad | −1.5 | 0.7 | – | – | 0.98 | −3.0 | 0.25 | 1.12 |
| crying | +1.0 | 1.2 | 6.0 / 0.8 | 0.3 | 1.00 | −1.0 | 0.35 | 1.08 |
| angry | +1.0 | 1.4 | – | – | 1.00 | +4.0 | – | 0.95 |
| afraid | +2.5 | 1.3 | 7.0 / 0.4 | 0.15 | 1.02 | +1.0 | 0.30 | 0.90 |
| whisper | 0 | 0.8 | – | – | 1.00 | +1.0 | 0.85 | 1.00 |
| shocked grief | +1.5 | 1.4 | 5.5 / 0.6 | 0.2 | 1.00 | −1.0 | 0.30 | 1.05 |
| tender | −0.5 | 0.9 | – | – | 1.00 | −1.5 | 0.20 | 1.05 |
| dry humour | 0 | 0.8 | – | – | 1.00 | 0 | – | 1.00 |
| neutral | no change | | | | | | | |

**Measurement (set before listening).** Same 12 lines; sources = round-3 renders. Shaped vs
unshaped: pitch spread up for the emotions whose preset widens range; ECAPA nearest designed
reference still 12 / 12 (shaping must not change who is speaking); Whisper WER ≤ 0.30 after
shaping (still intelligible); owner's blind listening decides adoption.

**Known limits.** Vocoder resynthesis can sound buzzy; presets are hand-set (no learning);
laughter and sobs as sounds still have to come from the TTS (tags), shaping only colours speech.
