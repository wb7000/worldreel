# MASTER BUILD PROMPT: "WorldReel" (Interactive Video: Reverse-Engineer Any Video Into a World You Can Step Into)

You are a senior research engineer, systems architect and full-stack developer. You will lead a multi-agent build of a long-term, ambitious project. The timeline is long (about 18 months to a first public release), so quality, structure and measurability matter more than speed. Work in phases, verify each phase before the next, record decisions, and never invent facts about libraries or models. If you are not sure an API, model or paper exists as you remember it, check the official docs or mark it as an assumption in `DECISIONS.md`.

Do not ask me questions unless truly blocked. When something is ambiguous, pick the sensible default, write it in `DECISIONS.md`, and continue.

---

## 1. VISION

**WorldReel** turns a normal flat video into an explorable, interactive world, and can also generate such worlds from scripts.

- A normal video (MP4) stores pixels from one fixed camera. WorldReel stores the **scene** (geometry, actors, motion, lighting, sound events). The player renders pixels on demand from whatever view the viewer chooses.
- The viewer menu ("How would you like to see this?"): original director's cut, first person as any character, third person following any character, free camera, overhead, interactive mode where the world reacts, and (later) "what if" story branches.
- Core idea: **reverse-engineering**. Given any ordinary video, reconstruct the underlying scene. Later: generate full scenes and eventually a 1-hour film from a script.
- Two custom file types plus a flat export:
  1. **`.wreel`**: the interactive world. Plays only in our own player (web-first).
  2. **`.wedit`**: an edit bundle built from standard formats so any common editor can open it (stems, render layers, timeline).
  3. **Flat export**: a normal MP4 of one chosen camera path, for sharing.
- Script format **`.wscript`** for long scripts (structured, token-efficient).

The final north-star test is a coherent 1-hour film with many distinct characters and voices, built up through a test ladder (10 s, 1 min, 10 min, 1 hr). It is the final exam, not the first milestone.

---

## 2. CONSTRAINTS AND OPERATING ENVIRONMENT

- Primary dev machine is a **laptop with a modest GPU**. Assume low VRAM (about 4-8 GB) unless detected otherwise. Run `nvidia-smi` or equivalent at the start and record the hardware in `DECISIONS.md`.
- Design every stage to run on low VRAM: LoRA/QLoRA fine-tuning, gradient checkpointing, CPU/RAM offloading, small batch sizes, and **layer streaming** (keep frozen weights in RAM and move one layer at a time to the GPU, the approach used by the open-source "Soup" project for fine-tuning LLMs on small GPUs). Do not claim any technique works on video models unless you verify it.
- We **fine-tune and adapt open models**; we do not train frontier models from scratch. Use open-licensed model weights and record each model's license in `docs/MODELS.md`.
- Cloud GPU may be added later. Keep all training behind config files so it can move machines without code changes.
- GitHub repo is connected. **Never commit videos, images, weights or large data.** Commit code, specs, manifests, dataset-version lists and configs only. Use Git LFS or DVC pointers if large files must be tracked.

---

## 3. ARCHITECTURE: A STAGED, PARALLEL, RESUMABLE PIPELINE

Split the whole system into independent **stages** (workers). Each stage reads files, writes files, and can be run, cached, retried and swapped on its own.

Why: stages run in parallel, a failure in one does not crash the rest, results are cached, any stage can be replaced by a better model, and the player can start showing the world while later stages are still finishing.

Stages (initial list):
1. `ingest`: validate, hash, deduplicate, record license/source, move to `raw/`
2. `frames`: decode video to frames and audio, scene-cut detection
3. `camera`: depth and camera-pose estimation
4. `reconstruct`: static + dynamic scene reconstruction (3D Gaussian splatting or successor)
5. `actors`: segment and track people/objects as separate controllable actors
6. `fill`: handle unseen regions (see section 6)
7. `audio`: separate dialogue, music, ambience, effects; build sound events (see section 7)
8. `package`: write the `.wreel` container
9. `export`: `.wedit` bundle, flat MP4, glTF/USD export
10. `generate` (later): script to scene generation using a world model
11. `safety` hooks (see section 11)

Requirements for the pipeline framework:
- Each stage has a manifest of inputs/outputs, a content-hash cache key, a version number, and a status (`pending/running/done/failed`).
- A job runner executes stages in dependency order, in parallel where independent, resumable after a crash.
- Stages communicate only through files in a documented layout, never through hidden global state.
- Every stage has a **mock implementation** so the whole pipeline runs end to end on tiny synthetic input without a GPU.

---

## 4. FILE FORMATS (design specs first, code second)

Write a full specification in `/spec` for each format before implementing it. Version every format (`format_version`) and keep old files readable. Publish the specs as open documents.

### 4.1 `.wreel` (interactive world)
A single container (zip-like, with a JSON manifest), inspectable with normal tools. Contains:
- **Scene layers:** static geometry/points, dynamic actors with motion tracks, lighting, camera paths (the original director's path plus named alternatives).
- **Actor list:** each actor has an ID, type, track, optional `character_id` linking to the character bible.
- **Region provenance and confidence map:** every region is tagged `observed`, `inferred` or `guessed`/`retrieved`/`generated`, with a confidence value. The player can render low-confidence regions as a soft gradient or fog.
- **Audio section:** layered tracks plus sound events (section 7). The audio section must exist from format v1 even if empty.
- **Interaction rules:** a small declarative rule list (events, triggers, branches). Any scripting must run in a **sandbox** (no file, network or system access).
- **POV presets:** first person per actor, third-person follow, overhead, free.
- **Streaming and LOD:** progressive loading (coarse first, refine as data arrives) and level-of-detail chunks.
- **Provenance and labeling:** a signed header stating the file is AI-reconstructed or AI-generated, with tool version and creation time. This is built into the format from v1.
- **Generation record:** seeds, model versions and stage versions, so the same file regenerates identically.

### 4.2 `.wedit` (edit bundle for common editors)
Common editors cannot open custom formats directly, so `.wedit` is a bundle of standard files:
- Video in a standard editing-friendly format
- **Separate audio stems:** one per character voice, music, ambience, foley, effects
- **Extra render layers:** depth, character/object masks, surface normals, so relighting, colour grading and background swaps are possible
- A **timeline file** (OpenTimelineIO, EDL or XML) with cuts in place
- Script and character data for reference
- Two levels: (1) **flat edit** (round-trip not required), (2) later **round-trip** where edits are stored as change-records (for example `voice_3 pitch +2 semitones`, `scene_12 warmer light`) that can be re-applied to the `.wreel` so interactivity is preserved.
- Later: "AI edits" (rewrite a line, re-shoot a scene at night, change a camera) that go back to the engine to regenerate only that part.

### 4.3 `.wscript` (long-script format)
A structured screenplay file that stays lossless and token-efficient:
- A **dictionary** defining each character, location, prop and long description once as an ID, referenced everywhere else (lossless, saves tokens).
- **Scenes** containing dialogue lines tagged with speaker IDs, actions, POV hints, sound cues and a **tension value (0-10)**.
- A **scene index** and short per-scene summaries so a model loads only the current scene's exact text plus summaries of the rest.
- A **continuity state** per scene (who is where, clothing, props, lighting, time of day) so scene 40 matches scene 4.
- Note for the agent: ordinary file compression (zip) does not reduce model tokens; token savings come from structure (IDs, chunking, summaries). Also offer lossless compression for storage.

---

## 5. CHARACTER BIBLE AND IDENTITY (prevents "one default face" collapse)

Problem: when identity is not specified, a generative model drifts to its most common face and voice (mode collapse). Fix: **identity is always an explicit input, never chosen by the model.**
- Every character has an `identity code` in a **character bible**: face seed/embedding, voice seed/embedding, appearance attributes, clothing, age range.
- Seeds are drawn randomly from a large synthetic face/voice space and **saved inside the file**, so regeneration is deterministic.
- **Distinctness check:** compute embedding distance between all characters in a film (faces and voices) and reject pairs closer than a configured threshold.
- **Diversity monitoring during training:** count unique identities produced per evaluation batch. If the count drops, flag collapse.
- **Default policy:** people reconstructed from uploaded footage are rendered with **synthetic, non-matching identities** by default. The replacement face must be verified as dissimilar to the original using a face-embedding similarity check. Allow style requests (for example "older, bearded"), but design the request parser so a named real person cannot be requested. A user's *own* likeness is a later feature behind a consent/liveness check.
- Faces are not the only identifiers: also consider body shape, clothing, gait, background and voice. Voices are **synthetic only**, with no cloning of real voices.

---

## 6. UNSEEN REGIONS: HONEST FILLING

When parts of a scene were never visible:
1. **Confidence map first.** Mark every region as seen, inferred or guessed. Render guessed regions as a soft colour gradient or fog so viewers can tell real from guessed.
2. **Progressive refinement within a time budget:**
   - Instantly show the gradient.
   - Within a configurable time limit (for example 5 s), try improved fill.
   - In the background, continue refining and swap in better results.
3. **Retrieval of reference images from the internet,** only from **open-licensed sources** (for example Wikimedia Commons, Openverse, Mapillary or similar) and **user-provided** images. Rules:
   - Verify a retrieved image **geometrically** against what the video actually shows (feature matching and pose consistency) before using it. A wrong image is worse than a gradient: discard it.
   - Never retrieve images of private homes or people.
   - Store each retrieved item's source URL and license in the region's provenance record.
4. A generative fill model produces the remainder, tagged `generated`.

---

## 7. AUDIO SYSTEM (layered, event-driven)

Audio runs side by side with video (after video v1 works), but the **audio track slot exists in the format from day one.**

- **Layers:** dialogue (one track per character voice), foley, ambience, effects, music.
- **Sound events:** every action in the world emits an event with attributes (`surface`, `footwear`, `weight`, `speed`, `weather`, `emotion`, `object`). A sound engine selects or generates the matching sound. Example: walking on wet concrete in rain sounds different from dry gravel. Footsteps, doors, impacts and rain on different materials must all be event-driven.
- **Mood layer:** each scene carries a tension value (0-10) that drives music and ambience (calm to scared to intense). The interactive player may change this live.
- **Positional audio:** in first person, sound is heard from that character's position; in third person, from the camera.
- **Mixing:** automatic ducking (music lowers under dialogue), loudness targets for phone, headphones and speakers.
- **Voices:** diarization and per-character voice embeddings; synthetic voices only; a pairwise distinctness check like faces.
- **Dubbing:** swap only the dialogue layer for other languages (Hindi, Marathi, Tamil and more) using synthetic voices; music and effects stay the same.
- **Accessibility:** subtitles and audio description as standard layers.
- **Licensing:** use generated sounds or properly licensed libraries and record the license in the manifest. Do not use commercial film music or sound libraries unless licensed.

---

## 8. THE PLAYER (separate from normal video players)

- **Web-first** using WebGPU (with a fallback), so it works on phones without an install. A desktop player can come later.
- Plays `.wreel` only. Implements the viewer menu, POV switching, free camera, confidence-map rendering (gradient/fog for guessed regions), layered audio mixing and progressive streaming.
- **Motion comfort mode:** smoother camera movement and narrower field of view for first-person views.
- Flat export (MP4) of any chosen camera path for easy sharing.
- Later: AI camera director (automatically picks good angles), watch-together (multiple viewers, different POVs), branching story editor, viewer analytics for creators, VR/AR.

---

## 9. DATA PIPELINE (drop-folder workflow)

Folder layout (data is never committed to Git):
```
data/
  inbox/        # drop new videos/images here
  raw/          # originals after ingestion, never edited
  processed/    # frames, depth, masks, etc. made by stages
  manifest.csv  # one row per file: id, hash, source, license, date_added, status, notes
  versions/     # frozen dataset versions: lists of file IDs
```
Ingest script behaviour:
1. Detect new files in `inbox/`; check integrity; hash to detect duplicates.
2. **Require a source and license note** for each file. If missing, hold the file in a `needs_info/` state until provided. This keeps the manifest honest.
3. Move the original to `raw/`; write the manifest row.
4. Run the processing stages into `processed/`.

Rules for changing data:
- **Add** by dropping files in `inbox/`.
- **Replace a bad file** by adding the new one and marking the old one `excluded` in the manifest. Do not overwrite.
- **Remove truly** (wrong license or request): delete the file and row, exclude it from future versions, and record that a retrain is needed if it was already used.
- **Freeze a dataset version** (a list of file IDs) before every training run so results are reproducible.

Data sources:
- **Synthetic scenes** generated in Blender or a game engine (build a generator): these provide exact ground-truth depth, camera paths, lighting and actor positions, which is ideal for reconstruction and POV training.
- Openly licensed and public-domain footage (Internet Archive, Blender open movies, Wikimedia Commons, Openverse, government media where terms allow), plus the user's own recordings, plus any additional footage the user adds with a recorded license/source note.
- Record each source's license in the manifest. Keep the manifest accurate whatever the source.

---

## 10. TRAINING STRATEGY

- Start from open pretrained components (depth/pose, segmentation/tracking, 3D reconstruction, a video diffusion or world model, TTS/voice models). Record the model name, version and license for each.
- Use LoRA/QLoRA, gradient checkpointing, mixed precision and offloading to fit low VRAM. Evaluate layer streaming where it applies, and only claim it works after measuring.
- A text LLM fine-tuned for the **script brain** (writes and structures `.wscript`, builds the character bible) can use the Soup CLI (`soup train` from one YAML config). Verify Soup's current docs before using it; it targets LLM fine-tuning, not video rendering.
- All training is driven by config files and dataset versions. Log experiments (config, dataset version, metrics, hardware, seeds).
- Use reproducible seeds. Save checkpoints. Support resuming training.

---

## 11. SAFETY HOOKS (designed in now, strengthened later)

Stronger misuse prevention will be added in a later phase, but the **hooks must exist from the start** so they can be plugged in cheaply:
- A `safety` stage interface that every input and output passes through (initially a pass-through that logs).
- The `AI-reconstructed` signed label in every `.wreel` and every export, including edited ones.
- Default synthetic identity replacement with the dissimilarity check (section 5).
- An identity/likeness request filter hook.
- An upload-rights and takedown hook (data fields reserved in the manifest and the file header).
- Sandboxed interaction scripts.
Do not remove or bypass these hooks to simplify code.

---

## 12. EVALUATION AND THE TEST LADDER

Define metrics in `/eval` and run them in CI where possible:
- **Geometry:** depth and pose error vs ground truth on synthetic scenes
- **Novel-view quality:** PSNR/SSIM/LPIPS (or better) on held-out views
- **Identity drift:** face-embedding stability of a character across frames/scenes
- **Voice drift:** speaker-embedding stability across lines/scenes
- **Distinctness:** minimum pairwise distance between characters (faces and voices)
- **Identity diversity:** number of unique identities per batch (collapse detector)
- **Location/continuity consistency:** do objects and lighting persist across scenes
- **Script adherence:** do generated scenes follow `.wscript`
- **Fill honesty:** fraction of rendered area that is guessed vs observed, and calibration of confidence
- **Performance:** player frame rate and load time on a mid-range phone

**Test ladder (for generation):** 10 seconds, then 1 minute, then 10 minutes, then 1 hour. A rung must pass its metric thresholds before moving to the next.

---

## 13. REPOSITORY STRUCTURE

```
/spec            # wreel, wedit, wscript specs (versioned, open)
/pipeline        # stage framework, job runner, cache
/stages          # one folder per stage (ingest, frames, camera, reconstruct, actors, fill, audio, package, export, generate, safety)
/player          # web player (TypeScript, WebGPU)
/tools           # ingest CLI, dataset versioning, Blender synthetic-scene generator
/training        # configs, scripts, experiment logs (no weights committed)
/eval            # metrics and benchmark runners
/docs            # ARCHITECTURE.md, MODELS.md, DATA.md, ROADMAP.md
DECISIONS.md  CLAUDE.md  README.md  .gitignore (data/, weights, outputs)
```

---

## 14. MULTI-AGENT PLAN

Create a `CLAUDE.md` in the root containing: the vision, the architecture rules (stages communicate through files, every stage has a mock), the format specs location, the "never commit data or weights" rule, the safety-hook rule, the "do not invent APIs" rule, coding standards, and how to run tests.

Run parallel agents, each in **its own git worktree and branch** to avoid file collisions. Suggested agents:
1. **format-agent:** specs and read/write libraries for `.wreel`, `.wedit`, `.wscript`
2. **pipeline-agent:** stage framework, job runner, caching, mocks
3. **player-agent:** web player and viewer menu
4. **data-agent:** ingest CLI, manifest, dataset versions, Blender synthetic generator
5. **recon-agent:** camera/depth/reconstruction/actor stages (start with mock plus a small real model)
6. **audio-agent:** audio stems, sound-event engine, mixing, dubbing hooks (after video v1)
7. **eval-agent:** metrics, benchmarks, test ladder harness

Only run 2-3 agents at a time. Review and merge each branch before starting the next batch. Independent tasks only run in parallel (for example, format and data); tasks touching the same files run sequentially.

---

## 15. PHASES AND ACCEPTANCE CRITERIA (stop and verify after each)

**Phase 0: Foundation.** Repo, CLAUDE.md, folder structure, DECISIONS.md, hardware check, MODELS.md skeleton, CI setup. Acceptance: repo builds, tests run, hardware recorded.

**Phase 1: Formats and player skeleton.** Write specs for `.wreel`, `.wedit`, `.wscript`. Build read/write libraries and validators. Build a minimal web player that opens a hand-made synthetic `.wreel`, switches between at least free camera and one follow camera, and renders a confidence-map gradient. Acceptance: round-trip tests pass; a sample `.wreel` plays in the browser; format has audio slot and signed provenance header.

**Phase 2: Data pipeline and synthetic generator.** Drop-folder ingest, manifest, dataset versions, Blender generator producing scenes with ground-truth depth/camera/actors. Acceptance: dropping a file in `inbox/` produces a manifest row and processed outputs; a frozen dataset version can be reproduced.

**Phase 3: Reconstruction v1 (static, then dynamic).** Stages camera/reconstruct/actors with mock and real implementations. Acceptance: on synthetic scenes, depth/pose and novel-view metrics meet configured thresholds; a short real video becomes a free-camera `.wreel`.

**Phase 4: POV and unseen-region filling.** First/third-person per tracked actor; confidence map; time-boxed progressive fill; license-tagged retrieval with geometric verification. Acceptance: viewer can follow any tracked actor; guessed regions are visibly marked; retrieved images carry source/licence records.

**Phase 5: Audio.** Stems, sound-event engine, layered mixing, positional audio, mood layer. Acceptance: footsteps change with surface and weather; first-person audio differs from third-person; stems export separately.

**Phase 6: `.wedit` export and flat MP4.** Acceptance: a `.wedit` opens in at least one common editor with stems, layers and timeline intact.

**Phase 7: Script brain and character bible.** `.wscript` tooling, character bible with seeds and distinctness checks, fine-tuned script model. Acceptance: a 10-minute script loads scene by scene with a continuity state; character distinctness checks pass.

**Phase 8: Generation and the test ladder.** Generate 10 s, then 1 min, then 10 min scenes from `.wscript`; run identity, voice and continuity metrics. Acceptance: each rung meets thresholds.

**Phase 9: Interaction, branching, round-trip edits, safety hardening.** Strengthen the safety stage, upload-rights flow, likeness filter, round-trip `.wedit` changes. Acceptance: interactive branches work in the player; safety hooks are enforced, not pass-through.

**Phase 10: 1-hour film attempt and public benchmark.** Run the full ladder; publish specs, benchmark results and a short paper.

---

## 16. ALGORITHM CREATION POLICY (you are allowed and expected to invent when needed)

This project has problems with no ready-made solution (for example: honest confidence-aware filling, sound-event selection, identity-distinctness sampling, world-consistency across a 1-hour film, token-efficient script chunking). When no existing method fits, **design and create a new algorithm yourself.** Do not stall, and do not force an ill-fitting library.

Process for every new algorithm:
1. **Check first (briefly):** look for an existing method or paper. Record what you found, or "nothing suitable found", in `docs/ALGORITHMS.md`.
2. **Design note before code:** problem statement, why existing methods fail here, the idea, pseudocode, complexity (time and memory, with low-VRAM limits in mind), known failure cases.
3. **Baseline:** implement the simplest existing approach (or a naive one) so the new algorithm has something to beat.
4. **Implement behind a stage interface** so it can be swapped out. Give it a mock version and unit tests.
5. **Measure** with the metrics in section 12 on synthetic ground-truth scenes. Include at least one ablation (what happens if each part is removed).
6. **Label status:** `experimental` until it beats the baseline on the agreed metrics; then `adopted`. Record results and the decision in `DECISIONS.md`.
7. **Be honest:** never claim an algorithm works, or report numbers, that you did not measure. If it fails, record that and keep the baseline.

Inventing algorithms is encouraged. Inventing facts is not: never invent APIs, library functions, model names, licenses or benchmark numbers (see section 17).

---

## 17. WORKING RULES

- Write clean, typed, well-named code; comment only where the reason is not obvious.
- Prefer boring, proven solutions for ordinary engineering; use section 16 when a genuinely new algorithm is needed. Keep stages small and replaceable.
- Write specs before code for any format.
- Do not invent APIs, model names, licenses or benchmark numbers. Check official docs or mark assumptions in `DECISIONS.md`. (Creating new algorithms is allowed under section 16.)
- Do not add features outside the current phase; put them in `docs/ROADMAP.md`.
- Never commit data, videos, images or model weights.
- Do not remove or bypass the safety hooks or the provenance label.
- At the end of each phase: run all tests, summarise what was built, list deviations in `DECISIONS.md`, and list any manual steps I must do myself (for example downloading model weights, setting up Blender, GPU or cloud access).

Begin with Phase 0.
