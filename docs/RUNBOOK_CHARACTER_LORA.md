# Runbook: character delivery (render -> assemble -> verify -> package)

The reproducible chain for the D-9.30 delivery render: Wan2.1-VACE with **six
reference views of the same person, normalised in-step, and no adapter**. Every
command runs from the repo root (`D:\new_IDEA\worldreel`) and relies only on
repo files -- nothing in `C:\Users\li\AppData\Local\Temp` (that directory is
wiped on server restarts).

> **D-9.30 (current recipe).** The user rejected the D-9.28 delivery
> (`S017_refs6`) for blur and inconsistency. The D-9.29 probe found the cause in
> the *conditioning set*, not in the model or the post-chain: the six references
> were 512x512 squares with mixed lighting, face scale and backgrounds, and
> diffusers' `preprocess_conditions` letterboxes each one onto a white canvas
> (176 px white bars at 480p). The render job therefore (a) normalises the
> references in-step with `tools/prepare_refs.py` (output aspect, one face
> scale, one tone), (b) uses the strengthened prompt/negative, and (c) exports
> at `quality=10`. There is **no refine pass**: the low-denoise V2V "sharpen"
> was measured and refuted (identity 0.92 -> 0.77, sharpness *fell*). Measured
> on the probe's clip: sharpness 91 -> 442 encoded (729 pre-encode), identity
> unchanged, face-scale cv 3x tighter, lighting sd 2.51 -> 1.72.

## Background

* **v12** (`character_consistency_v2`) = base Wan2.1-VACE with four references:
  sharp and moving, but the identity drifts inside every segment (mean 0.8804 /
  min 0.7860, 8 frames below 0.80).
* **v4** (`character_lora_v4`) = LoRA trained against a still portrait: identity
  and stability are excellent (0.9256 / 0.9105, 0 frames below 0.80) but the
  output is **a shaking still image** -- motion 0.414 mean grey levels, 211/239
  frames changed by < 0.5, face Laplacian 66 -> 15. This is the defect the user
  rejected.
* **v5** (`character_lora_v5`) = LoRA trained against real rendered motion: every
  checkpoint scored below the base model and the saved winner was a **no-op**
  (all 360 LoRA B matrices zero), so it produced nothing.
* **The scheduled adapter** (D-9.27 probe): `late75` is visually the base model
  (no identity gain), `late50` is the frozen portrait again. The adapter's
  identity gain and its pose-freezing are the same effect, so the LoRA
  direction is closed.
* **Six references, no adapter** (D-9.28 probe, 2x2 on the same clip):
  {4 refs, 6 refs} x {seed 42, 43}. Six references halve the within-segment
  drift on the same seed and lift the worst frame by +0.10, with the motion and
  the sharpness intact -- the one lever that does not freeze the pose.

Acceptance: keep v12's motion **and** v4's identity/stability.

## 0. Rules

* One Kaggle job at a time (weekly 30 h T4 quota).
* Never `fetch` while a kernel is RUNNING -- fetch only after COMPLETE/ERROR.
* `--requeue-after 30` re-pushes only a still-QUEUED kernel; it never re-runs
  an ERROR one.
* `run` uploads the code dataset automatically when the tracked code changed
  (`tools/kaggle_runner.py:ensure_code_pushed`), so a new repo module is never
  missing inside the kernel.
* **Fetch before re-pushing the same job.** Kaggle keeps only the newest kernel
  version's outputs, so the first half has to be on disk before the second half
  is pushed.

## 1. Render the first half (segments 43-45, ~3.0 h)

```powershell
python -m tools.kaggle_runner run training/kaggle/jobs/character_render_refs6.py --scene data:vacecast --scene data:facemodels --wait --requeue-after 30
```

The job renders `segments` segments of `frames_per_segment - 1` generated frames
(48 each, the conditioned portrait frame is dropped) and writes
`refs6_segs43-45.mp4`, `refs6_segs43-45.json`, `refs6_render.log` and the
normalised references (`refs_norm/C01_*_norm.png`) into the archived output
directory. The normalised references are written out on purpose: they are the
actual conditioning, and inspecting them is how the next defect gets found.

**Two things the job does that the D-9.28 version did not** (D-9.30):

* the references are normalised in-step (`tools/prepare_refs.py`, `tone=1.0`);
* the export runs at `quality=10`. The diffusers default is `quality=5.0`
  (variable bitrate), and it cost the probe 40% of the new detail in the first
  encode alone (729 -> 442). The soft D-9.28 delivery lost nothing in that
  encode (91.6 -> 91.5), which is why the defect never showed up.

Check the kernel state with `python -m tools.kaggle_runner status <job>`; fetch
only once it reports COMPLETE or ERROR:

```powershell
python -m tools.kaggle_runner fetch training/kaggle/jobs/character_render_refs6.py
```

**Read the JSON before rendering the rest.** It carries per segment: identity
mean / min / first / last, drift per frame, self-consistency against the
segment's own median face, face sharpness, peak GPU memory. A segment with a
negative drift and a low minimum is re-rendered on its own seed instead of
shipping.

**Keep the half.** Kaggle keeps only the newest kernel version's outputs, so
the first half is copied out of the fetch directory (the runner also moves the
previous fetch aside to `runs/kaggle/character_render_refs6.prev`):

```powershell
New-Item -ItemType Directory -Force runs\analysis\refsnorm_half1 | Out-Null
Copy-Item -Force runs\kaggle\character_render_refs6\out\* runs\analysis\refsnorm_half1\
```

## 2. Render the second half (segments 46-47, ~2.0 h)

Edit the job's args (same file, after the first half is fetched):

```python
"segment_seeds": [46, 47],
"segments": 2,
```

then run and fetch exactly as in step 1. The output is named from the seeds that
were used (`refs6_segs46-47.mp4`).

```powershell
New-Item -ItemType Directory -Force runs\analysis\refsnorm_half2 | Out-Null
Copy-Item -Force runs\kaggle\character_render_refs6\out\* runs\analysis\refsnorm_half2\
```

## 3. Join the halves (no re-encode)

```powershell
python tools/join_clips.py runs/analysis/refsnorm_raw.mp4 runs/analysis/refsnorm_half1/refs6_segs43-45.mp4 runs/analysis/refsnorm_half2/refs6_segs46-47.mp4
```

`tools/join_clips.py` uses a **stream copy** when the clips share their stream
parameters (the encoded bits are copied, nothing is decoded or re-encoded) and
falls back to a **lossless FFV1 re-encode** when they do not. A lossy re-encode
is never used: D-9.22/23 forbid passing a render through a lossy intermediate.

The joined clip is the raw timeline: 5 segments x 48 frames = 240 frames, with
the segment boundaries at frames 48 / 96 / 144 / 192.

## 4. Local lossless chain (D-9.21 / D-9.22 / D-9.23 / D-9.32)

FFV1 `.mkv` intermediates, one single x264 CRF 16 delivery encode. A render
never passes through a lossy intermediate.

**Assembly (D-9.32).** The five takes are independent seeds with no narrative
order, so the boundaries are placed by a permutation+trim search over the pair
jump matrix (worst boundary jump 0.222 -> 0.119), not by seed order. The tool
prints the new cut bounds — pass them to every later stage, because once the
jumps are small the landmark auto-detection stops finding them.

**Stabilizer settings for the D-9.30 footage** (rehearsed on the half-1 render,
`runs/analysis/contact_sheets/stab_cmp_f024.png`): the defaults
(`sigma 2.0 / window 1.5 / detail 0.3`) were tuned on the old *soft* render and
cost the new detail 17.5% (489.6 -> 403.9 Laplacian) for only 0.5 px of
placement correction. The light setting keeps the correction (0.24 px mean, the
new footage barely jitters) and *raises* measured sharpness to 648.9 because the
higher high-frequency reinjection (0.5) restores blend softness; the native-res
crop sheet shows genuine texture, not crunch. Identity stays at 0.9069 / 0.8518
with 0 frames < 0.80.

**The dissolve (D-9.32 fix).** `cut_smooth`'s blend weight now spans exactly
0..1: each window opens on the pure outgoing frame and closes on the pure
incoming one, with a single smoothstep driving both the landmark path and the
alpha. The old `(i+1)/(k+1)` alpha topped out at 0.857 and snapped — a pop at
every window exit the user heard as lag, and its partial-weight blend
double-imaged the face ("blur when the frame comes").

```powershell
python tools/assemble_segments.py --video runs/analysis/refsnorm_raw.mp4 --output runs/analysis/refsnorm_asm.mkv --bounds 48 96 144 192 --order 3 4 2 1 0 --trims 2:5 5:2 5:5 0:2
python -m stages.generate.face_stabilize --video runs/analysis/refsnorm_asm.mkv --output runs/analysis/refsnorm_asm_stab.mkv --sigma 1.0 --window 1.0 --detail 0.5
python -m stages.generate.cut_smooth --video runs/analysis/refsnorm_asm_stab.mkv --output runs/analysis/refsnorm_asm_dissolved.mkv --dissolve 6 --bounds 46 84 125 168
& D:\ffmpeg\bin\ffmpeg.exe -y -v error -i runs/analysis/refsnorm_asm_dissolved.mkv -c:v libx264 -crf 16 -pix_fmt yuv420p runs/videos/S019_asmfix_11s875.mp4
```

The assembly outputs 214 frames (240 minus the trims) with cuts at
46/84/125/168; after the four 6-frame dissolves the delivery is 190 frames
(11.875 s at 16 fps) with the windows at output 40/72/107/144.

## 5. Verification battery

```powershell
python -m tools.v2_check runs/videos/S019_asmfix_11s875.mp4
python -m tools.clip_consistency.py runs/videos/S019_asmfix_11s875.mp4
python -m tools.identity_curve runs/videos/S019_asmfix_11s875.mp4 runs/kaggle/portraits_run3_merged/C01_0.png 40 72 107 144
python -m stages.generate.video_analyzer --video runs/videos/S019_asmfix_11s875.mp4 --portrait runs/kaggle/portraits_run3_merged/C01_0.png --output runs/analysis/s019_final_analyzer.json
```

`tools.v2_check` is the single source of truth for motion: it prints the mean
frame-to-frame difference and the still-frame count, so no separate probe
script is needed.

**Gates (from the D-9.25/D-9.26/D-9.28/D-9.30/D-9.31/D-9.32 measurements):**

| Metric | v12 base | v4 adapter | refs6 (D-9.28, rejected) | refsnorm (D-9.31, rejected) | **asmfix (D-9.32, delivered)** |
| --- | --- | --- | --- | --- | --- |
| identity mean / min | 0.8804 / 0.7860 | 0.9256 / 0.9105 | 0.8967 / 0.7637 | 0.9176 / 0.8466 | **0.9186 / 0.8606, 0 frames < 0.80** |
| motion (mean frame diff) | 2.468 | 0.414 | 3.522 | 1.917, 0 still | **1.903, 0 still** |
| face sharpness (Laplacian, encoded) | 66.2 | 15.3 | 58.6 | 549.5 (min 183.7) | **568.4 (min 164.9)** |
| window-exit frame diff | - | - | - | 6.97 / 10.98 / 9.24 / 5.24 (the "lag"/"cut") | **1.10 / 2.16 / 1.28 / 2.15** (interior ~1.6) |
| window sharpness (face Lap) | - | - | - | 167-310 (the "blur") | **347-391** |
| face-scale cv (scale walk) | - | - | 0.039 (probe) | 0.032 (IOD) / 0.020 (height) | **0.033 (IOD) / 0.019 (height)** |
| lighting L sd (per-segment jump) | - | - | 2.51 (probe) | cut flash -4.53, gains within 1.8% | **gains within 1.8%, exit steps ≈ interior** |
| analyzer issues | 0 | 0 | 0 | 0 (flicker index 0.008) | **0** |
| PTS | - | - | - | 62.50 ms exact | **62.50 ms exact, 0 irregular** |

Quantitative gates are necessary but **not sufficient**: the user's visual
verdict is the acceptance criterion. Inspect the delivered frames directly
(`tools/contact_sheet.py`, `tools/contact_sheet_range.py`), especially every
dissolve window: endpoints must be pixel-continuous with their raw
neighbours and the interior single-imaged.

## 6. Package and deliver

```powershell
python stages/generate/package_wreel_video.py --video runs/videos/S019_asmfix_11s875.mp4 --portrait runs/kaggle/portraits_run3_merged/C01_0.png --output runs/generated/S019_asmfix.wreel --title "S019 Face-Consistent 11.875s (reordered assembly, fixed dissolve)"
python -m formats validate runs/generated/S019_asmfix.wreel
python -m stages.generate.video_annotator --video runs/videos/S019_asmfix_11s875.mp4 --portrait runs/kaggle/portraits_run3_merged/C01_0.png --output runs/videos/S019_asmfix_11s875_annotated.mp4
```

Then record the outcome in `DECISIONS.md` and
`AI_VIDEO_GENERATION_KNOWLEDGE_BASE.md`, and commit.

## 7. A/B reference numbers

| Build | identity mean / min | motion | sharpness (raw) | frames < 0.80 | analyzer |
| --- | --- | --- | --- | --- | --- |
| v12 base (4 refs) | 0.8804 / 0.7860 | 2.631 | 66.2 | 8 | 0 issues |
| v4 adapter | 0.9256 / 0.9105 | 0.484 | 15.3 | 0 | 0 issues |
| **refs6 (6 refs, no adapter)** | **0.8967 / 0.7637** | **3.522** | **58.6** | **1** | **0 issues** |

The single frame below 0.80 is inside the fourth dissolve window (two poses blended); the frames of
that window show a clean single-face transition. Per-segment identity and drift are in
`runs/analysis/refs6_half1/` and `runs/analysis/refs6_half2/` (the kernel reports) and in
`DECISIONS.md` D-9.28.

## 8. Closed: the LoRA chain

The training and adapter render steps live in the git history of this file
(D-9.24 - D-9.27). They are kept for the record and must not be repeated: the
adapter's identity gain and its pose-freezing are the same effect, and the
"best" v5 checkpoint was a no-op. The reasoning is in `DECISIONS.md` D-9.27 and
D-9.28.
