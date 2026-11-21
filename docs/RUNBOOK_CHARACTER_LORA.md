# Runbook: character LoRA (v5) train -> render -> deliver

The reproducible chain for the D-9.26 motion-preserving character adapter.
Every command runs from the repo root (`D:\new_IDEA\worldreel`) and relies only
on repo files -- nothing in `C:\Users\li\AppData\Local\Temp` (that directory is
wiped on server restarts).

## Background

* **v12** (`character_consistency_v2`) = base Wan2.1-VACE, sharp and moving, but
  identity drifts (mean 0.8804 / min 0.7860, 8 frames below 0.80).
* **v4** (`character_lora_v4`) = adapter trained against a still portrait: identity
  and stability are excellent (0.9256 / 0.9105, 0 frames below 0.80) but the
  output is **a shaking still image** -- motion 0.414 mean grey levels, 211/239
  frames changed by < 0.5, face Laplacian 66 -> 15. This is the defect the user
  rejected.
* **v5** (`character_lora_v5`) = adapter trained against real rendered motion
  (`worldreel-data-v12render`), with a gated reward (identity, but a checkpoint is
  rolled back if motion or face sharpness falls below 75% of the base model's).

Acceptance: keep v12's motion **and** v4's identity/stability.

## 0. Rules

* One Kaggle job at a time (weekly 30 h T4 quota).
* Never `fetch` while a kernel is RUNNING -- fetch only after COMPLETE/ERROR.
* `--requeue-after 30` re-pushes only a still-QUEUED kernel; it never re-runs an
  ERROR one.
* `run` now uploads the code dataset automatically when the tracked code changed
  (`tools/kaggle_runner.py:ensure_code_pushed`), so a new repo module is never
  missing inside the kernel.

## 1. Train (Kaggle, T4)

```powershell
python -m tools.kaggle_runner run training/kaggle/jobs/character_lora_v5.py --scene data:vacecast --scene data:facemodels --scene data:v12render --wait --requeue-after 30
```

The job writes `out/lora_v5.log` (target take scores, reward curve),
`out/lora_v5_reward.json`, `out/lora_v5_report.json` (baseline / best / final
identity, motion, sharpness, gates, `roundtrip_ok`), `out/lora_v5_final.mp4` and
the adapter snapshots `out/lora_adapter_best/`, `out/lora_adapter_stepNNN/`.

Check the kernel state with `python -m tools.kaggle_runner status <job>`; fetch
only once it reports COMPLETE or ERROR:

```powershell
python -m tools.kaggle_runner fetch training/kaggle/jobs/character_lora_v5.py
```

**Read the report before rendering.** It carries the baseline motion/sharpness
the gates compare against and the identity the adapter actually reached.

## 2. Publish the adapter

```powershell
python -m tools.kaggle_runner push-data --name lorav5 runs/kaggle/character_lora_v5/out
```

## 3. Render with the adapter

```powershell
python -m tools.kaggle_runner run training/kaggle/jobs/character_render_lora_v5.py --scene data:vacecast --scene data:facemodels --scene data:lorav5 --wait
python -m tools.kaggle_runner fetch training/kaggle/jobs/character_render_lora_v5.py
```

Artefacts in `runs/kaggle/character_render_lora_v5/out/`:

* `lora_v5_face_consistency.mp4` -- 240 raw frames (5 segments x 48), 16 fps.
* `lora_v5_face_consistency.json` -- per-segment identity, `motion_mean`,
  `motion_median`, `motion_max`, `still_fraction`, `adapter_strength`.
* `lora_v5_face_consistency.log` -- the render log.

`adapter_strength` (in the job's `args`) scales the LoRA delta linearly:
0 = base model, 1 = trained adapter. The D-9.25 sweep found identity peaks at
0.5 and motion degrades above ~0.75 -- if the report shows the adapter froze
the pose again, lower the strength and re-render (one job at a time).

## 4. Local lossless chain (D-9.21 / D-9.22 / D-9.23)

FFV1 `.mkv` intermediates, one single x264 CRF 16 delivery encode. A render
never passes through a lossy intermediate.

```powershell
python -m stages.generate.face_stabilize --video runs/kaggle/character_render_lora_v5/out/lora_v5_face_consistency.mp4 --output runs/kaggle/character_render_lora_v5/out/lora_v5_stab.mkv
python -m stages.generate.cut_smooth --video runs/kaggle/character_render_lora_v5/out/lora_v5_stab.mkv --output runs/kaggle/character_render_lora_v5/out/lora_v5_dissolved.mkv --dissolve 6 --bounds 48 96 144 192
& D:\ffmpeg\bin\ffmpeg.exe -y -v error -i runs/kaggle/character_render_lora_v5/out/lora_v5_dissolved.mkv -c:v libx264 -crf 16 -pix_fmt yuv420p runs/kaggle/character_render_lora_v5/out/lora_v5_dissolved.mp4
```

The cut bounds are the segment boundaries in the **raw** render (48/96/144/192);
after the 6-frame dissolve they land at 42/84/126/168 in the 216-frame output.

## 5. Verification battery

```powershell
python -m tools.v2_check runs/kaggle/character_render_lora_v5/out/lora_v5_dissolved.mp4
python -m tools.identity_curve runs/kaggle/character_render_lora_v5/out/lora_v5_dissolved.mp4 runs/kaggle/portraits_run3_merged/C01_0.png 48 90 132 174
python -m stages.generate.video_analyzer --video runs/kaggle/character_render_lora_v5/out/lora_v5_dissolved.mp4 --portrait runs/kaggle/portraits_run3_merged/C01_0.png --output runs/analysis/lorav5_final_analyzer.json
```

`tools.v2_check` is the single source of truth for motion: it prints the mean
frame-to-frame difference and the still-frame count, so no separate probe
script is needed.

**Gates (from the D-9.25/D-9.26 measurements):**

| Metric | v12 base | v4 adapter | v5 target |
| --- | --- | --- | --- |
| identity mean / min | 0.8804 / 0.7860 | 0.9256 / 0.9105 | >= 0.90, no frame < 0.80 |
| motion (mean frame diff) | 2.468 | 0.414 | >= 1.0, still frames near 0 |
| face sharpness (Laplacian, raw) | 66.2 | 15.3 | >= 40 |
| analyzer issues | 0 | 0 | 0 |
| cuts detected | 4 | 0 | 0 (geometrically continuous) |

Quantitative gates are necessary but **not sufficient**: the user's visual
verdict is the acceptance criterion. Inspect the delivered frames directly.

## 6. Package and deliver

```powershell
python stages/generate/package_wreel_video.py --video runs/kaggle/character_render_lora_v5/out/lora_v5_dissolved.mp4 --portrait runs/kaggle/portraits_run3_merged/C01_0.png --output runs/generated/S017_lorav5.wreel --title "S017 Face-Consistent 13.5s (LoRA v5 motion-preserving)"
python -m formats validate runs/generated/S017_lorav5.wreel
Copy-Item -Force runs\kaggle\character_render_lora_v5\out\lora_v5_dissolved.mp4 runs\videos\S017_lorav5_13s5.mp4
python -m stages.generate.video_annotator --video runs/kaggle/character_render_lora_v5/out/lora_v5_dissolved.mp4 --portrait runs/kaggle/portraits_run3_merged/C01_0.png --output runs/videos/S017_lorav5_13s5_annotated.mp4
```

Then record the outcome in `DECISIONS.md` and
`AI_VIDEO_GENERATION_KNOWLEDGE_BASE.md`, and commit.

## 7. A/B reference numbers

| Build | identity mean / min | motion | sharpness (raw) | frames < 0.80 |
| --- | --- | --- | --- | --- |
| v12 base | 0.8804 / 0.7860 | 2.468 | 66.2 | 8 |
| v4 adapter | 0.9256 / 0.9105 | 0.414 | 15.3 | 0 |
| v5 adapter | pending | pending | pending | pending |
