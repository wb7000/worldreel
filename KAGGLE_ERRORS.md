# KAGGLE ERRORS & FIXES

## Purpose
Record every Kaggle-related error encountered and the fix applied, so we never repeat them.

---

## Error 1: `KeyError: 'scaling_factor'` (CRITICAL — blocked job entirely)

**File:** `training/kaggle/jobs/character_consistency_final.py`

**Cause:** `pipe.vae.config` is a `FrozenDict` that may not contain `scaling_factor`. The code did:
```python
scaling_factor = vae_config["scaling_factor"]
```
and the except clause only caught `(AttributeError, TypeError)`, NOT `KeyError`.

**Fix:**
```python
try:
    scaling_factor = float(vae_config["scaling_factor"])
except (AttributeError, KeyError, TypeError):
    scaling_factor = 0.18215  # Default VAE scaling factor
```

---

## Error 2: `NameError: name 'pos' is not defined`

**Cause:** `pos` and `neg` (prompt embeddings) were referenced in `pipe()` but never computed. Missing `pipe.encode_prompt()` call.

**Fix:** Added before generation loop:
```python
pipe.text_encoder.to("cuda")
with torch.no_grad():
    pos, neg = pipe.encode_prompt(prompt=..., negative_prompt=..., do_classifier_free_guidance=True, device="cuda")
pipe.text_encoder.to("cpu")
pipe.text_encoder = None
gc.collect()
torch.cuda.empty_cache()
```

---

## Error 3: `NameError: name 'seg' is not defined` (loop variable mismatch)

**Cause:** Loop was `for s in range(...)` but code used `seg` in `if seg == 0:` and `manual_seed(42 + seg)`.

**Fix:** Changed all `seg` references to `s`.

---

## Error 4: `TypeError: 'Image' object is not subscriptable`

**Cause:** `prev_last_frame = out[-1]` stores a single PIL Image, then `prev_last_frame[-OVERLAP:]` tries to slice it.

**Fix:** Store list of frames:
```python
prev_last_frames = out[-OVERLAP:]   # list of frames
video = [*prev_last_frames, ...]     # unpack list
```

---

## Error 5: `torch.OutOfMemoryError: CUDA out of memory` (T4 15GB VRAM)

**Cause:** Too many frames (80) at high resolution (832×480) with VAE float32.

**Fixes:**
- `frames_per_segment`: 80 → 12
- Resolution: 832×480 → 512×288
- VAE dtype: float32 → float16
- Removed unused `ref_latents` encoding block
- Added `pipe.vae.enable_tiling()`
- Freed text encoder after prompt encoding

---

## Error 6: Kaggle API 500/503 on `kernels delete` during requeue

**File:** `tools/kaggle_runner.py`

**Cause:** Kaggle's API returns 500/503 Internal Server Error when deleting a kernel during requeue logic. The runner crashed instead of retrying.

**Fix:** Added retry logic (3 attempts, 10s delay) around `kernels delete`:
```python
for attempt in range(3):
    try:
        kaggle("kernels", "delete", f"{username()}/{kernel_slug(job)}", "-y")
        deleted = True
        break
    except RuntimeError as e:
        print(f"delete attempt {attempt + 1}/3 failed: {e}", flush=True)
        time.sleep(10)
```
Changed default `--requeue-after` from 20 → 30 minutes.

---

## Error 7: Kaggle cancels job during `uv venv` + `pip install` (~1 min)

**Cause:** Kaggle kills kernels that don't produce output or take too long during environment setup. Creating a uv venv + installing 12 packages takes 2–4 minutes, during which Kaggle cancels.

**Fix:** Use Kaggle's system Python (PyTorch pre-installed) instead of uv venv. Only `pip install` missing packages (`diffusers`, `opencv-python-headless`, `scipy`) on system Python. Add `flush=True` to all `log()`/`print()` calls so Kaggle sees output.

---

## Error 8: `fetch` on a RUNNING job cancels it

**Cause:** Calling `tools.kaggle_runner fetch` while the job is RUNNING tells Kaggle the job is complete, which cancels it.

**Rule:** NEVER call `fetch` until status is `COMPLETE` or `ERROR`. Only call `status` to check progress.

---

## Error 9: `&&` is not valid in PowerShell

**Cause:** PowerShell doesn't support `&&` as command separator.

**Fix:** Run `git add`, `git commit`, `git push` as separate commands.

---

## Error 10: `SyntaxError` from nested triple-quoted strings

**Cause:** A `"""..."""` docstring inside a `textwrap.dedent("""...""")` block prematurely closes the outer string.

**Fix:** Use `#` comments instead of `"""..."""` inside dedented blocks.

---

## Checklist Before Every Kaggle Push

- [ ] Syntax check: `python -c "import ast; ast.parse(open('file.py').read())"`
- [ ] No `&&` in PowerShell commands
- [ ] No nested `"""` inside `textwrap.dedent("""...""")`
- [ ] All `log()`/`print()` calls have `flush=True`
- [ ] `--requeue-after 30` (not 20, not 0 unless debugging)
- [ ] Delete old kernel first: `kaggle kernels delete <slug> -y`
- [ ] Push once, then ONLY call `status` — never `fetch` until COMPLETE/ERROR
- [ ] Wait for completion notification, do not poll excessively

---

## Output Location

```
D:\new_IDEA\worldreel\runs\kaggle\character_consistency_final\out\
  face_consistency.mp4    (generated video)
  face_consistency.json   (per-segment report)
  face_consistency.log    (full Kaggle job log)
  log.txt                 (Kaggle runner log)
```

---

## Last Known Working Config (2026-11-09)

- Job: `training/kaggle/jobs/character_consistency_final.py`
- Kaggle T4 GPU, system Python (no uv venv)
- 3 segments × 12 frames (1s each) at 512×288
- Face swap post-processing (YuNet + Delaunay + affine + seamlessClone)
- VAE: float16, tiling enabled
- Text encoder freed after prompt encoding
