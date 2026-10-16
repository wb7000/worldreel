# Data

Status: Phase 2 tooling in place. **No media, frames or weights are ever committed to Git.**
Only the small text records (`manifest.csv`, `removals.csv`, `versions/*.json`) may be committed.

## Folder layout
```
data/
  inbox/          drop new videos/images here, each with <file>.info.json      (ignored)
  needs_info/     files waiting for a source + license note                      (ignored)
  raw/            originals after ingestion, never edited                        (ignored)
  processed/      stage outputs, one folder per file id                          (ignored)
  rejected/       duplicates and unreadable files, each with a .reason.txt       (ignored)
  synthetic/      Blender scenes with ground truth (tools/synth.py)              (ignored)
  manifest.csv    one row per file                                       (may be committed)
  removals.csv    tombstones of removed files, for retrain bookkeeping   (may be committed)
  versions/       frozen dataset versions (JSON lists of ids + hashes)   (may be committed)
```
The data root defaults to `./data`; use `--root` or `$WORLDREEL_DATA` to put it elsewhere
(for example a bigger drive).

## Adding a file
1. Copy the video (or image) into `data/inbox/`.
2. Next to it, create `<file name>.info.json`:
   ```json
   {"source": "where it came from (URL, 'my own recording', ...)",
    "license": "CC0-1.0 / CC-BY-4.0 / public-domain / owner ...",
    "upload_rights": "owner | licensed | public-domain | synthetic | unknown",
    "notes": "optional"}
   ```
3. Run `python -m tools.data ingest`.

What ingest does: checks the file type and that ffprobe can read it, hashes it (SHA-256),
rejects exact duplicates, requires `source` and `license` (otherwise the file waits in
`needs_info/`), refuses No-Derivatives licences (reconstruction is a derivative work), moves the
original to `raw/<id>.<ext>` and writes the manifest row. A held file is released with
`python -m tools.data info <id> --source ... --license ...` or by adding the sidecar next to it in
`needs_info/` and re-running ingest.

## manifest.csv
| Column | Meaning |
|---|---|
| `id` | `f` + first 12 hex digits of the SHA-256 |
| `hash` | full SHA-256 of the original |
| `source`, `license` | from the sidecar (required) |
| `date_added` | UTC timestamp |
| `status` | `needs_info`, `active` or `excluded` |
| `notes` | sidecar notes, or why a file is held/excluded |
| `original_name`, `path`, `bytes`, `media_type` | bookkeeping |
| `upload_rights`, `takedown_status` | safety hooks (brief §11), enforced from Phase 9 |

## Changing data
| Want to | Command | Effect |
|---|---|---|
| Add | drop in `inbox/` + `ingest` | new `active` row |
| Replace a bad file | add the new one, then `exclude <old id> --reason ...` | old row kept, marked `excluded`; old versions still reproduce |
| Remove for real (licence, takedown) | `remove <id> --reason ... --yes` | file + row deleted, processed outputs deleted, tombstone in `removals.csv`, **"RETRAIN NEEDED"** if a frozen version used it |
| Freeze a version | `freeze <name>` | `versions/<name>.json` with every active id + hash; never overwritten |
| Prove a version reproduces | `verify <name>` | checks every file exists with the same hash |

## Processing
`python -m tools.data process` runs the stage pipeline (`pipeline/default.toml`) on every active
video into `processed/<id>/<stage>/`. Finished stages are cached by content, so re-running only
redoes what changed. `--mock` runs everything with mocks (no ffmpeg needed).

## Synthetic scenes (Blender)
```bash
python -m tools.synth render --seed 1          # data/synthetic/seed0001: rgb, depth, gt.json
python -m tools.synth check data/synthetic/seed0001
python -m tools.synth to-inbox data/synthetic/seed0001   # MP4 + CC0 sidecar into inbox
```
`gt.json` holds exact camera intrinsics and per-frame poses, actor tracks and lights, in WorldReel
coordinates (Y up). `check` back-projects depth through the cameras and verifies the result is
geometrically consistent (nothing below the floor, cross-frame reprojection, actors facing their
motion).

## Sources
Synthetic scenes (Blender generator, exact ground truth); openly licensed / public-domain footage
(Internet Archive, Blender open movies, Wikimedia Commons, Openverse, government media where terms
allow); the owner's own recordings. Every item's license is recorded in the manifest.
