# Data

Status: layout fixed now, tooling built in Phase 2. **No data is ever committed to Git.**

## Folder layout (git-ignored)
```
data/
  inbox/        # drop new videos/images here
  needs_info/   # files waiting for a source + license note
  raw/          # originals after ingestion, never edited
  processed/    # stage outputs (frames, depth, masks, ...)
  manifest.csv  # one row per file
  versions/     # frozen dataset versions: lists of file IDs
```

## manifest.csv columns (draft)
`id, hash, source, license, date_added, status, notes`, plus reserved safety fields
`upload_rights, takedown_status` (brief §11). Final schema in Phase 2.

Status values (draft): `needs_info, active, excluded, removed`.

## Rules
- **Add:** drop files in `data/inbox/`. A file without a source and license note is held in
  `needs_info/`.
- **Replace:** add the new file, mark the old one `excluded`. Never overwrite.
- **Remove (license problem or request):** delete file and row, exclude from future versions, and
  record that a retrain is needed if it was used.
- **Freeze** a dataset version before every training run.

## Sources
Synthetic scenes (Blender generator, exact ground truth); openly licensed / public-domain footage
(Internet Archive, Blender open movies, Wikimedia Commons, Openverse, government media where terms
allow); the owner's own recordings. Every item's license recorded in the manifest.
