# `.wedit` — WorldReel edit bundle

**Format version: 1.0** · Status: draft for Phase 1 (container + manifest only; media production is
Phase 6) · Schema: `spec/schemas/wedit-manifest-1.schema.json`

Common video editors cannot open `.wreel`. A `.wedit` is a **bundle of standard files** an editor
can import directly, plus a small manifest that ties them back to the world they came from.

## 1. Container

Two equivalent forms:

| Form | Use |
|---|---|
| **Directory** `name.wedit/` | Working form. Editors import the files inside directly. |
| **ZIP** `name.wedit` | Transport form (same layout, same rules as `.wreel` §1). |

Required entries: `wedit.json` (manifest) and `signature.json` (Ed25519 over the exact bytes of
`wedit.json`; same structure as `.wreel` §4.1 but `"signed": "wedit.json"`). Every other file is
listed in `wedit.json` → `files` with SHA-256, byte size and media type.

## 2. Recommended layout

```
name.wedit/
  wedit.json
  signature.json
  video/main.mov              # editing-friendly codec (e.g. ProRes or DNxHR), or .mp4 for review
  stems/dlg_C01.wav           # one per character voice
  stems/music.wav  stems/ambience.wav  stems/foley.wav  stems/effects.wav
  layers/depth/000001.exr     # image sequences, one folder per layer
  layers/mask_a1/000001.png
  layers/normals/000001.exr
  timeline/main.otio          # OpenTimelineIO; EDL / FCPXML MAY be added alongside
  script/script.wscript       # optional reference copy
  PROVENANCE.txt              # plain-text copy of the AI label, for people opening the folder
```

## 3. Manifest `wedit.json`

```jsonc
{
  "format": "wedit",
  "format_version": "1.0",
  "level": "flat",                       // "flat" (no round trip) | "round-trip" (Phase 9)
  "provenance": { ... },                 // same object as .wreel §4; ai_label carried over
  "source_wreel": { "sha256": "...", "title": "..." },
  "camera_path": "director",             // camera or POV preset id the video was rendered from
  "video": { "path": "video/main.mov", "fps": 24, "width": 1920, "height": 1080,
             "duration": 12.0, "codec": "prores_422" },
  "stems": [ { "id": "dlg_C01", "layer": "dialogue", "character_id": "C01",
               "path": "stems/dlg_C01.wav", "sample_rate": 48000, "channels": 1,
               "license": "synthetic" } ],
  "render_layers": [ { "id": "depth", "kind": "depth", "path": "layers/depth/",
                       "pattern": "%06d.exr", "encoding": "linear-metres" } ],
                       // kind: depth | mask | normals | albedo | confidence
  "timeline": [ { "format": "otio", "path": "timeline/main.otio" } ],  // otio | edl | fcpxml
  "script": { "path": "script/script.wscript" },
  "characters": [ { "id": "C01", "name": "..." } ],
  "changes": [],                         // round-trip change records, §4
  "files": { "path": { "sha256": "...", "bytes": 0, "media_type": "..." } },
  "extensions": {}
}
```

Rules:
- `provenance` is **required** and its `ai_label` MUST be carried over from the source `.wreel`.
  An edited bundle uses the `-and-edited` label variant. Exporters MUST also write
  `PROVENANCE.txt` and SHOULD burn or embed the label in the video metadata (Phase 6).
- Directory references in `render_layers[].path` end with `/`; every frame file inside is listed
  in `files`.

## 4. Change records (level `round-trip`, Phase 9)

Edits made in an external editor are captured as structured records that can be re-applied to
the `.wreel` so interactivity survives:

```json
{ "id": "ch1", "target": "audio.track:dlg_C01", "op": "pitch_shift", "value": 2, "unit": "semitones" }
{ "id": "ch2", "target": "scene:12", "op": "color_temperature", "value": -500, "unit": "kelvin" }
```

`target` grammar: `<section>[.<kind>]:<id>`. The `op` vocabulary is defined in Phase 9; v1 readers
preserve records they do not understand.

## 5. Versioning
As `.wreel` §13.
