# `.wreel` — WorldReel interactive world container

**Format version: 1.0** · Status: draft for Phase 1 · Schema: `spec/schemas/wreel-manifest-1.schema.json`

A `.wreel` file stores a *scene* rather than pixels: geometry, actors and their motion, lighting,
camera paths, sound, interaction rules, and an honest record of what was seen versus guessed.
The WorldReel player renders pixels from whatever view the viewer chooses.

The key words MUST, SHOULD and MAY are used as in RFC 2119.

---

## 1. Container

A `.wreel` file is a **ZIP archive** (PKWARE APPNOTE; deflate or store only, no encryption, no
ZIP64 requirement for files < 4 GiB). Any ZIP tool can list and extract it.

| Entry | Required | Purpose |
|---|---|---|
| `manifest.json` | yes | Everything except bulk data. UTF-8 JSON. |
| `signature.json` | yes (v1) | Ed25519 signature over the exact bytes of `manifest.json`. |
| everything else | as listed | Bulk data referenced from the manifest (`files` table). |

Rules:
- Entry paths use `/`, are relative, contain no `..`, no leading `/`, no drive letters.
- Every entry other than `manifest.json` and `signature.json` MUST be listed in `manifest.files`
  with its SHA-256 and byte size. Readers MUST reject an archive whose entries do not match the
  table (missing, extra, or wrong hash).
- Writers SHOULD produce deterministic archives: entries sorted by path, `manifest.json` first,
  `signature.json` second, fixed timestamp 1980-01-01 00:00:00. Then the same scene always produces
  byte-identical files (needed for the generation record).
- Readers MUST NOT execute anything found in the archive. There is no executable content in v1.

### 1.1 Why the signature covers `manifest.json` bytes

`manifest.json` contains the SHA-256 of every other entry, so signing its exact bytes signs the
whole archive without needing JSON canonicalisation. Readers verify the signature against the
bytes as stored, before parsing.

## 2. Conventions

- Units: metres, seconds, degrees (for `fov_deg` only), linear RGB in `[0, 1]`.
- Coordinate system: right-handed, **+Y up**, −Z forward (the same as glTF 2.0).
- Rotations: unit quaternions `[x, y, z, w]`.
- Time: seconds from the start of the scene; `duration` is the scene length.
- IDs: `^[A-Za-z][A-Za-z0-9_-]{0,63}$`, unique within their list.
- Binary data: little-endian.

## 3. Manifest (top level)

```jsonc
{
  "format": "wreel",
  "format_version": "1.0",
  "title": "Courtyard test",
  "duration": 12.0,
  "provenance": { ... },        // §4, required, signed
  "generation": { ... },        // §5, required
  "scene": { ... },             // §6
  "actors": [ ... ],            // §7
  "cameras": [ ... ],           // §8
  "pov_presets": [ ... ],       // §9
  "regions": [ ... ],           // §10 confidence / provenance map
  "audio": { ... },             // §11, required even if empty
  "interaction": { ... },       // §12
  "files": { "path": { "sha256": "...", "bytes": 123, "media_type": "..." } },
  "extensions": { }             // §14
}
```

## 4. Provenance header (`provenance`, required)

Every `.wreel` states that it is AI-made. This is part of the format from v1 and MUST NOT be
removed by editors or exporters; it is also copied into `.wedit` bundles and flat exports.

| Field | Type | Notes |
|---|---|---|
| `ai_label` | enum | `AI-reconstructed`, `AI-generated`, `AI-reconstructed-and-edited`, `AI-generated-and-edited` |
| `notice` | string | Human-readable label shown by the player, e.g. "AI-reconstructed scene. Not a recording." |
| `tool` | string | `"worldreel"` |
| `tool_version` | string | Version of the tool that wrote the file |
| `created_at` | string | RFC 3339 UTC timestamp |
| `source.kind` | enum | `video`, `script`, `synthetic`, `mixed` |
| `source.sha256` | string? | Hash of the source file when there is one |
| `identity_policy` | enum | `synthetic-replacement` (default for people from footage), `synthetic-only` (generated), `none` (no people) |
| `rights.upload_rights` | enum | `unknown`, `owner`, `licensed`, `public-domain`, `synthetic` |
| `rights.license` | string? | SPDX id or licence name of the source |
| `rights.takedown.status` | enum | `none`, `requested`, `removed` |
| `rights.takedown.contact` | string? | Where takedown requests go |

The `rights` fields are reserved hooks for the upload-rights and takedown flow (brief §11). In
Phase 1 they are written with defaults; enforcement comes in Phase 9.

### 4.1 `signature.json`

```json
{ "alg": "Ed25519", "public_key": "<base64 raw 32 bytes>", "signature": "<base64 64 bytes>",
  "signed": "manifest.json" }
```

The signature proves that the manifest (and therefore every hashed entry) has not changed since
the key holder signed it. It does **not** prove the content is true, and it does not stop someone
from stripping the signature; a reader that finds no valid signature MUST say so (the player shows
"unsigned / signature invalid"). Key distribution and trust lists are out of scope for v1.

## 5. Generation record (`generation`, required)

Enough to regenerate the same file.

```jsonc
{
  "seeds": { "global": 1234, "identity": 99 },
  "stages": [ { "name": "camera", "version": "0.1.0", "config_sha256": "..." } ],
  "models": [ { "name": "...", "version": "...", "license": "..." } ],
  "inputs": [ { "role": "video", "sha256": "..." } ]
}
```

## 6. Scene (`scene`)

```jsonc
{
  "bounds": { "min": [x,y,z], "max": [x,y,z] },
  "background": [r,g,b],
  "lighting": {
    "ambient": { "color": [r,g,b], "intensity": 0.3 },
    "lights": [ { "id": "sun", "type": "directional", "color": [1,1,1], "intensity": 1.0,
                  "direction": [x,y,z] } ]          // type: directional | point (with "position")
  },
  "layers": [ <geometry layer>, ... ]                // static geometry
}
```

### 6.1 Geometry layer

```jsonc
{
  "id": "static",
  "kind": "points",                 // v1: "points" only. Reserved: "gaussians", "mesh".
  "point_size": 0.05,               // world-space diameter hint (metres)
  "lods": [                         // ordered coarse → fine; progressive streaming
    { "level": 0, "path": "scene/static/lod0.pts", "count": 2000 },
    { "level": 1, "path": "scene/static/lod1.pts", "count": 20000 }
  ]
}
```

Each LOD is **independent**: LOD *n* is a complete representation on its own, so the player can
replace LOD *n* with LOD *n+1* once it arrives. (Additive LODs are a candidate for v1.x.)

### 6.2 Point chunk binary layout (`application/x-wreel-points`, `.pts`)

No header; the record count comes from the manifest. Record stride **20 bytes**:

| Offset | Type | Field |
|---|---|---|
| 0 | float32 × 3 | position x, y, z |
| 12 | uint8 × 3 | colour r, g, b (sRGB-encoded 0–255) |
| 15 | uint8 | provenance class (§10.1) |
| 16 | float32 | confidence in `[0, 1]` |

File size MUST equal `count × 20`.

## 7. Actors (`actors`)

People, animals, vehicles and objects that move independently and can be followed.

```jsonc
{
  "id": "a1",
  "type": "person",                 // person | animal | vehicle | object
  "label": "Walker A",
  "character_id": "C01",            // optional link to the character bible (.wscript)
  "identity": { "policy": "synthetic-replacement", "identity_seed": 42 },  // people only; §7.1
  "geometry": { <geometry layer, in the actor's local frame> },
  "eye_height": 1.65,               // metres above actor origin, for first-person
  "track": { "path": "actors/a1/track.json" }
}
```

Track file (`application/json`):
```json
{ "times": [0.0, 0.5], "positions": [[0,0,0],[0.7,0,0]], "rotations": [[0,0,0,1],[0,0,0,1]] }
```
`times` strictly increasing; arrays of equal length; the player interpolates linearly (positions)
and with slerp (rotations), holding the first/last value outside the range.

### 7.1 Identity hook
For `type: person`, `identity` is required. `policy` is `synthetic-replacement` for people
reconstructed from footage (rendered with a synthetic, verified-dissimilar identity) or
`synthetic` for generated characters. The dissimilarity check itself is implemented in Phase 7/9;
the field exists now so files written today carry the policy.

## 8. Cameras (`cameras`)

```jsonc
{
  "id": "director",
  "name": "Director's cut",
  "kind": "director",               // exactly one "director" camera if the scene has one; others "alternative"
  "keyframes": [ { "t": 0.0, "position": [x,y,z], "rotation": [x,y,z,w], "fov_deg": 50 } ]
}
```
Interpolation as for tracks; `fov_deg` interpolates linearly.

## 9. POV presets (`pov_presets`)

The viewer menu.

| `mode` | Extra fields | Meaning |
|---|---|---|
| `directors-cut` | `camera_id` | Play a camera path |
| `first-person` | `actor_id` | From the actor's eye height, looking along its forward axis |
| `third-person` | `actor_id`, `offset` [x,y,z] | Follow camera, offset in the actor's frame |
| `overhead` | `height` | Top-down view over scene bounds |
| `free` | — | Viewer-controlled camera |
| `interactive` | — | Free camera with interaction rules active |

Common optional fields: `id`, `name`, `fov_deg`, `comfort` (bool, default `true` for
`first-person`: smoothed motion and narrower field of view).

## 10. Confidence and provenance map

### 10.1 Provenance classes (per point and per region)

| Code | Name | Meaning |
|---|---|---|
| 0 | `observed` | Seen directly in the source |
| 1 | `inferred` | Derived from observations (e.g. interpolated between views) |
| 2 | `guessed` | Plausible filler with no direct evidence |
| 3 | `retrieved` | From a licensed reference image, geometrically verified |
| 4 | `generated` | Produced by a generative model |

### 10.2 Regions (`regions`)

```jsonc
{
  "id": "r_behind_wall",
  "bounds": { "min": [x,y,z], "max": [x,y,z] },
  "class": "guessed",
  "confidence": 0.2,
  "sources": [                      // required (non-empty) for "retrieved"
    { "kind": "retrieval", "url": "https://commons.wikimedia.org/...", "license": "CC-BY-SA-4.0",
      "verified": "geometric" },
    { "kind": "model", "model": "name", "version": "..." }
  ]
}
```

### 10.3 Rendering rule (normative for the WorldReel player)
Points whose class is not `observed` or whose confidence is below `0.5` MUST be visibly
distinguishable: the player blends them towards a fog colour in proportion to `1 − confidence`
and MAY add a gradient by class. A viewer setting MAY weaken the effect, but it is on by default.

## 11. Audio (`audio`, required)

The slot exists from v1 even when empty: `{ "tracks": [], "events": [], "mood": [] }`.

```jsonc
{
  "tracks": [ { "id": "dlg_c01", "layer": "dialogue", "character_id": "C01",
                "path": "audio/dlg_c01.ogg", "media_type": "audio/ogg", "start": 0.0,
                "license": "synthetic", "positional": true, "actor_id": "a1" } ],
                // layer: dialogue | foley | ambience | effects | music | description
  "events": [ { "t": 1.2, "type": "footstep", "actor_id": "a1", "position": [x,y,z],
                "attributes": { "surface": "concrete", "wet": true, "footwear": "boot",
                                "weight": 70, "speed": 1.4, "weather": "rain" } } ],
  "mood": [ { "t": 0.0, "tension": 2 }, { "t": 8.0, "tension": 7 } ],   // 0–10
  "mix": { "ducking": true, "loudness_target_lufs": -16 },
  "subtitles": [ { "lang": "en", "path": "audio/subs_en.vtt" } ]
}
```
`attributes` is an open string→(string|number|boolean) map; the Phase 5 sound engine defines the
vocabulary.

## 12. Interaction (`interaction`)

Declarative rules only. **There is no executable script in v1.** Any future scripting MUST run in a
sandbox with no file, network or system access and needs a new minor version.

```jsonc
{
  "rules": [
    { "id": "r1",
      "trigger": { "type": "time", "t": 5.0 },        // time | proximity | view_mode | select
      "action":  { "type": "set_tension", "value": 8 } // switch_pov | set_tension | play_event | branch
    }
  ],
  "branches": []                                      // reserved for Phase 9
}
```
Trigger fields: `time` → `t`; `proximity` → `actor_id`, `radius`; `view_mode` → `mode`;
`select` → `actor_id`. Action fields: `switch_pov` → `preset_id`; `set_tension` → `value`;
`play_event` → `event` (an audio event object); `branch` → `branch_id`.

## 13. Versioning and compatibility

- `format_version` is `MAJOR.MINOR`.
- A reader for version *M.n* MUST read any *M.m* file. Unknown fields from a newer minor version
  are ignored (and reported as a warning); unknown fields are therefore allowed only inside
  `extensions` for writers of the current version.
- A new MAJOR version is a breaking change. Readers upgrade older majors through explicit
  migration functions; they never guess.
- Every format change is logged in `DECISIONS.md`.

## 14. Extensions

`extensions` is an object keyed by reverse-DNS or project names (`"org.example.foo": {...}`).
Readers ignore extensions they do not understand.

## 15. Media types

| Use | Media type |
|---|---|
| Whole file | `application/x-wreel+zip` (provisional) |
| Point chunk | `application/x-wreel-points` |
| Track | `application/json` |

## 16. Out of scope for v1 (see ROADMAP)
Gaussian splat layers, meshes, spatial (octree) chunking, HTTP range streaming, branching story
graphs, embedded scripting, trust lists for signing keys.
