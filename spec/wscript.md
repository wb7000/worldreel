# `.wscript` — WorldReel long-form script

**Format version: 1.1** · Status: draft (1.0 Phase 1; 1.1 Phase 7) · Schema (header):
`spec/schemas/wscript-header-1.schema.json` · Scene grammar: §4

A structured screenplay that stays **lossless** and **token-efficient** for language models.

> Token savings come from *structure*, not from ZIP: compression shrinks bytes on disk, but a model
> reads the decompressed text. `.wscript` saves tokens by (1) defining every character, location,
> prop and long description **once** in a dictionary and referring to it by short ID, (2) splitting
> the script into per-scene files, and (3) keeping a scene index with short summaries so a model
> loads only the current scene's full text plus summaries of the rest.

## 1. Container

Same two forms as `.wedit`: a directory `name.wscript/` (editing) or a ZIP `name.wscript`
(storage/transport; deflate gives lossless compression on disk).

```
name.wscript/
  wscript.json          # header: dictionary, scene index, continuity
  scenes/S001.wsl       # one scene per file, WorldReel Scene Language (§4)
  scenes/S002.wsl
```

No signature is required for `.wscript` (scripts are human-authored inputs, not AI outputs).

## 2. Header `wscript.json`

```jsonc
{
  "format": "wscript",
  "format_version": "1.1",
  "title": "The Courtyard",
  "language": "en",
  "dictionary": {
    "characters": { "C01": { "name": "Mira", "description": "...", "voice": "low, calm",
                             "identity_seed": 42 } },
    "locations":  { "L01": { "name": "Courtyard", "description": "..." } },
    "props":      { "P01": { "name": "red umbrella", "description": "..." } },
    "text":       { "T01": "A long reusable description used in several places ..." }
  },
  "scenes": [
    { "id": "S001", "path": "scenes/S001.wsl", "sha256": "...", "title": "Arrival",
      "summary": "Mira arrives in the rain and finds the gate locked.",
      "location": "L01", "time_of_day": "night", "tension": 3,
      "continuity": {
        "characters": { "C01": { "location": "L01", "wearing": ["P02"], "holding": ["P01"],
                                 "state": "wet, tired" } },
        "lighting": "street lamps, rain", "weather": "rain"
      } }
  ]
}
```

- `continuity` is the state **at the start** of the scene. Tools check that scene *n+1*'s state
  equals scene *n*'s start state plus its state changes (§4.2). Optional keys (1.1): `time`
  (time of day, world state) and `offscreen`: paths that changed between scenes without being
  written as a change, e.g. `["C01.wearing", "world.time"]` (a time jump). Paths:
  `<CharID>.location|wearing|holding|state` and `world.lighting|weather|time`.
- `identity_seed` links to the character bible; it is optional in v1.
- `identity` (1.1, optional, characters): the character-bible code — `seed`, the style request
  `requested` (allowlisted style words only, never a name), and the expanded `appearance` and
  `voice` vectors (each dimension 0–1). The vectors are a pure function of seed + request
  (`stages.generate.bible.expand`), so tools can check they were not edited by hand, and two
  characters of one script must be at least the configured distance apart (D-7.4).
- IDs: characters `C…`, locations `L…`, props `P…`, text `T…`, scenes `S…`, followed by digits
  or `[A-Za-z0-9_]` (pattern `^[CLPTS][A-Za-z0-9_]{1,31}$`). Prefixes are mandatory so a reference
  says what it points to.

## 3. Loading for a model (the reason this format exists)

`load_context(script, scene_id)` returns: the dictionary entries referenced by the scene, the
scene's continuity state, the full scene text, and the one-line summaries of every other scene.
Its size grows with the *current* scene plus one line per other scene, not with the whole script.

## 4. Scene file grammar — WorldReel Scene Language (`.wsl`)

UTF-8, `\n` line endings, one statement per line. The first character decides the statement:

| Line | Meaning | Example |
|---|---|---|
| `@scene <ID> [key=value …]` | Scene header, first line, exactly once | `@scene S001 loc=L01 time=night tension=3` |
| `<CharID>: <text>` | Dialogue | `C01: I told you not to come.` |
| `<CharID> (<manner>): <text>` | Dialogue with delivery note | `C02 (whisper): You knew I would.` |
| `! <text>` | Action / description | `! {C01} shakes the {P01} dry.` |
| `~ <cue> [key=value …]` | Sound cue | `~ rain intensity=heavy` |
| `> <pov> [<ID>]` | POV hint: `first`, `third`, `over`, `wide`, `free` | `> first C02` |
| `% <0-10>` | Tension change from this point | `% 7` |
| `= <ID\|world> <change> …` | State change (1.1, §4.2) | `= C01 holding+=P01 location=L02` |
| `# <text>` | Comment (kept, not performed) | `# TODO tighten` |
| empty line | Kept as a blank line (layout only) | |

Text rules:
- `{ID}` inside text refers to a dictionary entry (`{C01}`, `{P01}`, `{T01}`). A literal `{` is
  written `{{` and a literal `}` is written `}}`.
- Header `key=value` values have no spaces; use `_` or a dictionary reference.
- Known header keys: `loc` (location ID), `time` (time of day), `tension` (0–10). Unknown keys are
  kept.

### 4.1 Losslessness
The parser produces a list of typed statements; the serializer writes them back. For any file in
**canonical form** (single space after sigils and colons, `key=value` pairs in the order written,
no trailing whitespace, final newline), `serialize(parse(text)) == text` byte for byte. Files not
in canonical form normalise on the first round trip and are stable afterwards. Information is
never dropped: comments, blank lines and unknown header keys are preserved.

### 4.2 State changes (1.1)
`= <target> <change> …` records what changes, so continuity can be checked by replaying scenes.
A change is `key=value` (set), `key+=value` (add to a list) or `key-=value` (remove from a list).

| Target | Key | Operators | Value |
|---|---|---|---|
| character `C…` | `location` | `=` | location ID |
| | `wearing`, `holding` | `+=`, `-=` | prop ID |
| | `state` | `=` | text with `_` for spaces |
| `world` | `lighting`, `weather`, `time` | `=` | text with `_` for spaces |

Checks (by tools, not by the reader): a speaker must be at the scene's location when speaking; a
prop is worn or held by at most one character; `-=` removes something the character has.

## 5. Validation
- Every `{ID}` and every speaker/POV ID exists in the dictionary.
- Scene `@scene` ID matches the index entry; `sha256` matches the file.
- Tension values are 0–10.
- `continuity` references existing IDs.

## 6. Versioning
As `.wreel` §13. 1.1 adds the `=` statement and the continuity keys `time` and `offscreen`;
every 1.0 file is a valid 1.1 file. A 1.0 reader rejects scenes containing `=` lines.
