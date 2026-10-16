"""Read, write, validate and load `.wscript` long-form scripts (spec/wscript.md).

The scene language (`.wsl`) parses into typed statements and serializes back losslessly for files
in canonical form.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from formats.common import (
    FormatError,
    Report,
    check_version,
    dump_json,
    load_json,
    read_container,
    schema_check,
    sha256_hex,
    write_container,
)

FORMAT = "wscript"
SUPPORTED_VERSION = (1, 0)
FORMAT_VERSION = "1.0"
HEADER = "wscript.json"
SCHEMA = "wscript-header-1.schema.json"

POV_MODES = ("first", "third", "over", "wide", "free")
_ID = r"[CLPTS][A-Za-z0-9_]{1,31}"
_CHAR_ID = r"C[A-Za-z0-9_]{1,31}"
_DIALOGUE = re.compile(rf"^({_CHAR_ID})(?:\s+\(([^()]*)\))?\s*:\s*(\S.*)$")
_KV = re.compile(r"^([a-z_][a-z0-9_]*)=(\S+)$")
_ANY_ID = re.compile(rf"^{_ID}$")


# --------------------------------------------------------------------------- statements


@dataclass(frozen=True)
class SceneHeader:
    scene_id: str
    params: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Dialogue:
    speaker: str
    text: str
    manner: str | None = None


@dataclass(frozen=True)
class Action:
    text: str


@dataclass(frozen=True)
class Sound:
    cue: str
    params: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Pov:
    mode: str
    target: str | None = None


@dataclass(frozen=True)
class Tension:
    value: int


@dataclass(frozen=True)
class Comment:
    text: str


@dataclass(frozen=True)
class Blank:
    pass


Statement = SceneHeader | Dialogue | Action | Sound | Pov | Tension | Comment | Blank


class WslSyntaxError(FormatError):
    def __init__(self, line_no: int, message: str) -> None:
        super().__init__(f"line {line_no}: {message}")
        self.line_no = line_no


def _params(tokens: list[str], line_no: int) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for token in tokens:
        match = _KV.match(token)
        if not match:
            raise WslSyntaxError(line_no, f"expected key=value, got {token!r}")
        out.append((match.group(1), match.group(2)))
    return tuple(out)


def parse_scene(text: str) -> list[Statement]:
    """Parse a `.wsl` scene. Tolerates extra spaces and trailing whitespace (normalised)."""
    statements: list[Statement] = []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for no, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if not line:
            statements.append(Blank())
            continue
        sigil, rest = line[0], line[1:].strip()
        if sigil == "@":
            tokens = rest.split()
            if (
                len(tokens) < 2
                or tokens[0] != "scene"
                or not re.fullmatch(r"S[A-Za-z0-9_]{1,31}", tokens[1])
            ):
                raise WslSyntaxError(no, "expected '@scene S<id> [key=value ...]'")
            statements.append(SceneHeader(tokens[1], _params(tokens[2:], no)))
        elif sigil == "!":
            if not rest:
                raise WslSyntaxError(no, "empty action")
            statements.append(Action(rest))
        elif sigil == "~":
            tokens = rest.split()
            if not tokens:
                raise WslSyntaxError(no, "empty sound cue")
            statements.append(Sound(tokens[0], _params(tokens[1:], no)))
        elif sigil == ">":
            tokens = rest.split()
            if not tokens or tokens[0] not in POV_MODES or len(tokens) > 2:
                raise WslSyntaxError(no, f"expected '> <{'|'.join(POV_MODES)}> [ID]'")
            statements.append(Pov(tokens[0], tokens[1] if len(tokens) == 2 else None))
        elif sigil == "%":
            if not re.fullmatch(r"\d{1,2}", rest) or int(rest) > 10:
                raise WslSyntaxError(no, "tension must be an integer 0-10")
            statements.append(Tension(int(rest)))
        elif sigil == "#":
            statements.append(Comment(rest))
        else:
            match = _DIALOGUE.match(line)
            if not match:
                raise WslSyntaxError(no, f"unrecognised statement {line!r}")
            manner = match.group(2).strip() if match.group(2) else None
            statements.append(Dialogue(match.group(1), match.group(3), manner or None))
    headers = [s for s in statements if isinstance(s, SceneHeader)]
    first = next((s for s in statements if not isinstance(s, Blank | Comment)), None)
    if len(headers) != 1 or first is not headers[0]:
        raise WslSyntaxError(1, "a scene needs exactly one @scene header, before any statement")
    return statements


def _fmt_params(params: tuple[tuple[str, str], ...]) -> str:
    return "".join(f" {k}={v}" for k, v in params)


def serialize_scene(statements: list[Statement]) -> str:
    out: list[str] = []
    for s in statements:
        match s:
            case SceneHeader(scene_id, params):
                out.append(f"@scene {scene_id}{_fmt_params(params)}")
            case Dialogue(speaker, text, manner):
                out.append(f"{speaker} ({manner}): {text}" if manner else f"{speaker}: {text}")
            case Action(text):
                out.append(f"! {text}")
            case Sound(cue, params):
                out.append(f"~ {cue}{_fmt_params(params)}")
            case Pov(mode, target):
                out.append(f"> {mode} {target}" if target else f"> {mode}")
            case Tension(value):
                out.append(f"% {value}")
            case Comment(text):
                out.append(f"# {text}" if text else "#")
            case Blank():
                out.append("")
    return "\n".join(out) + "\n"


def text_refs(text: str) -> list[str]:
    """IDs referenced as `{ID}` in a text; `{{` and `}}` are literal braces."""
    refs: list[str] = []
    i = 0
    while i < len(text):
        two = text[i : i + 2]
        if two in ("{{", "}}"):
            i += 2
            continue
        if text[i] == "{":
            end = text.find("}", i)
            if end == -1:
                raise FormatError(f"unclosed '{{' in {text!r}")
            refs.append(text[i + 1 : end])
            i = end + 1
            continue
        if text[i] == "}":
            raise FormatError(f"stray '}}' in {text!r} (write '}}}}' for a literal brace)")
        i += 1
    return refs


def expand_text(text: str, dictionary: Mapping[str, Mapping[str, Any]]) -> str:
    """Replace `{ID}` with the dictionary name (or full text for `T…` entries)."""
    lookup = _flat_dictionary(dictionary)
    out: list[str] = []
    i = 0
    while i < len(text):
        two = text[i : i + 2]
        if two in ("{{", "}}"):
            out.append(two[0])
            i += 2
        elif text[i] == "{":
            end = text.index("}", i)
            out.append(lookup.get(text[i + 1 : end], text[i : end + 1]))
            i = end + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _flat_dictionary(dictionary: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    flat: dict[str, str] = {}
    for section in ("characters", "locations", "props"):
        for ident, entry in dictionary.get(section, {}).items():
            flat[ident] = str(entry.get("name", ident))
    for ident, value in dictionary.get("text", {}).items():
        flat[ident] = str(value)
    return flat


# --------------------------------------------------------------------------- script


@dataclass
class Script:
    """`header` is the `wscript.json` object; `scenes` maps scene ID to parsed statements.
    The writer fills each index entry's `path` and `sha256`."""

    header: dict[str, Any]
    scenes: dict[str, list[Statement]]
    report: Report = field(default_factory=Report)

    def scene_entry(self, scene_id: str) -> dict[str, Any]:
        for entry in self.header["scenes"]:
            if entry["id"] == scene_id:
                return dict(entry)
        raise KeyError(scene_id)


def _entries_for(script: Script) -> dict[str, bytes]:
    header = dict(script.header)
    entries: dict[str, bytes] = {}
    index: list[dict[str, Any]] = []
    for entry in header["scenes"]:
        scene_id = entry["id"]
        if scene_id not in script.scenes:
            raise FormatError(f"scene {scene_id} is in the index but has no statements")
        data = serialize_scene(script.scenes[scene_id]).encode("utf-8")
        path = f"scenes/{scene_id}.wsl"
        entries[path] = data
        index.append({**entry, "path": path, "sha256": sha256_hex(data)})
    missing = set(script.scenes) - {e["id"] for e in header["scenes"]}
    if missing:
        raise FormatError(f"scenes not in the index: {sorted(missing)}")
    header["scenes"] = index
    entries[HEADER] = dump_json(header)
    return entries


def write_wscript(
    path: Path, script: Script, *, as_zip: bool = True, validate: bool = True
) -> None:
    entries = _entries_for(script)
    if validate:
        report = Report()
        _validate_entries(entries, report)
        if not report.ok:
            raise FormatError(
                "refusing to write invalid .wscript:\n" + "\n".join(map(str, report.errors))
            )
    write_container(path, entries, [HEADER], as_zip=as_zip)


def read_wscript(path: Path) -> Script:
    entries = read_container(path)
    report = Report()
    scenes = _validate_entries(entries, report)
    header = load_json(entries[HEADER], HEADER)
    return Script(header=header, scenes=scenes, report=report)


def validate_wscript(path: Path) -> Report:
    try:
        return read_wscript(path).report
    except FormatError as exc:
        report = Report()
        report.error("unreadable", str(exc))
        return report


def _validate_entries(entries: Mapping[str, bytes], rep: Report) -> dict[str, list[Statement]]:
    if HEADER not in entries:
        raise FormatError("wscript.json is missing")
    header = load_json(entries[HEADER], HEADER)
    if not isinstance(header, dict):
        raise FormatError("wscript.json must be a JSON object")
    version = check_version(header, FORMAT, SUPPORTED_VERSION, rep)
    schema_check(header, SCHEMA, rep, lenient_unknown=version[1] > SUPPORTED_VERSION[1])
    scenes: dict[str, list[Statement]] = {}
    if not rep.ok:
        return scenes
    dictionary = header["dictionary"]
    known = set(_flat_dictionary(dictionary))
    characters = set(dictionary["characters"])
    locations = set(dictionary["locations"])
    props = set(dictionary["props"])
    listed = {e["path"] for e in header["scenes"]}
    for extra in sorted(set(entries) - listed - {HEADER}):
        rep.warning("unlisted_entry", "file is not referenced by the scene index", extra)
    seen: set[str] = set()
    for entry in header["scenes"]:
        sid, where = entry["id"], f"scenes.{entry['id']}"
        if sid in seen:
            rep.error("duplicate_id", f"duplicate scene id {sid!r}", where)
        seen.add(sid)
        data = entries.get(entry["path"])
        if data is None:
            rep.error("missing_entry", f"{entry['path']} is missing", where)
            continue
        if sha256_hex(data) != entry["sha256"]:
            rep.error("hash_mismatch", "scene file does not match its sha256", where)
        try:
            statements = parse_scene(data.decode("utf-8"))
        except (UnicodeDecodeError, FormatError) as exc:
            rep.error("wsl_syntax", str(exc), where)
            continue
        scenes[sid] = statements
        header_stmt = next(s for s in statements if isinstance(s, SceneHeader))
        if header_stmt.scene_id != sid:
            rep.error(
                "scene_id_mismatch", f"@scene {header_stmt.scene_id} in file for {sid}", where
            )
        if entry.get("location") and entry["location"] not in locations:
            rep.error("bad_reference", f"unknown location {entry['location']!r}", where)
        _check_continuity(entry["continuity"], characters, locations, props, rep, where)
        _check_statements(statements, known, characters, rep, where)
    return scenes


def _check_continuity(
    cont: Mapping[str, Any],
    characters: set[str],
    locations: set[str],
    props: set[str],
    rep: Report,
    where: str,
) -> None:
    for cid, state in cont.get("characters", {}).items():
        if cid not in characters:
            rep.error("bad_reference", f"continuity: unknown character {cid!r}", where)
        if "location" in state and state["location"] not in locations:
            rep.error("bad_reference", f"continuity: unknown location {state['location']!r}", where)
        for pid in [*state.get("wearing", []), *state.get("holding", [])]:
            if pid not in props:
                rep.error("bad_reference", f"continuity: unknown prop {pid!r}", where)


def _check_statements(
    statements: list[Statement], known: set[str], characters: set[str], rep: Report, where: str
) -> None:
    for s in statements:
        texts: list[str] = []
        match s:
            case SceneHeader(_, params):
                for key, value in params:
                    if key == "loc" and value not in known:
                        rep.error(
                            "bad_reference", f"@scene loc={value} is not in the dictionary", where
                        )
                    if key == "tension" and (not value.isdigit() or int(value) > 10):
                        rep.error("bad_tension", f"@scene tension={value} must be 0-10", where)
            case Dialogue(speaker, text, _):
                if speaker not in characters:
                    rep.error("bad_reference", f"unknown speaker {speaker!r}", where)
                texts.append(text)
            case Action(text):
                texts.append(text)
            case Pov(_, target):
                if target is not None and target not in characters:
                    rep.error("bad_reference", f"POV target {target!r} is not a character", where)
            case _:
                pass
        for text in texts:
            try:
                refs = text_refs(text)
            except FormatError as exc:
                rep.error("bad_text", str(exc), where)
                continue
            for r in refs:
                if not _ANY_ID.match(r) or r not in known:
                    rep.error("bad_reference", f"unknown reference {{{r}}}", where)


# --------------------------------------------------------------------------- model context


@dataclass
class SceneContext:
    """What a model needs to work on one scene: only this scene's full text."""

    scene_id: str
    title: str
    continuity: dict[str, Any]
    dictionary: dict[str, Any]
    scene_text: str
    other_summaries: list[tuple[str, str]]

    def to_prompt(self) -> str:
        lines = [f"# Scene {self.scene_id}: {self.title}", "", "## Dictionary (used here)"]
        for section, entries in self.dictionary.items():
            for ident, entry in entries.items():
                value = entry if isinstance(entry, str) else entry.get("name", "")
                desc = "" if isinstance(entry, str) else entry.get("description", "")
                lines.append(f"- {ident} ({section}): {value}" + (f" — {desc}" if desc else ""))
        lines += ["", "## Continuity at scene start", dump_json(self.continuity).decode().strip()]
        lines += ["", "## Other scenes"] + [f"- {sid}: {s}" for sid, s in self.other_summaries]
        lines += ["", "## Scene text", self.scene_text.rstrip()]
        return "\n".join(lines) + "\n"


def load_context(script: Script, scene_id: str) -> SceneContext:
    entry = script.scene_entry(scene_id)
    statements = script.scenes[scene_id]
    text = serialize_scene(statements)
    used: set[str] = set()
    for s in statements:
        match s:
            case Dialogue(speaker, body, _):
                used.add(speaker)
                used.update(text_refs(body))
            case Action(body):
                used.update(text_refs(body))
            case Pov(_, target) if target:
                used.add(target)
            case SceneHeader(_, params):
                used.update(v for k, v in params if k == "loc")
    cont = entry["continuity"]
    used.update(cont.get("characters", {}))
    for state in cont.get("characters", {}).values():
        used.update([*state.get("wearing", []), *state.get("holding", [])])
        if "location" in state:
            used.add(state["location"])
    if entry.get("location"):
        used.add(entry["location"])
    full = script.header["dictionary"]
    subset = {
        section: {k: v for k, v in full[section].items() if k in used}
        for section in ("characters", "locations", "props", "text")
    }
    others = [(e["id"], e["summary"]) for e in script.header["scenes"] if e["id"] != scene_id]
    return SceneContext(scene_id, entry["title"], cont, subset, text, others)
