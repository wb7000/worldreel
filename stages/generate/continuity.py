"""Continuity tracking across `.wscript` scenes (A-7, spec/wscript.md §2, §4.2).

A scene's `continuity` is the world state at its start. Replaying the scene's `=` lines gives
the state at its end, which must equal the next scene's start state except for paths the next
scene declares as `offscreen` (time jumps). While replaying, speakers must be at the scene's
location and a prop can be worn or held by one character at a time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from formats.common import Report
from formats.wscript import WORLD, Change, Dialogue, SceneHeader, Script, Statement

LIST_KEYS = ("wearing", "holding")
WORLD_FIELDS = ("lighting", "weather", "time")


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    code: str
    message: str
    line: int  # 1-based line in the scene file; 0 = scene boundary


def scene_location(script: Script, scene_id: str) -> str | None:
    entry = script.scene_entry(scene_id)
    if entry.get("location"):
        return str(entry["location"])
    header = next(s for s in script.scenes[scene_id] if isinstance(s, SceneHeader))
    return next((v for k, v in header.params if k == "loc"), None)


def _owners(state: dict[str, Any]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for cid, ch in state.get("characters", {}).items():
        for key in LIST_KEYS:
            for pid in ch.get(key, []):
                owners.setdefault(pid, []).append(cid)
    return owners


def replay(
    start: dict[str, Any], statements: list[Statement], location: str | None
) -> tuple[dict[str, Any], list[Issue]]:
    """Apply the scene's state changes to `start`; returns the end state and the issues found."""
    state = copy.deepcopy(start)
    state.pop("offscreen", None)
    chars: dict[str, dict[str, Any]] = state.setdefault("characters", {})
    issues: list[Issue] = []
    for line, s in enumerate(statements, start=1):
        match s:
            case Dialogue(speaker, _, _):
                where = chars.get(speaker, {}).get("location")
                if location is not None and where != location:
                    issues.append(
                        Issue(
                            "error",
                            "speaker_absent",
                            f"{speaker} speaks but is at {where or 'no known location'}, "
                            f"not {location}",
                            line,
                        )
                    )
            case Change(target, ops):
                if target == WORLD:
                    for key, _, value in ops:
                        state[key] = value
                    continue
                ch = chars.setdefault(target, {})
                for key, op, value in ops:
                    if key in LIST_KEYS:
                        items: list[str] = ch.setdefault(key, [])
                        if op == "+=":
                            if value in items:
                                issues.append(
                                    Issue(
                                        "warning",
                                        "prop_already_had",
                                        f"{target} already has {value} ({key})",
                                        line,
                                    )
                                )
                            else:
                                items.append(value)
                        elif value in items:
                            items.remove(value)
                        else:
                            issues.append(
                                Issue(
                                    "error",
                                    "prop_not_held",
                                    f"{target} {key}-={value}, but {target} does not have it",
                                    line,
                                )
                            )
                    else:
                        ch[key] = value
                for pid, who in _owners(state).items():
                    if len(who) > 1:
                        issues.append(
                            Issue(
                                "error",
                                "prop_in_two_places",
                                f"{pid} is with {', '.join(sorted(who))} at once",
                                line,
                            )
                        )
            case _:
                pass
    return state, issues


def _paths(state: dict[str, Any]) -> dict[str, Any]:
    """Flatten a continuity state to comparable `path -> value` pairs (lists as sorted tuples)."""
    out: dict[str, Any] = {}
    for cid, ch in state.get("characters", {}).items():
        for key, value in ch.items():
            out[f"{cid}.{key}"] = tuple(sorted(value)) if isinstance(value, list) else value
        for key in LIST_KEYS:
            out.setdefault(f"{cid}.{key}", ())
    for key in WORLD_FIELDS:
        if key in state:
            out[f"world.{key}"] = state[key]
    return out


def diff(end: dict[str, Any], start: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    a, b = _paths(end), _paths(start)
    return {p: (a.get(p), b.get(p)) for p in sorted(set(a) | set(b)) if a.get(p) != b.get(p)}


def check(script: Script) -> tuple[Report, dict[str, list[Issue]]]:
    """Replay every scene in index order. Returns a report (`where` = scene and line) and the
    issues per scene."""
    report = Report()
    per_scene: dict[str, list[Issue]] = {}
    end: dict[str, Any] | None = None
    for entry in script.header["scenes"]:
        sid = entry["id"]
        start = entry["continuity"]
        issues: list[Issue] = []
        if end is not None:
            declared = set(start.get("offscreen", []))
            changed = diff(end, start)
            for path, (before, after) in changed.items():
                if path not in declared:
                    issues.append(
                        Issue(
                            "error",
                            "continuity_break",
                            f"{path}: {before!r} at the end of the previous scene, {after!r} here "
                            "(write a change, or list it in offscreen)",
                            0,
                        )
                    )
            for path in sorted(declared - set(changed)):
                issues.append(
                    Issue("warning", "offscreen_unused", f"{path} is listed but unchanged", 0)
                )
        end, found = replay(start, script.scenes[sid], scene_location(script, sid))
        issues += found
        per_scene[sid] = issues
        for i in issues:
            where = f"scenes.{sid}" + (f":{i.line}" if i.line else "")
            (report.error if i.level == "error" else report.warning)(i.code, i.message, where)
    return report, per_scene


def _set_path(state: dict[str, Any], path: str, value: Any) -> None:
    owner, key = path.split(".", 1)
    if owner == WORLD:
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
        return
    ch = state.setdefault("characters", {}).setdefault(owner, {})
    if value is None or value == ():
        ch.pop(key, None)
    else:
        ch[key] = list(value) if isinstance(value, tuple) else value


def propagate(script: Script) -> Script:
    """Fill each scene's start state from the previous scene's end state. Paths a scene lists in
    `offscreen` keep the values written in that scene's continuity. Returns a new script; the
    first scene's continuity is taken as written."""
    header = copy.deepcopy(script.header)
    end: dict[str, Any] | None = None
    for entry in header["scenes"]:
        written = entry["continuity"]
        if end is not None:
            declared = written.get("offscreen", [])
            own = _paths(written)
            start = copy.deepcopy(end)
            for path in declared:
                _set_path(start, path, own.get(path))
            if declared:
                start["offscreen"] = list(declared)
            entry["continuity"] = start
        loc = scene_location(script, entry["id"])
        end, _ = replay(entry["continuity"], script.scenes[entry["id"]], loc)
    return Script(header=header, scenes=script.scenes, report=script.report)
