"""The script model's task and its scoring (D-7.5): continue a `.wsl` scene.

Prompt = the scene context `load_context` gives (characters, places and props used, continuity at
scene start, one-line summaries of the other scenes) with the scene text cut at some line, plus
an instruction to write the next `TARGET_LINES` lines. The answer is plain `.wsl` lines.

Scoring needs no model and runs anywhere (tests, Kaggle):
- syntax: every output line parses as a `.wsl` statement and there is at least one;
- references: every speaker, POV target, state-change target and `{ID}` is in the script's
  dictionary (the model may not invent characters or props).
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from formats.common import FormatError
from formats.wscript import (
    WORLD,
    Action,
    Blank,
    Change,
    Comment,
    Dialogue,
    Pov,
    SceneHeader,
    Script,
    load_context,
    parse_scene,
    serialize_scene,
    text_refs,
)

TARGET_LINES = 8
MAX_PREFIX_LINES = 24

SYSTEM = (
    "You write screenplay scenes in WorldReel Scene Language (.wsl). One statement per line:\n"
    "- dialogue: `C01: text` or `C01 (manner): text`, where C01 is a character ID;\n"
    "- action / stage direction: `! text`;\n"
    "- in any text, `{ID}` refers to a dictionary entry, e.g. `! {C02} opens the door.`\n"
    "Use only character IDs and references from the dictionary in the prompt. Answer with the "
    "requested number of lines and nothing else."
)


def instruction(n: int = TARGET_LINES) -> str:
    return f"Continue the scene with the next {n} lines."


@dataclass(frozen=True)
class Example:
    script_title: str
    scene_id: str
    cut: int  # index of the first target statement
    prompt: str
    target: str

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.target},
        ]


def _body(stmts: list[Any]) -> list[Any]:
    return [s for s in stmts if not isinstance(s, SceneHeader | Blank)]


def make_example(script: Script, scene_id: str, cut: int, n: int = TARGET_LINES) -> Example:
    """`cut` indexes the scene's body statements (header and blank lines excluded)."""
    stmts = script.scenes[scene_id]
    header = next(s for s in stmts if isinstance(s, SceneHeader))
    body = _body(stmts)
    prefix = body[max(0, cut - MAX_PREFIX_LINES) : cut]
    target = body[cut : cut + n]
    ctx = load_context(script, scene_id)
    shown = dataclasses.replace(ctx, scene_text=serialize_scene([header, *prefix]))
    prompt = shown.to_prompt() + "\n" + instruction(len(target)) + "\n"
    return Example(
        script.header["title"],
        scene_id,
        cut,
        prompt,
        serialize_scene([header, *target]).split("\n", 1)[1].rstrip("\n"),
    )


def examples(
    script: Script, stride: int, rng: random.Random, min_prefix: int = 6
) -> Iterator[Example]:
    """Windows over every scene: a cut every `stride` statements (random offset per scene)."""
    for sid, stmts in script.scenes.items():
        body = _body(stmts)
        start = min_prefix + rng.randrange(stride)
        for cut in range(start, len(body) - TARGET_LINES + 1, stride):
            yield make_example(script, sid, cut)


@dataclass(frozen=True)
class Score:
    syntax_ok: bool
    refs_ok: bool
    lines: int
    problem: str = ""


def _strip_fences(text: str) -> str:
    lines = [ln for ln in text.strip().split("\n") if not ln.strip().startswith("```")]
    return "\n".join(ln.rstrip() for ln in lines).strip("\n")


def score(output: str, dictionary: dict[str, Any]) -> Score:
    """Is a model's answer valid `.wsl`, using only the script's IDs? Code fences are ignored
    (fair to a base model that wraps its answer)."""
    text = _strip_fences(output)
    try:
        stmts = [
            s for s in parse_scene("@scene S999\n" + text + "\n")[1:] if not isinstance(s, Blank)
        ]
    except FormatError as exc:
        return Score(False, False, 0, str(exc)[:120])
    if not stmts:
        return Score(False, False, 0, "empty")
    chars = set(dictionary.get("characters", {}))
    known = chars | set(dictionary.get("locations", {})) | set(dictionary.get("props", {}))
    known |= set(dictionary.get("text", {}))
    for s in stmts:
        try:
            match s:
                case Dialogue(speaker, body, _):
                    bad = [speaker] if speaker not in chars else []
                    bad += [r for r in text_refs(body) if r not in known]
                case Action(body):
                    bad = [r for r in text_refs(body) if r not in known]
                case Pov(_, target):
                    bad = [target] if target and target not in chars else []
                case Change(target, _):
                    bad = [target] if target != WORLD and target not in chars else []
                case Comment():
                    bad = []
                case _:
                    bad = []
        except FormatError as exc:
            return Score(True, False, len(stmts), str(exc)[:120])
        if bad:
            return Score(True, False, len(stmts), f"unknown {bad}")
    return Score(True, True, len(stmts))
