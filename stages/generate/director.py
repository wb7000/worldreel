"""The director (A-13): per-line manner, emotion, intensity and spoken text for a scene.

An open instruction model reads the scene in context (`load_context`) and answers in strict JSON;
this module builds that prompt, validates the answer, checks that every spoken rewrite still says
what the script says (`meaning_check`, no model involved) and turns the result into a delivery
layer (spec/wscript.md §4.3). The model runs on Kaggle (`training/kaggle/jobs/director.py`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from formats.wscript import EMOTIONS, NONVERBALS, Dialogue, Script, expand_text, load_context

SYSTEM = (
    "You are a film director preparing actors. For every numbered dialogue line of the scene, "
    "decide how the character says it to the person they are talking to, given the situation, "
    "their relationship and the tension. Answer with JSON only."
)

STOP = set(
    (  # noqa: SIM905
        "a an the and or but so to of in on at for with by from as is are was were be been am i "
        "you he she it we they me him her us them my your his its our their this that these "
        "those there here what who whom which do does did not no yes if then than too very just "
        "may might can could will would shall should must let it's i'm you're we're"
    ).split()
)
FILLERS = set("uh um hm hmm oh ah well like okay ok yeah right huh eh mm".split())  # noqa: SIM905
WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    text = re.sub(r"\([a-z ]+\)|\[[a-z ]+\]", " ", text.lower())
    return WORD.findall(text)


def _content(text: str) -> list[str]:
    return [w for w in _words(text) if w not in STOP and w not in FILLERS]


def dialogue_lines(script: Script, scene_id: str) -> list[tuple[int, Dialogue, str]]:
    """(statement index, statement, text with IDs expanded) for every dialogue line."""
    d = script.header["dictionary"]
    return [
        (i, s, expand_text(s.text, d))
        for i, s in enumerate(script.scenes[scene_id])
        if isinstance(s, Dialogue)
    ]


def build_prompt(script: Script, scene_id: str) -> list[dict[str, str]]:
    names = {c: str(v["name"]) for c, v in script.header["dictionary"]["characters"].items()}
    ctx = load_context(script, scene_id).to_prompt()
    numbered = "\n".join(
        f"{i}: {names.get(s.speaker, s.speaker)} ({s.speaker})"
        + (f" [{s.manner}]" if s.manner else "")
        + f": {text}"
        for i, s, text in dialogue_lines(script, scene_id)
    )
    task = (
        f"{ctx}\n## Dialogue lines to direct\n{numbered}\n\n"
        "For each line give:\n"
        '- "index": the number before the line;\n'
        '- "manner": how it is said toward the listener, at most 6 words '
        '(e.g. "respectful, hesitant", "teasing", "irritated, clipped");\n'
        f'- "emotion": one of {", ".join(EMOTIONS)};\n'
        '- "intensity": 0.0 to 1.0;\n'
        '- "spoken": the same line as a real person would say it in this moment — the same '
        "meaning and the same facts, but natural speech: contractions, a hesitation, a repeated "
        "word or trailing off only where it fits; keep it about as long as the original; you may "
        "put one of (laugh) (chuckle) (sigh) (gasp) (cry) where it truly fits; no new names;\n"
        f'- "nonverbal": the list of those used, from {", ".join(NONVERBALS)}.\n'
        'Answer exactly as {"lines": [{...}, ...]} with one entry per line, in order.'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]


@dataclass(frozen=True)
class MeaningCheck:
    ok: bool
    content_recall: float
    length_ratio: float
    new_names: tuple[str, ...]


def meaning_check(original: str, spoken: str, names: set[str]) -> MeaningCheck:
    """Does the spoken rewrite still say what the script says? (A-13: no model involved.)
    `names` are names the rewrite may NOT add (characters absent from the scene); calling a
    present character by name is ordinary speech."""
    orig_c, spoken_c = _content(original), set(_content(spoken))
    recall = sum(w in spoken_c for w in orig_c) / len(orig_c) if orig_c else 1.0
    ow = max(1, len(_words(original)))
    ratio = len(_words(spoken)) / ow
    orig_words = set(_words(original))
    new = tuple(
        sorted(n for n in names if n.lower() in _words(spoken) and n.lower() not in orig_words)
    )
    ok = recall >= 0.8 and 0.8 <= ratio <= 1.8 and not new
    return MeaningCheck(ok, round(recall, 3), round(ratio, 3), new)


def parse_answer(text: str) -> list[dict[str, Any]]:
    """The JSON object in a model answer (code fences and chatter around it are ignored)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the answer")
    data = json.loads(text[start : end + 1])
    lines = data.get("lines") if isinstance(data, dict) else None
    if not isinstance(lines, list):
        raise ValueError('answer has no "lines" list')
    return [x for x in lines if isinstance(x, dict)]


def to_layer(
    script: Script, scene_id: str, answer: list[dict[str, Any]], source: str
) -> tuple[dict[str, Any], list[str]]:
    """Validated delivery layer for the scene, plus a list of problems (lines that fell back)."""
    chars = script.header["dictionary"]["characters"]
    loc = next((e.get("location") for e in script.header["scenes"] if e["id"] == scene_id), None)
    present = {
        c
        for c, st in next(e for e in script.header["scenes"] if e["id"] == scene_id)["continuity"]
        .get("characters", {})
        .items()
        if st.get("location") == loc
    } | {s.speaker for _, s, _ in dialogue_lines(script, scene_id)}
    names = {str(v["name"]) for c, v in chars.items() if c not in present}  # may not be added
    by_index = {int(x["index"]): x for x in answer if str(x.get("index", "")).isdigit()}
    lines: dict[str, Any] = {}
    problems: list[str] = []
    for i, _s, text in dialogue_lines(script, scene_id):
        x = by_index.get(i)
        if x is None:
            problems.append(f"{i}: missing")
            lines[str(i)] = {"manner": "", "emotion": "neutral", "intensity": 0.3, "spoken": text,
                             "nonverbal": [], "source": source, "meaning_ok": False}  # fmt: skip
            continue
        emotion = x.get("emotion") if x.get("emotion") in EMOTIONS else "neutral"
        if emotion != x.get("emotion"):
            problems.append(f"{i}: emotion {x.get('emotion')!r} -> neutral")
        try:
            intensity = min(max(float(x.get("intensity", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            intensity = 0.5
        spoken = str(x.get("spoken") or "").strip()
        check = meaning_check(text, spoken, names) if spoken else MeaningCheck(False, 0, 0, ())
        if not check.ok:
            problems.append(f"{i}: meaning check failed {check}")
        nonverbal = [n for n in x.get("nonverbal", []) if n in NONVERBALS]
        lines[str(i)] = {
            "manner": " ".join(str(x.get("manner", "")).split()[:6]),
            "emotion": emotion,
            "intensity": round(intensity, 2),
            "spoken": spoken if check.ok else text,
            "nonverbal": nonverbal if check.ok else [],
            "source": source,
            "meaning_ok": check.ok,
            "proposed": spoken if not check.ok else None,
        }
    return {"lines": lines}, problems
