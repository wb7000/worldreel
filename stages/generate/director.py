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


EMOTION_GUIDE = """Emotions (pick the one that fits the moment, not the words alone):
- neutral: plain information, no feeling coloured into it
- tender: warm, gentle, caring, intimate
- joyful: happy, warm laughter, delight
- excited: thrilled, high energy, disbelief at good news
- comic_sarcastic: teasing, playful mockery, mischief
- dry_humour: deadpan, understated irony
- angry: irritated, cross, sharp, furious
- afraid: scared, panicked, urgent, nervous
- despair: hopeless, pleading, defeated
- crying: in tears, voice breaking
- whisper_tense: hushed, secretive, holding breath
- shocked_grief: sudden bad news, stunned sorrow"""

TALK_GUIDE = """How people really talk (the "spoken" field): nobody speaks like a book.
Use contractions (I am -> I'm, it is -> it's), start with a small word ("Oh,", "Well,", "Hey,",
"Look,", "Hm."), repeat a word when someone is moved or hesitant ("I... I can't"), trail off
("Just... walk."), call the listener by name when it feels natural, and put (laugh) (chuckle)
(sigh) (gasp) or (cry) where the feeling breaks through. Keep the meaning and the facts; keep it
about as long as the line. Only a one- or two-word line may stay unchanged."""

EXAMPLE = """Example (a different scene): an old fisherman, Joss, and his daughter Ama on a pier
in a storm; she has just told him she is leaving the village.
1: Joss (C01): You are leaving. After everything.
2: Ama (C02): I have to go, father. There is nothing here for me.
3: Joss (C01): Then go. I will not stop you.
Answer:
{"lines": [
 {"index": 1, "manner": "hurt, quietly bitter", "emotion": "despair", "intensity": 0.6,
  "spoken": "So you're leaving. After... after everything.", "nonverbal": []},
 {"index": 2, "manner": "pleading, guilty", "emotion": "crying", "intensity": 0.7,
  "spoken": "I have to go, Dad. (sigh) There's nothing here for me. Nothing.",
  "nonverbal": ["sigh"]},
 {"index": 3, "manner": "cold, proud, turning away", "emotion": "angry", "intensity": 0.5,
  "spoken": "Then go. Go. I won't stop you.", "nonverbal": []}
]}
Another (comedy): two friends, Pell and Rue, whose cake has just collapsed at a party.
1: Pell (C01): Well. That is not how it looked in the picture.
2: Rue (C02): It is fine. It is a modern cake.
Answer:
{"lines": [
 {"index": 1, "manner": "deadpan, amused", "emotion": "dry_humour", "intensity": 0.5,
  "spoken": "Well. That's... not how it looked in the picture.", "nonverbal": []},
 {"index": 2, "manner": "bluffing, laughing", "emotion": "comic_sarcastic", "intensity": 0.7,
  "spoken": "It's fine! (laugh) It's a... modern cake.", "nonverbal": ["laugh"]}
]}"""


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
        f"{EMOTION_GUIDE}\n\n{TALK_GUIDE}\n\n{EXAMPLE}\n\n"
        f"Now the real scene.\n{ctx}\n## Dialogue lines to direct\n{numbered}\n\n"
        "For EVERY numbered line give:\n"
        '- "index": the number before the line;\n'
        '- "manner": how it is said toward the listener, at most 6 words;\n'
        f'- "emotion": one of {", ".join(EMOTIONS)};\n'
        '- "intensity": 0.0 to 1.0;\n'
        '- "spoken": the line as this person would really say it now (see the guide);\n'
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


# --------------------------------------------------------------------------- stage 2 (v3, D-8.8c)

REWRITE_EXAMPLE = """Example. Line: "You are leaving. After everything."
(manner: hurt, quietly bitter; emotion: despair)
{"index": 1, "takes": ["So you're leaving. After... after everything.",
                       "You're leaving. Huh. After everything.",
                       "You're really leaving? After all of it... everything."]}
Line: "It is fine. It is a modern cake." (manner: bluffing, laughing; emotion: comic_sarcastic)
{"index": 2, "takes": ["It's fine! (laugh) It's a... modern cake.",
                       "It's fine, it's fine. It's a modern cake, okay?",
                       "Oh, it's totally fine. (chuckle) Modern cake."]}"""


def build_rewrite_prompt(
    script: Script, scene_id: str, layer: dict[str, Any]
) -> list[dict[str, str]]:
    """Stage 2: three spoken takes per line, given the manner and emotion from stage 1."""
    names = {c: str(v["name"]) for c, v in script.header["dictionary"]["characters"].items()}
    rows = []
    for i, s, text in dialogue_lines(script, scene_id):
        d = layer["lines"].get(str(i), {})
        rows.append(
            f'{i}: {names.get(s.speaker, s.speaker)}: "{text}" '
            f"(manner: {d.get('manner') or 'natural'}; emotion: {d.get('emotion', 'neutral')})"
        )
    lines_text = "\n".join(rows)
    task = (
        f"{TALK_GUIDE}\n\n{REWRITE_EXAMPLE}\n\n"
        f"Scene situation:\n{load_context(script, scene_id).to_prompt()}\n"
        f"## Lines\n{lines_text}\n\n"
        "For EVERY line write three different takes of how this person really says it, in this "
        "manner and emotion. Every take must differ from the written line (contractions, a small "
        "opening word, a repeat, trailing off, a (laugh)/(sigh)/(gasp)/(chuckle)/(cry) where it "
        "fits) but keep the meaning, the facts and about the same length. No new names.\n"
        'Answer exactly as {"lines": [{"index": <n>, "takes": ["...", "...", "..."]}, ...]}.'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]


def _norm(text: str) -> str:
    return " ".join(text.lower().replace(chr(0x2019), "'").split())


def choose_spoken(original: str, takes: list[str], names: set[str]) -> tuple[str, bool, str | None]:
    """(spoken, meaning_ok, first rejected take): the first take that differs from the written
    line and passes the meaning check; the original if none does. Rejected = a changed take that
    failed the meaning check (kept for review)."""
    rejected = None
    for take in takes:
        take = str(take).strip()
        if not take or _norm(take) == _norm(original):
            continue
        if meaning_check(original, take, names).ok:
            return take, True, rejected
        rejected = rejected or take
    return original, False, rejected


def apply_rewrites(
    script: Script, scene_id: str, layer: dict[str, Any], answer: list[dict[str, Any]]
) -> list[str]:
    """Put the chosen takes into the layer (in place); returns problems."""
    chars = script.header["dictionary"]["characters"]
    entry = next(e for e in script.header["scenes"] if e["id"] == scene_id)
    present = {
        c
        for c, st in entry["continuity"].get("characters", {}).items()
        if st.get("location") == entry.get("location")
    } | {s.speaker for _, s, _ in dialogue_lines(script, scene_id)}
    names = {str(v["name"]) for c, v in chars.items() if c not in present}
    by_index = {int(x["index"]): x for x in answer if str(x.get("index", "")).isdigit()}
    problems = []
    for i, _s, text in dialogue_lines(script, scene_id):
        d = layer["lines"][str(i)]
        takes = [str(t) for t in by_index.get(i, {}).get("takes", []) if isinstance(t, str)]
        spoken, ok, rejected = choose_spoken(text, takes, names)
        d["spoken"], d["meaning_ok"], d["takes"] = spoken, ok, takes
        d["proposed"] = rejected
        d["nonverbal"] = [n for n in NONVERBALS if f"({n})" in spoken]
        if not ok:
            problems.append(f"{i}: no take passed ({len(takes)} takes)")
    return problems
