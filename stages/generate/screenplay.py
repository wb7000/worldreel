"""Render a `.wscript` as a conventional screenplay and estimate its running time.

The rendering expands every `{ID}` to its dictionary name, so it is also the "plain script"
baseline for token measurements (D-7.3).

Running time (D-7.2, assumptions set before measuring): the convention that one screenplay page
is about one minute of screen time, on a page laid out like the common US format (Courier 12 pt:
55 lines per page, action 60 characters wide, dialogue 35, parentheticals 25, character cues
on their own line, one blank line between elements). A second, independent floor: dialogue
spoken at 150 words per minute.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

from formats.wscript import Action, Dialogue, Script, expand_text

LINES_PER_PAGE = 55
ACTION_WIDTH = 60
DIALOGUE_WIDTH = 35
PAREN_WIDTH = 25
WORDS_PER_MINUTE = 150.0


def _wrap(text: str, width: int, indent: int) -> list[str]:
    return [" " * indent + line for line in textwrap.wrap(text, width)] or [""]


def render(script: Script) -> str:
    """Plain screenplay text: scene headings, action, character cues, dialogue. Sound, POV,
    tension, state changes and comments are production notes and are left out."""
    d = script.header["dictionary"]
    names = {cid: str(c["name"]).upper() for cid, c in d["characters"].items()}
    out: list[str] = []
    for entry in script.header["scenes"]:
        loc = d["locations"].get(entry.get("location", ""), {}).get("name", "")
        time = str(entry.get("time_of_day", "")).upper()
        out += [f"{loc.upper()} - {time}".strip(" -"), ""]
        for s in script.scenes[entry["id"]]:
            match s:
                case Action(text):
                    out += [*_wrap(expand_text(text, d), ACTION_WIDTH, 0), ""]
                case Dialogue(speaker, text, manner):
                    out.append(" " * 22 + names.get(speaker, speaker))
                    if manner:
                        out += _wrap(f"({expand_text(manner, d)})", PAREN_WIDTH, 16)
                    out += [*_wrap(expand_text(text, d), DIALOGUE_WIDTH, 10), ""]
                case _:
                    pass
    return "\n".join(out).rstrip() + "\n"


@dataclass(frozen=True)
class Runtime:
    pages: float
    page_minutes: float  # 1 page ~ 1 minute
    dialogue_words: int
    dialogue_minutes: float  # spoken at WORDS_PER_MINUTE: a floor, not an estimate


def runtime(script: Script) -> Runtime:
    lines = render(script).count("\n")
    d = script.header["dictionary"]
    words = sum(
        len(expand_text(s.text, d).split())
        for stmts in script.scenes.values()
        for s in stmts
        if isinstance(s, Dialogue)
    )
    pages = lines / LINES_PER_PAGE
    return Runtime(round(pages, 2), round(pages, 2), words, round(words / WORDS_PER_MINUTE, 2))
