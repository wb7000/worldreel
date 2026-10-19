"""Style-request filter for character identities (brief §5, §11 "identity/likeness request
filter hook").

Requests like "older, bearded, deep voice" set ranges on identity dimensions. The parser is an
**allowlist**: every word must be a known style word (or a filler word), so a named real person
("looks like <name>") cannot be requested — an unknown word rejects the whole request. Words that
ask for a likeness ("like", "resembling", "lookalike", ...) are rejected with that reason even when
the rest is valid. Adults only: the age scale starts at 18.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

Range = tuple[float, float]

# phrase -> {dimension: (low, high)} on the [0, 1] scale of stages.generate.bible
STYLE_WORDS: dict[str, dict[str, Range]] = {
    # age: 18 + 67 * value years
    "young": {"age": (0.0, 0.18)},
    "adult": {"age": (0.18, 0.45)},
    "middle-aged": {"age": (0.33, 0.55)},
    "older": {"age": (0.55, 0.8)},
    "elderly": {"age": (0.75, 1.0)},
    "old": {"age": (0.75, 1.0)},
    # presentation (0 feminine .. 1 masculine), with a matching voice range
    "woman": {"presentation": (0.0, 0.35), "pitch": (0.45, 1.0)},
    "female": {"presentation": (0.0, 0.35), "pitch": (0.45, 1.0)},
    "man": {"presentation": (0.65, 1.0), "pitch": (0.0, 0.55)},
    "male": {"presentation": (0.65, 1.0), "pitch": (0.0, 0.55)},
    "androgynous": {"presentation": (0.35, 0.65)},
    # body
    "tall": {"height": (0.7, 1.0)},
    "short": {"height": (0.0, 0.3)},
    "slim": {"build": (0.0, 0.3)},
    "thin": {"build": (0.0, 0.25)},
    "wiry": {"build": (0.0, 0.3)},
    "athletic": {"build": (0.4, 0.7)},
    "stocky": {"build": (0.7, 1.0)},
    "heavy": {"build": (0.75, 1.0)},
    "broad": {"build": (0.65, 1.0)},
    # hair (tone: 0 black .. 1 white)
    "bald": {"hair_length": (0.0, 0.05)},
    "long hair": {"hair_length": (0.7, 1.0)},
    "short hair": {"hair_length": (0.1, 0.35)},
    "black hair": {"hair_tone": (0.0, 0.15)},
    "dark hair": {"hair_tone": (0.0, 0.3)},
    "brown hair": {"hair_tone": (0.3, 0.55)},
    "blond": {"hair_tone": (0.6, 0.8)},
    "fair hair": {"hair_tone": (0.6, 0.8)},
    "grey hair": {"hair_tone": (0.85, 1.0)},
    "gray hair": {"hair_tone": (0.85, 1.0)},
    "white hair": {"hair_tone": (0.92, 1.0)},
    # skin (0 light .. 1 deep)
    "light skin": {"skin_tone": (0.0, 0.3)},
    "pale skin": {"skin_tone": (0.0, 0.2)},
    "medium skin": {"skin_tone": (0.35, 0.65)},
    "brown skin": {"skin_tone": (0.5, 0.8)},
    "dark skin": {"skin_tone": (0.7, 1.0)},
    "deep skin": {"skin_tone": (0.75, 1.0)},
    # facial hair
    "bearded": {"facial_hair": (0.7, 1.0)},
    "beard": {"facial_hair": (0.7, 1.0)},
    "moustache": {"facial_hair": (0.4, 0.6)},
    "clean-shaven": {"facial_hair": (0.0, 0.1)},
    # face
    "round face": {"face_width": (0.7, 1.0), "jaw": (0.0, 0.4)},
    "narrow face": {"face_width": (0.0, 0.3)},
    "square jaw": {"jaw": (0.7, 1.0)},
    "big eyes": {"eye_size": (0.7, 1.0)},
    "small eyes": {"eye_size": (0.0, 0.3)},
    "long nose": {"nose_length": (0.7, 1.0)},
    "wide mouth": {"mouth_width": (0.7, 1.0)},
    "full lips": {"lip_fullness": (0.7, 1.0)},
    "thin lips": {"lip_fullness": (0.0, 0.3)},
    "heavy brows": {"brow": (0.7, 1.0)},
    # voice
    "deep voice": {"pitch": (0.0, 0.25)},
    "low voice": {"pitch": (0.0, 0.3)},
    "high voice": {"pitch": (0.75, 1.0)},
    "fast": {"rate": (0.7, 1.0)},
    "quick": {"rate": (0.65, 1.0)},
    "slow": {"rate": (0.0, 0.3)},
    "calm": {"rate": (0.15, 0.45), "roughness": (0.0, 0.4)},
    "raspy": {"roughness": (0.7, 1.0)},
    "gravelly": {"roughness": (0.75, 1.0)},
    "breathy": {"breathiness": (0.7, 1.0)},
    "clear": {"breathiness": (0.0, 0.3), "roughness": (0.0, 0.3)},
}

FILLER = {"a", "an", "and", "with", "the", "very", "slightly", "bit", "of", "voice", "voiced"}
LIKENESS = {
    "like",
    "likeness",
    "resembling",
    "resembles",
    "resemble",
    "lookalike",
    "look-alike",
    "looks",
    "look",
    "as",
    "similar",
    "same",
    "copy",
    "clone",
    "impersonate",
    "celebrity",
    "famous",
}


@dataclass
class StyleRequest:
    text: str
    allowed: bool
    reason: str
    ranges: dict[str, Range] = field(default_factory=dict)
    words: list[str] = field(default_factory=list)


def parse_style(text: str) -> StyleRequest:
    tokens = [t for t in re.split(r"[^a-z\-]+", text.lower()) if t]
    words: list[str] = []
    ranges: dict[str, Range] = {}
    unknown: list[str] = []
    likeness: list[str] = []
    i = 0
    while i < len(tokens):
        two = " ".join(tokens[i : i + 2])
        if len(tokens) > i + 1 and two in STYLE_WORDS:
            phrase, i = two, i + 2
        else:
            phrase, i = tokens[i], i + 1
            if phrase in FILLER:
                continue
            if phrase in LIKENESS:
                likeness.append(phrase)
                continue
            if phrase not in STYLE_WORDS:
                unknown.append(phrase)
                continue
        words.append(phrase)
        for dim, (lo, hi) in STYLE_WORDS[phrase].items():
            old_lo, old_hi = ranges.get(dim, (0.0, 1.0))
            ranges[dim] = (max(lo, old_lo), min(hi, old_hi))
    if likeness:
        return StyleRequest(
            text, False, f"likeness requests are not allowed ({', '.join(likeness)})"
        )
    if unknown:
        return StyleRequest(
            text,
            False,
            f"unknown words {unknown}: only style words are accepted, never names",
        )
    empty = [d for d, (lo, hi) in ranges.items() if lo > hi]
    if empty:
        return StyleRequest(text, False, f"contradictory request for {empty}")
    return StyleRequest(text, True, "allowed", ranges, words)
