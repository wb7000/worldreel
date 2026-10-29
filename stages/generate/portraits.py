"""Portrait prompts from the character bible (A-14). Appearance only — voice words never reach
the image prompt (D-8.1). Pure Python; images are made on Kaggle
(`training/kaggle/jobs/portraits.py`, Kandinsky 2.2) and chosen with SFace locally.
"""

from __future__ import annotations

import re
from typing import Any

from stages.generate.bible import Identity, expand

CANDIDATES = 6  # 4 left the closest pair at 0.386 > 0.363 (D-8.10)
NEGATIVE = "cartoon, drawing, painting, blurry, text, watermark, sunglasses, hands, two people"


def _pick(v: float, words: tuple[str, ...]) -> str:
    return words[min(int(v * len(words)), len(words) - 1)]


AGE_WORDS = re.compile(
    r"\b(teen|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|elderly|\d+)\b",
    re.IGNORECASE,
)


def appearance_words(ident: Identity, description: str = "") -> list[str]:
    a = ident.appearance
    age = 18 + 67 * a["age"]
    who = "woman" if a["presentation"] < 0.35 else "man" if a["presentation"] > 0.65 else "person"
    if AGE_WORDS.search(description):
        words = [who]  # the description states the age; a second age word would contradict it
    else:
        words = [f"{int(age // 10) * 10}s {who}" if age < 70 else f"elderly {who}"]
    words.append(
        _pick(
            a["skin_tone"],
            ("fair skin", "light skin", "olive skin", "brown skin", "dark brown skin"),
        )
    )
    if a["hair_length"] < 0.05:
        words.append("bald")
    else:
        # grey / white only from 50 up; younger people map the same position onto natural colours
        tones = (
            ("black", "dark brown", "brown", "light brown", "blond", "grey", "white")
            if age >= 50
            else ("black", "dark brown", "brown", "auburn", "light brown", "blond", "red")
        )
        tone = _pick(a["hair_tone"], tones)
        length = _pick(
            a["hair_length"], ("very short", "short", "medium length", "long", "very long")
        )
        words.append(f"{length} {tone} hair")
    if who != "woman" and a["facial_hair"] > 0.7:
        words.append("full beard")
    elif who != "woman" and a["facial_hair"] > 0.4:
        words.append("moustache")
    words.append(_pick(a["build"], ("slim", "lean", "average build", "sturdy", "heavy-set")))
    words.append(_pick(a["face_width"], ("narrow face", "oval face", "broad face")))
    # the remaining face traits of the bible, when clearly high or low (D-8.10: without them the
    # image model drew characters of the same sex and hair too alike)
    for dim, low, high in (
        ("jaw", "soft rounded jaw", "strong square jaw"),
        ("brow", "thin light eyebrows", "heavy dark eyebrows"),
        ("eye_size", "narrow eyes", "large eyes"),
        ("nose_length", "short nose", "long nose"),
        ("nose_width", "narrow nose", "broad nose"),
        ("mouth_width", "small mouth", "wide mouth"),
        ("lip_fullness", "thin lips", "full lips"),
    ):
        if a[dim] < 0.25:
            words.append(low)
        elif a[dim] > 0.75:
            words.append(high)
    return words


def portrait_prompt(character: dict[str, Any]) -> str:
    ident = expand(
        int(character["identity"]["seed"]), str(character["identity"].get("requested", ""))
    )
    desc = str(character.get("description", ""))
    return (
        "studio portrait photo of a "
        + ", ".join(appearance_words(ident, str(character.get("description", ""))))
        + (f", {desc}" if desc else "")
        + ", head and shoulders, facing the camera, looking into the lens, neutral grey background,"
        " soft even light, sharp focus"
    )


def candidate_seeds(character: dict[str, Any], n: int = CANDIDATES) -> list[int]:
    seed = int(character["identity"]["seed"])
    return [seed * 10 + k for k in range(n)]
