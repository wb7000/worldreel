"""Character bible -> MPFB2 human preset (A-14 v2, D-8.11).

MPFB2 (GPL code, CC0 assets) builds a real human mesh in Blender from macro sliders
(gender / age / height / weight / muscle / proportions / ethnic blend) and named shape targets.
This module only writes the preset dict that `HumanService.deserialize_from_dict` reads
(keys from MPFB 2.0.18 `services/humanservice.py`); Blender runs it in
`tools/blender/human.py`. Pure Python, so the mapping is testable without Blender.
"""

from __future__ import annotations

import itertools
from typing import Any

from stages.generate.avatars import HAIR_OLD, HAIR_YOUNG
from stages.generate.bible import Identity

# MPFB's age slider: 0 baby, 0.1875 child, 0.5 young adult, 1.0 old (humanproperties/age.json).
# The year anchors 1 / 11 / 25 / 90 are MakeHuman's (assumption, recorded in D-8.11).
AGE_ANCHORS = ((1.0, 0.0), (11.0, 0.1875), (25.0, 0.5), (90.0, 1.0))

# bible face trait -> MPFB target pairs (decr, incr); value 0.5 = neutral. No eye size: the
# portrait's eyes are projected over the mesh's, and opened-up mesh eyes stared (D-8.11)
FACE_TARGETS: dict[str, tuple[tuple[str, str], ...]] = {
    "face_width": (("head-scale-horiz-decr", "head-scale-horiz-incr"),),
    "jaw": (("chin-width-decr", "chin-width-incr"), ("chin-bones-decr", "chin-bones-incr")),
    "eye_spacing": (("l-eye-trans-in", "l-eye-trans-out"), ("r-eye-trans-in", "r-eye-trans-out")),
    "nose_length": (("nose-scale-vert-decr", "nose-scale-vert-incr"),),
    "nose_width": (("nose-scale-horiz-decr", "nose-scale-horiz-incr"),),
    "mouth_width": (("mouth-scale-horiz-decr", "mouth-scale-horiz-incr"),),
    "lip_fullness": (
        ("mouth-upperlip-volume-decr", "mouth-upperlip-volume-incr"),
        ("mouth-lowerlip-volume-decr", "mouth-lowerlip-volume-incr"),
    ),
}
# a bible extreme (0 or 1) sets the target to this weight; kept mild because the projected
# portrait carries the face's look and a strong push fights it (D-8.11: narrow mesh nose under
# the portrait's broad one)
STRENGTH = 0.4


def age_years(ident: Identity) -> float:
    return 18 + 67 * ident.appearance["age"]  # same rule as avatars / portraits


def mpfb_age(years: float) -> float:
    for (y0, v0), (y1, v1) in itertools.pairwise(AGE_ANCHORS):
        if years <= y1:
            return v0 + (max(years, y0) - y0) / (y1 - y0) * (v1 - v0)
    return 1.0


def phenotype(ident: Identity) -> dict[str, Any]:
    a = ident.appearance
    skin = a["skin_tone"]
    return {
        "gender": round(a["presentation"], 4),
        "age": round(mpfb_age(age_years(ident)), 4),
        "height": round(a["height"], 4),
        "weight": round(0.25 + 0.5 * a["build"], 4),
        "muscle": 0.5,
        "proportions": 0.5,
        "cupsize": 0.5,
        "firmness": 0.5,
        # MPFB blends three ethnic body/face shapes; the bible has skin tone only
        "race": {"african": round(skin, 4), "asian": 0.0, "caucasian": round(1 - skin, 4)},
    }


# MPFB's default eyes are wide open; portraits (calm studio faces) are not, and the projected
# lids then sat around staring mesh eyes (D-8.11). Applied to every character.
LIDS = (("l-eye-height2-decr", 0.6), ("r-eye-height2-decr", 0.6),
        ("l-eye-eyefold-down", 0.4), ("r-eye-eyefold-down", 0.4))  # fmt: skip


def face_targets(ident: Identity) -> list[dict[str, Any]]:
    out = [{"target": t, "value": v} for t, v in LIDS]
    for trait, pairs in FACE_TARGETS.items():
        v = 2 * (ident.appearance[trait] - 0.5) * STRENGTH
        if abs(v) < 0.05:
            continue
        for decr, incr in pairs:
            out.append({"target": incr if v > 0 else decr, "value": round(abs(v), 4)})
    return out


# MakeHuman system assets (CC0, makehuman_system_assets pack) by file name; MPFB finds them
HAIR_SHORT = ("short02", "short03", "short04")  # short01 renders as a flat cap
HAIR_MEDIUM = ("bob01", "bob02")
HAIR_LONG = ("long01", "ponytail01")  # braid01 has a fringe over one eye
EYEBROWS = tuple(f"eyebrow{n:03d}" for n in range(1, 13))
# casual suits 02 / 04 (male) and 01 / 02 (female) carry the MakeHuman logo: left out
CLOTHES_MALE = ("male_casualsuit01", "male_casualsuit03", "male_casualsuit05",
                "male_casualsuit06", "male_worksuit01")  # fmt: skip
CLOTHES_FEMALE = ("female_elegantsuit01", "female_sportsuit01")
SHOES = tuple(f"shoes0{n}" for n in range(1, 7))


def _pick(v: float, options: tuple[str, ...]) -> str:
    return options[min(int(v * len(options)), len(options) - 1)]


def skin_name(ident: Identity) -> str:
    """Skin texture by age band, tone band and sex. The tone bands pick the texture whose
    colour is closest to the bible's skin tone (light / medium / deep), not an ethnicity."""
    a = ident.appearance
    years = age_years(ident)
    band = "young" if years < 35 else "middleage" if years < 55 else "old"
    tone = "caucasian" if a["skin_tone"] < 0.35 else "asian" if a["skin_tone"] < 0.6 else "african"
    sex = "female" if a["presentation"] < 0.5 else "male"
    return f"{band}_{tone}_{sex}"


def human_preset(ident: Identity) -> dict[str, Any]:
    """The full MPFB preset for one character (seeded choices use the identity seed)."""
    a = ident.appearance
    female = a["presentation"] < 0.5
    hair = ""
    if a["hair_length"] >= 0.05:
        group = (
            HAIR_SHORT
            if a["hair_length"] < 0.4
            else HAIR_MEDIUM
            if a["hair_length"] < 0.6
            else HAIR_LONG
        )
        hair = group[ident.seed % len(group)] + ".mhclo"
    if female:  # only two logo-free outfits: the sport one for the young
        clothes: tuple[str, ...] = (
            CLOTHES_FEMALE[1] if age_years(ident) < 30 else CLOTHES_FEMALE[0],
        )
    else:
        clothes = CLOTHES_MALE
    skin = skin_name(ident)
    return {
        "phenotype": phenotype(ident),
        "targets": face_targets(ident),
        "skin_mhmat": f"{skin}.mhmat",
        "skin_material_type": "ENHANCED_SSS",
        "eyes": "high-poly.mhclo",
        "eyes_material_type": "PROCEDURAL_EYES",
        "eyes_material_settings": {},
        "eyebrows": _pick(a["brow"], EYEBROWS) + ".mhclo",
        "eyelashes": ("eyelashes02" if female else "eyelashes01") + ".mhclo",
        "teeth": "teeth_base.mhclo",
        "tongue": "tongue01.mhclo",
        "hair": hair,
        "proxy": "",
        "rig": "",
        "clothes": [
            clothes[ident.seed % len(clothes)] + ".mhclo",
            SHOES[(ident.seed // 7) % len(SHOES)] + ".mhclo",
        ],
        "skin_material_settings": {},
        "alternative_materials": {},
        # read by tools/blender/human.py, not by MPFB
        "worldreel": {"height_m": round(1.50 + 0.45 * a["height"], 3), "hair_rgb": hair_rgb(ident)},
    }


def hair_rgb(ident: Identity) -> list[float]:
    """sRGB 0-1, the same palette the portraits' hair words come from (grey / white from 50)."""
    palette = HAIR_OLD if age_years(ident) >= 50 else HAIR_YOUNG
    v = ident.appearance["hair_tone"]
    rgb = palette[min(int(v * len(palette)), len(palette) - 1)]
    return [round(c / 255, 4) for c in rgb]
