from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from formats.common import FormatError
from formats.wscript import (
    Action,
    Dialogue,
    Script,
    WslSyntaxError,
    expand_text,
    load_context,
    parse_scene,
    read_wscript,
    serialize_scene,
    text_refs,
    validate_wscript,
    write_wscript,
)

SCENE_1 = """@scene S001 loc=L01 time=night tension=3
# Opening. Keep it quiet.
~ rain intensity=heavy
> wide
! {C01} walks into {L01} holding the {P01}.

C01: Hello? Is anyone here?
C02 (whisper): Over here. Don't use the {{light}}.
% 6
> first C02
! {T01}
"""

SCENE_2 = """@scene S002 loc=L02 time=dawn
! {C01} and {C02} sit by the window.
C01: We made it.
"""

SCENE_3 = """@scene S003 loc=L01
! The courtyard is empty.
"""


def make_header() -> dict[str, Any]:
    return {
        "format": "wscript",
        "format_version": "1.0",
        "title": "The Courtyard",
        "language": "en",
        "dictionary": {
            "characters": {
                "C01": {"name": "Mira", "description": "tall, red coat", "identity_seed": 42},
                "C02": {"name": "Ade", "description": "older, grey beard", "voice": "low"},
                "C03": {"name": "Unused", "description": "never appears in S001"},
            },
            "locations": {"L01": {"name": "the courtyard"}, "L02": {"name": "the kitchen"}},
            "props": {"P01": {"name": "umbrella"}, "P02": {"name": "red coat"}},
            "text": {
                "T01": "Lightning shows the whole yard for one second: wet stone, a locked gate."
            },
        },
        "scenes": [
            {
                "id": "S001",
                "path": "",
                "sha256": "0" * 64,
                "title": "Arrival",
                "summary": "Mira arrives in the rain and finds Ade hiding.",
                "location": "L01",
                "time_of_day": "night",
                "tension": 3,
                "continuity": {
                    "characters": {
                        "C01": {
                            "location": "L01",
                            "wearing": ["P02"],
                            "holding": ["P01"],
                            "state": "wet",
                        },
                        "C02": {"location": "L01"},
                    },
                    "weather": "rain",
                },
            },
            {
                "id": "S002",
                "path": "",
                "sha256": "0" * 64,
                "title": "Morning",
                "summary": "They rest in the kitchen at dawn.",
                "location": "L02",
                "tension": 1,
                "continuity": {
                    "characters": {"C01": {"location": "L02"}, "C02": {"location": "L02"}}
                },
            },
            {
                "id": "S003",
                "path": "",
                "sha256": "0" * 64,
                "title": "Empty yard",
                "summary": "The courtyard, deserted.",
                "tension": 0,
                "continuity": {},
            },
        ],
    }


def make_script() -> Script:
    return Script(
        header=make_header(),
        scenes={
            "S001": parse_scene(SCENE_1),
            "S002": parse_scene(SCENE_2),
            "S003": parse_scene(SCENE_3),
        },
    )


# --------------------------------------------------------------------------- scene language


@pytest.mark.parametrize("text", [SCENE_1, SCENE_2, SCENE_3])
def test_canonical_scene_round_trips_byte_for_byte(text: str) -> None:
    assert serialize_scene(parse_scene(text)) == text


def test_non_canonical_scene_normalises_then_is_stable() -> None:
    messy = "@scene  S009   loc=L01  \n!   {C01} waits.   \nC01   (quietly)  :   Fine.\n%   4\n"
    once = serialize_scene(parse_scene(messy))
    assert once == "@scene S009 loc=L01\n! {C01} waits.\nC01 (quietly): Fine.\n% 4\n"
    assert serialize_scene(parse_scene(once)) == once


def test_dialogue_text_with_colons_and_parens_is_lossless() -> None:
    text = "@scene S001\nC01: Note: it's 5:30 :) (really).\n"
    statements = parse_scene(text)
    assert statements[1] == Dialogue("C01", "Note: it's 5:30 :) (really).")
    assert serialize_scene(statements) == text


@pytest.mark.parametrize(
    "bad",
    [
        "! no header first\n",
        "@scene S001\n@scene S002\n",
        "@scene S001\n% 11\n",
        "@scene S001\n> sideways\n",
        "@scene S001\nnot a statement\n",
        "@scene S001 loc\n",
    ],
)
def test_syntax_errors(bad: str) -> None:
    with pytest.raises(WslSyntaxError):
        parse_scene(bad)


def test_text_refs_and_escapes() -> None:
    assert text_refs("{C01} meets {C02} with {{braces}}") == ["C01", "C02"]
    with pytest.raises(FormatError):
        text_refs("unclosed {C01")
    with pytest.raises(FormatError):
        text_refs("stray } brace")
    dictionary = make_header()["dictionary"]
    assert expand_text("{C01} sees {{x}} in {L01}", dictionary) == "Mira sees {x} in the courtyard"


# --------------------------------------------------------------------------- container


@pytest.mark.parametrize("as_zip", [True, False])
def test_script_round_trip(tmp_path: Path, as_zip: bool) -> None:
    script = make_script()
    path = tmp_path / "story.wscript"
    write_wscript(path, script, as_zip=as_zip)
    loaded = read_wscript(path)
    assert loaded.report.ok, loaded.report.issues
    assert loaded.scenes == script.scenes
    assert loaded.header["dictionary"] == script.header["dictionary"]
    assert [s["path"] for s in loaded.header["scenes"]] == [
        "scenes/S001.wsl",
        "scenes/S002.wsl",
        "scenes/S003.wsl",
    ]
    again = tmp_path / ("again.wscript" if as_zip else "again_dir.wscript")
    write_wscript(again, loaded, as_zip=as_zip)
    if as_zip:
        assert again.read_bytes() == path.read_bytes()


def test_unknown_speaker_rejected(tmp_path: Path) -> None:
    script = make_script()
    script.scenes["S002"].append(Dialogue("C99", "Who am I?"))
    with pytest.raises(FormatError, match="unknown speaker"):
        write_wscript(tmp_path / "x.wscript", script)


def test_unknown_text_reference_rejected(tmp_path: Path) -> None:
    script = make_script()
    script.scenes["S002"].append(Action("{P77} falls."))
    with pytest.raises(FormatError, match="unknown reference"):
        write_wscript(tmp_path / "x.wscript", script)


def test_bad_continuity_reference_rejected(tmp_path: Path) -> None:
    script = make_script()
    script.header["scenes"][0]["continuity"]["characters"]["C01"]["holding"] = ["P99"]
    with pytest.raises(FormatError, match="unknown prop"):
        write_wscript(tmp_path / "x.wscript", script)


def test_tampered_scene_detected(tmp_path: Path) -> None:
    path = tmp_path / "story.wscript"
    write_wscript(path, make_script(), as_zip=False)
    (path / "scenes" / "S002.wsl").write_text(SCENE_2 + "! Extra line.\n", encoding="utf-8")
    assert "hash_mismatch" in validate_wscript(path).codes()


def test_scene_id_must_match_index(tmp_path: Path) -> None:
    script = make_script()
    script.scenes["S003"] = parse_scene(SCENE_3.replace("S003", "S004"))
    with pytest.raises(FormatError, match="@scene S004"):
        write_wscript(tmp_path / "x.wscript", script)


# --------------------------------------------------------------------------- model context


def test_load_context_contains_only_current_scene_and_used_entries() -> None:
    script = make_script()
    ctx = load_context(script, "S001")
    assert ctx.scene_text == SCENE_1
    assert set(ctx.dictionary["characters"]) == {"C01", "C02"}
    assert set(ctx.dictionary["props"]) == {"P01", "P02"}
    assert set(ctx.dictionary["text"]) == {"T01"}
    assert ctx.other_summaries == [
        ("S002", "They rest in the kitchen at dawn."),
        ("S003", "The courtyard, deserted."),
    ]
    prompt = ctx.to_prompt()
    assert "We made it." not in prompt  # other scenes' full text is not loaded
    assert "Unused" not in prompt
    assert "Mira" in prompt


def test_context_size_grows_with_current_scene_not_whole_script() -> None:
    script = make_script()
    base = len(load_context(script, "S003").to_prompt())
    bigger = copy.deepcopy(script)
    bigger.scenes["S002"] = bigger.scenes["S002"] + [Action("{C01} talks.")] * 500
    assert len(load_context(bigger, "S003").to_prompt()) == base
