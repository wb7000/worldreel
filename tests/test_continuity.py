from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from formats.common import FormatError
from formats.wscript import (
    Change,
    Script,
    parse_scene,
    read_wscript,
    serialize_scene,
    validate_wscript,
    write_wscript,
)
from stages.generate.continuity import check, diff, propagate, replay

S1 = """@scene S001 loc=L01
C01: The key is under the pot.
= C01 holding+=P01
! {C01} pockets the {P01} and leaves for the kitchen.
= C01 location=L02
"""
S2 = """@scene S002 loc=L02
C01: Here it is.
= C01 holding-=P01
= C02 holding+=P01
C02: Thank you.
= world lighting=dawn
"""


def _script(s1: str = S1, s2: str = S2, cont2: dict[str, Any] | None = None) -> Script:
    start = {
        "characters": {
            "C01": {"location": "L01", "wearing": ["P02"]},
            "C02": {"location": "L02"},
        },
        "lighting": "night",
    }
    end1 = {
        "characters": {
            "C01": {"location": "L02", "wearing": ["P02"], "holding": ["P01"]},
            "C02": {"location": "L02"},
        },
        "lighting": "night",
    }
    header = {
        "format": "wscript",
        "format_version": "1.1",
        "title": "t",
        "language": "en",
        "dictionary": {
            "characters": {"C01": {"name": "Mira"}, "C02": {"name": "Ade"}},
            "locations": {"L01": {"name": "yard"}, "L02": {"name": "kitchen"}},
            "props": {"P01": {"name": "key"}, "P02": {"name": "coat"}},
            "text": {},
        },
        "scenes": [
            {"id": "S001", "title": "a", "summary": "a", "location": "L01", "tension": 2,
             "continuity": start},
            {"id": "S002", "title": "b", "summary": "b", "location": "L02", "tension": 3,
             "continuity": cont2 if cont2 is not None else end1},
        ],
    }  # fmt: skip
    return Script(header=header, scenes={"S001": parse_scene(s1), "S002": parse_scene(s2)})


def test_change_lines_round_trip_and_validate(tmp_path: Path) -> None:
    assert serialize_scene(parse_scene(S1)) == S1
    assert Change("C01", (("holding", "+=", "P01"),)) in parse_scene(S1)
    path = tmp_path / "t.wscript"
    write_wscript(path, _script())
    assert validate_wscript(path).ok and read_wscript(path).header["format_version"] == "1.1"


@pytest.mark.parametrize(
    ("line", "code"),
    [
        ("= C09 location=L01", "bad_reference"),  # not a character
        ("= C01 holding+=P99", "bad_reference"),  # unknown prop
        ("= C01 location+=L01", "bad_change"),  # location is set, not added
        ("= C01 mood=angry", "bad_change"),  # unknown key
        ("= world rain=yes", "bad_change"),
    ],
)
def test_invalid_changes_are_rejected(tmp_path: Path, line: str, code: str) -> None:
    with pytest.raises(FormatError, match=code):
        write_wscript(tmp_path / "t.wscript", _script(s1=S1 + line + "\n"))


def test_clean_script_has_no_issues() -> None:
    report, per_scene = check(_script())
    assert report.ok and not report.warnings, report.issues
    assert per_scene == {"S001": [], "S002": []}


def test_replay_gives_the_end_state() -> None:
    s = _script()
    end, issues = replay(s.header["scenes"][1]["continuity"], s.scenes["S002"], "L02")
    assert not issues
    assert end["characters"]["C02"]["holding"] == ["P01"] and end["lighting"] == "dawn"
    assert end["characters"]["C01"]["holding"] == []


def test_each_violation_is_detected() -> None:
    # Speaker not in the room: C01 left for the kitchen, then speaks in the yard.
    bad = S1 + "C01: Wait!\n"
    codes = {i.code for i in check(_script(s1=bad))[1]["S001"]}
    assert "speaker_absent" in codes
    # One key in two hands: C02 takes it without C01 giving it.
    bad2 = S2.replace("= C01 holding-=P01\n", "")
    assert "prop_in_two_places" in {i.code for i in check(_script(s2=bad2))[1]["S002"]}
    # Giving away what you do not have.
    bad3 = S2.replace("= C01 holding-=P01", "= C01 holding-=P02")
    assert "prop_not_held" in {i.code for i in check(_script(s2=bad3))[1]["S002"]}


def test_scene_break_must_be_written_or_declared_offscreen() -> None:
    s = _script()
    cont = copy.deepcopy(s.header["scenes"][1]["continuity"])
    cont["characters"]["C01"]["wearing"] = []  # coat vanished between scenes
    issues = check(_script(cont2=cont))[1]["S002"]
    assert [i.code for i in issues] == ["continuity_break"] and "C01.wearing" in issues[0].message
    cont["offscreen"] = ["C01.wearing"]  # ...unless it is a declared time jump
    assert check(_script(cont2=cont))[0].ok
    cont["offscreen"] = ["C01.wearing", "world.weather"]
    assert {i.code for i in check(_script(cont2=cont))[0].warnings} == {"offscreen_unused"}


def test_propagate_fills_start_states_and_keeps_offscreen_values() -> None:
    cont2 = {"characters": {"C01": {"wearing": []}}, "offscreen": ["C01.wearing"]}
    filled = propagate(_script(cont2=cont2))
    start2 = filled.header["scenes"][1]["continuity"]
    assert start2["characters"]["C01"]["location"] == "L02"  # carried over
    assert start2["characters"]["C01"]["holding"] == ["P01"]
    assert "wearing" not in start2["characters"]["C01"]  # the declared offscreen change
    assert check(filled)[0].ok
    assert diff(start2, start2) == {}
