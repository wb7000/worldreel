from __future__ import annotations

import json
from pathlib import Path

import pytest

from formats.common import FormatError
from formats.wscript import read_wscript, write_wscript
from stages.generate.director import (
    build_prompt,
    dialogue_lines,
    meaning_check,
    parse_answer,
    to_layer,
)
from tools.make_sample_wscript import OUT


def test_meaning_check_accepts_natural_speech_and_rejects_changes() -> None:
    names = {"Mira", "Ade", "Tomas", "Lena"}
    orig = "They said a team. I see one wet woman with a paper map."
    assert meaning_check(
        orig, "They said a team. And I see... one wet woman with a paper map.", names
    ).ok
    assert not meaning_check(orig, "Nobody came at all.", names).ok  # meaning lost
    assert not meaning_check(orig, orig + " Lena told me so.", names).ok  # absent character
    assert meaning_check("You may ask.", "Hm. You can ask.", names).ok  # modal swap
    long = orig + " " + "really " * 20
    assert not meaning_check(orig, long, names).ok  # far too long


def test_prompt_lists_every_dialogue_line_with_its_index() -> None:
    script = read_wscript(OUT)
    msgs = build_prompt(script, "S001")
    lines = dialogue_lines(script, "S001")
    assert all(f"{i}: " in msgs[1]["content"] for i, _, _ in lines)
    assert "comic_sarcastic" in msgs[1]["content"] and msgs[0]["role"] == "system"


def test_answer_becomes_a_valid_delivery_layer_with_fallbacks(tmp_path: Path) -> None:
    script = read_wscript(OUT)
    lines = dialogue_lines(script, "S001")
    (i0, _, _), (i1, _, t1) = lines[0], lines[1]
    said = "Mr. Ade? Hi, I'm Mira, from the archive. They said... they said you'd be expecting me."
    answer = "```json\n" + json.dumps({"lines": [
        {"index": i0, "manner": "polite, a little out of breath", "emotion": "neutral",
         "intensity": 0.3, "spoken": said,
         "nonverbal": []},
        {"index": i1, "manner": "dry", "emotion": "smug", "intensity": 7,
         "spoken": "Completely different words.",
         "nonverbal": ["laugh"]},
    ]}) + "\n```"  # fmt: skip
    layer, problems = to_layer(script, "S001", parse_answer(answer), "director:test")
    first, second = layer["lines"][str(i0)], layer["lines"][str(i1)]
    assert first["meaning_ok"] and first["spoken"].startswith("Mr. Ade? Hi")
    assert not second["meaning_ok"] and second["spoken"] == t1 and second["emotion"] == "neutral"
    assert second["intensity"] == 1.0 and second["nonverbal"] == []
    assert any("missing" in p for p in problems)  # lines the answer skipped fall back too
    script.delivery["S001"] = layer
    path = tmp_path / "d.wscript"
    write_wscript(path, script, as_zip=False)
    back = read_wscript(path)
    assert back.report.ok and back.header["format_version"] in ("1.1", "1.2")
    assert back.delivery["S001"]["lines"][str(i0)]["manner"] == "polite, a little out of breath"


def test_invalid_layers_are_rejected(tmp_path: Path) -> None:
    script = read_wscript(OUT)
    script.delivery["S001"] = {
        "lines": {"0": {"emotion": "neutral", "intensity": 0.5, "spoken": "x"}}
    }
    with pytest.raises(FormatError, match="not dialogue"):
        write_wscript(tmp_path / "x.wscript", script, as_zip=False)
    with pytest.raises(ValueError):
        parse_answer("I think the scene is sad.")
