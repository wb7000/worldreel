from __future__ import annotations

from formats.wscript import read_wscript
from stages.generate.portraits import candidate_seeds, portrait_prompt
from tools.make_sample_wscript import OUT


def test_prompts_use_appearance_only_and_stay_consistent() -> None:
    chars = read_wscript(OUT).header["dictionary"]["characters"]
    for c in chars.values():
        p = portrait_prompt(c)
        assert not any(w in p for w in ("low voice", "quick", "deep voice", "breathy"))  # D-8.1
        assert len(candidate_seeds(c)) == 4
    young = portrait_prompt(chars["C03"])  # a man in his twenties
    assert "white hair" not in young and "grey hair" not in young
    assert "20s" not in young  # the description already gives the age
