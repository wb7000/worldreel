from __future__ import annotations

import pytest

from formats.wscript import continuity_lines, read_wscript
from tools.make_sample_wscript import OUT

pytest.importorskip("tiktoken")


def test_continuity_lines_keep_every_fact() -> None:
    cont = read_wscript(OUT).header["scenes"][7]["continuity"]  # S008, after a time jump
    text = "\n".join(continuity_lines(cont))
    for cid, st in cont["characters"].items():
        assert cid in text and st["location"] in text
        assert all(p in text for p in st.get("holding", []) + st.get("wearing", []))
    assert "grey_dawn" in text and "world.time" in text  # world state and the declared jump


def test_scene_context_stays_small_as_the_script_grows() -> None:
    from eval.tokens import measure, repeat, tokenizers

    count = tokenizers()["o200k_base"]
    script = read_wscript(OUT)
    short, long = measure(script, count), measure(repeat(script, 6), count)
    assert short["context_max"] < short["whole_plain"] / 2
    assert long["whole_plain"] > 5.5 * short["whole_plain"]  # the whole script grows ~6x
    assert long["context_mean"] < 3 * short["context_mean"]  # one scene's context does not
