from __future__ import annotations

import copy

import pytest

from formats.wscript import load_context, read_wscript
from stages.generate.bible import (
    APPEARANCE_MIN,
    check_distinct,
    check_script,
    distance,
    expand,
    sample_cast,
    unique_identities,
    verify,
)
from stages.safety.likeness import parse_style
from tools.make_sample_wscript import OUT


@pytest.mark.parametrize(
    "text",
    [
        "looks like Brad Pitt",
        "Shah Rukh Khan",
        "tall woman resembling a famous actress",
        "same face as my neighbour",
        "older man, Morgan Freeman voice",
    ],
)
def test_names_and_likeness_requests_are_refused(text: str) -> None:
    assert not parse_style(text).allowed


def test_style_words_set_ranges_and_contradictions_are_refused() -> None:
    req = parse_style("an older, bearded man with a deep voice")
    assert req.allowed and req.words == ["older", "bearded", "man", "deep voice"]
    assert req.ranges["age"] == (0.55, 0.8) and req.ranges["pitch"] == (0.0, 0.25)
    assert not parse_style("tall and short").allowed
    assert expand(3, "woman").appearance["facial_hair"] <= 0.1  # default: clean-shaven
    assert expand(3, "woman, bearded").appearance["facial_hair"] >= 0.7  # unless asked


def test_identity_is_a_pure_function_of_seed_and_request() -> None:
    a, b = expand(11, "elderly man, grey hair"), expand(11, "elderly man, grey hair")
    assert a == b and distance(a, b) == (0.0, 0.0)
    assert 0.75 <= a.appearance["age"] <= 1.0 and a.appearance["hair_tone"] >= 0.85
    assert expand(12, "elderly man, grey hair") != a
    with pytest.raises(ValueError, match="refused"):
        expand(1, "looks like someone famous")


def test_too_close_pairs_are_named() -> None:
    cast = {"C01": expand(5), "C02": expand(5), "C03": expand(6)}
    report = check_distinct(cast)
    assert not report.ok and ("C01", "C02", "appearance", 0.0) in report.too_close


def test_sampler_beats_independent_seeds() -> None:
    req = {f"C{i:02d}": "" for i in range(8)}
    baseline = sum(not check_distinct(sample_cast(req, s, 1)).ok for s in range(30))
    sampled = sum(not check_distinct(sample_cast(req, s, 32)).ok for s in range(30))
    assert sampled == 0 and baseline >= 10  # measured 0 % vs 71 % on 300 casts (D-7.4)
    assert check_distinct(sample_cast(req, 3)).min_appearance >= APPEARANCE_MIN


def test_collapse_counter() -> None:
    assert unique_identities([expand(7)] * 20) == 1
    assert unique_identities(list(sample_cast({f"C{i}": "" for i in range(10)}, 0).values())) == 10


def test_sample_script_cast_is_distinct_and_untampered() -> None:
    script = read_wscript(OUT)
    report, tampered = check_script(script.header["dictionary"])
    assert report.ok and not tampered
    d = copy.deepcopy(script.header["dictionary"])
    d["characters"]["C02"]["identity"]["appearance"]["age"] = 0.1  # hand-made "younger" Ade
    assert check_script(d)[1] == ["C02"] and not verify(d["characters"]["C02"]["identity"])
    assert "appearance" not in load_context(script, "S001").to_prompt()  # vectors stay out
