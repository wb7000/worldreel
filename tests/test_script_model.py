from __future__ import annotations

import random

from formats.wscript import Dialogue, parse_scene, read_wscript
from stages.generate.continuity import check
from stages.generate.script_model import SYSTEM, TARGET_LINES, examples, make_example, score
from tools.make_sample_wscript import OUT
from tools.plays_to_wscript import convert

PLAY = """Title: A Tiny Test Play
*** START OF THE PROJECT GUTENBERG EBOOK TINY ***

CONTENTS
ACT I
ACT II

A preface nobody performs.

ACT I

A garden at dusk. A bench, a lamp.

ANNA. Is it late?

BORIS [looking at his watch] Not yet. [He sits.] Sit with me, Anna.

ANNA. I cannot stay long.

BORIS.
Then stay short.

ACT II

The same garden, morning.

ANNA. You waited all night?

BORIS. [Yawning.] All night.

CLARA. Breakfast is ready, both of you.

ANNA. Coming!

CLARA. The tea is getting cold.

BORIS. Then we drink it cold.

CLARA. Suit yourselves.

THE END

*** END OF THE PROJECT GUTENBERG EBOOK TINY ***
License text that must not be converted.
"""


def test_play_conversion_keeps_speeches_directions_and_continuity() -> None:
    script = convert(PLAY, "A Tiny Test Play")
    assert script is not None
    names = {c["name"] for c in script.header["dictionary"]["characters"].values()}
    assert names == {"Anna", "Boris", "Clara"}  # names with 3+ speeches
    s1 = [s for s in script.scenes["S001"] if isinstance(s, Dialogue)]
    assert [d.text for d in s1][:2] == ["Is it late?", "Not yet."]
    assert s1[1].manner == "looking at his watch"
    text = "".join(str(s) for v in script.scenes.values() for s in v)
    assert "preface" not in text and "License" not in text and "CONTENTS" not in text
    report, _ = check(script)
    assert report.ok, report.issues  # Clara enters in act II: declared offscreen, no false alarm


def test_examples_cut_scenes_and_hide_the_answer() -> None:
    script = read_wscript(OUT)
    ex = make_example(script, "S003", 10)
    assert ex.target.count("\n") == TARGET_LINES - 1
    assert ex.target.split("\n")[0] not in ex.prompt  # the answer is not in the prompt
    assert ex.prompt.rstrip().endswith(f"next {TARGET_LINES} lines.")
    assert ex.messages()[0]["content"] == SYSTEM
    many = list(examples(script, 6, random.Random(0)))
    assert len(many) >= 15 and all(
        score(e.target, script.header["dictionary"]).refs_ok for e in many
    )


def test_score_rejects_invented_ids_and_prose() -> None:
    d = read_wscript(OUT).header["dictionary"]
    assert score("C01: Hello.\n! {C02} nods.", d) == score(
        "```\nC01: Hello.\n! {C02} nods.\n```", d
    )
    assert score("C01: Hello.", d).refs_ok
    assert not score("C09: Who am I?", d).refs_ok  # no such character
    assert not score("! {P99} falls.", d).refs_ok  # no such prop
    assert not score("Mira: Hello.", d).syntax_ok  # a name, not an ID
    assert not score("Sure! Here is the scene:", d).syntax_ok
    assert parse_scene("@scene S1\n" + make_example(read_wscript(OUT), "S001", 6).target + "\n")
