from __future__ import annotations

from pathlib import Path

from formats.wscript import read_wscript
from stages.generate.voicing import scene_chunks
from tools.make_sample_wscript import OUT


def test_chunks_never_hold_more_than_two_speakers_and_keep_every_line_in_order() -> None:
    script = read_wscript(OUT)
    for sid in script.scenes:
        chunks = scene_chunks(script, sid)
        assert all(len(c.speakers) <= 2 for c in chunks)
        indices = [t.index for c in chunks for t in c.turns]
        assert indices == sorted(indices) and len(indices) == len(set(indices))


def test_dia_text_uses_the_delivery_layer_and_dia_nonverbals() -> None:
    script = read_wscript(OUT)
    first = scene_chunks(script, "S001")[0]
    i = first.turns[0].index
    script.delivery["S001"] = {"lines": {str(i): {"spoken": "Hi! (laugh) I'm Mira."}}}
    text = scene_chunks(script, "S001")[0].dia_text("Ref.")
    assert text.startswith("[S1] Ref. [S2] Ref. [S1] Hi! (laughs) I'm Mira.")


def test_three_speaker_scene_is_split() -> None:
    script = read_wscript(OUT)
    multi = [
        sid
        for sid in script.scenes
        if len({t.speaker for c in scene_chunks(script, sid) for t in c.turns}) > 2
    ]
    assert multi and all(len(scene_chunks(script, sid)) > 1 for sid in multi)
    assert Path(OUT).exists()
