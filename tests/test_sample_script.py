from __future__ import annotations

from pathlib import Path

from eval.continuity import run
from formats.wscript import Dialogue, load_context, read_wscript, serialize_scene
from stages.generate.continuity import check
from stages.generate.screenplay import runtime
from tools.make_sample_wscript import OUT, build


def test_committed_script_matches_the_builder() -> None:
    on_disk = read_wscript(OUT)
    assert on_disk.report.ok and not on_disk.report.warnings
    built = build()
    assert on_disk.header["scenes"] == [
        {**e, "path": f"scenes/{e['id']}.wsl", "sha256": d["sha256"]}
        for e, d in zip(built.header["scenes"], on_disk.header["scenes"], strict=True)
    ]
    assert {k: serialize_scene(v) for k, v in on_disk.scenes.items()} == {
        k: serialize_scene(v) for k, v in built.scenes.items()
    }


def test_ten_minute_script_by_the_page_convention() -> None:
    rt = runtime(build())
    assert 9.0 <= rt.pages <= 11.0  # 1 page ~ 1 minute (D-7.2)
    assert rt.dialogue_minutes >= 4.0  # dialogue alone, spoken at 150 wpm


def test_script_has_no_continuity_issues() -> None:
    report, per_scene = check(build())
    assert report.ok and not report.issues
    assert len(per_scene) == 10


def test_loads_scene_by_scene_with_its_continuity_state() -> None:
    script = read_wscript(OUT)
    full = sum(len(serialize_scene(v)) for v in script.scenes.values())
    for entry in script.header["scenes"]:
        ctx = load_context(script, entry["id"])
        assert set(ctx.continuity["characters"]) == {"C01", "C02", "C03", "C04"}
        assert ctx.scene_text == serialize_scene(script.scenes[entry["id"]])
        others = [s for sid, s in script.scenes.items() if sid != entry["id"]]
        lines = {
            x.text
            for stmts in others
            for x in stmts
            if isinstance(x, Dialogue) and len(x.text.split()) >= 6  # short lines recur
        }
        prompt = ctx.to_prompt()
        own = {x.text for x in script.scenes[entry["id"]] if isinstance(x, Dialogue)}
        assert not any(line in prompt for line in lines - own)  # only this scene's text
        assert len(ctx.other_summaries) == 9
        assert len(ctx.scene_text) < full / 4


def test_seeded_errors_are_caught() -> None:
    rates = run(trials=25, seed=1)
    assert rates["wrong_location"]["full"] == 1.0 and rates["edit_start"]["full"] == 1.0
    assert rates["phantom_prop"]["full"] == 1.0


def test_sample_path_is_a_directory() -> None:
    assert Path(OUT).is_dir()
