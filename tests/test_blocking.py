"""Tests for blocking (stages/generate/blocking.py)."""

from typing import Any

from formats.wscript import Change, Dialogue
from stages.generate.blocking import Blocking, plan_blocking
from stages.generate.places import Place, build_place


def _scene_place() -> Place:
    return build_place(
        {
            "id": "L01",
            "name": "the fort gate",
            "description": "stone arch, iron chain, monsoon path",
        },
        7,
    )


def test_present_characters_get_spots_and_face_each_other() -> None:
    continuity: dict[str, Any] = {
        "__location__": "L01",
        "characters": {
            "C01": {"location": "L01"},
            "C02": {"location": "L01"},
            "C03": {"location": "L02"},
        },
    }
    stmts: list[Any] = []
    bl: Blocking = plan_blocking(stmts, continuity, _scene_place(), None, 12.0)
    assert bl.present == ["C01", "C02"]
    assert set(bl.spots) == {"C01", "C02"}
    assert bl.eye_target["C01"] != bl.eye_target["C02"]
    # they face each other: C01's target is C02's spot and vice versa
    assert bl.eye_target["C01"] == bl.spots["C02"]
    assert bl.eye_target["C02"] == bl.spots["C01"]
    for cid in bl.present:
        tr = bl.tracks[cid]
        assert tr["times"][0] == 0.0 and tr["times"][-1] == 12.0


def test_absent_character_has_no_spot_or_track() -> None:
    continuity: dict[str, Any] = {"__location__": "L01", "characters": {"C01": {"location": "L01"}}}
    bl: Blocking = plan_blocking([], continuity, _scene_place(), None, 12.0)
    assert "C02" not in bl.spots and "C02" not in bl.tracks


def test_move_becomes_walk_inside_duration() -> None:
    continuity: dict[str, Any] = {"__location__": "L01", "characters": {"C01": {"location": "L01"}}}
    place = _scene_place()
    home = place.spots[next(iter(place.spots))]
    far = max(
        place.spots.values(),
        key=lambda sp: ((sp[0] - home[0]) ** 2 + (sp[2] - home[2]) ** 2) ** 0.5,
    )
    stmts: list[Any] = [Dialogue("C01", "a"), Change("C01", (("location", "=", "L04"),))]
    timing: list[dict[str, Any]] = [{"index": 0, "start": 0.0, "end": 4.0}]
    bl: Blocking = plan_blocking(stmts, continuity, place, timing, 12.0)
    tr = bl.tracks["C01"]
    assert tr["times"][0] == 0.0 and tr["times"][-1] == 12.0
    assert tr["positions"][0] == home
    assert tr["positions"][-1] == list(far)
    assert len(tr["times"]) == len(tr["positions"]) == len(tr["rotations"])


def test_move_after_duration_is_a_hold() -> None:
    continuity: dict[str, Any] = {"__location__": "L01", "characters": {"C01": {"location": "L01"}}}
    stmts: list[Any] = [Dialogue("C01", "a"), Change("C01", (("location", "=", "L04"),))]
    timing: list[dict[str, Any]] = [{"index": 0, "start": 12.0, "end": 16.0}]
    bl: Blocking = plan_blocking(stmts, continuity, _scene_place(), timing, 12.0)
    tr = bl.tracks["C01"]
    assert tr["positions"][0] == tr["positions"][-1]


def test_timing_maps_statement_to_time() -> None:
    continuity: dict[str, Any] = {"__location__": "L01", "characters": {"C01": {"location": "L01"}}}
    timing: list[dict[str, Any]] = [
        {"index": 5, "start": 0.0, "end": 4.0},
        {"index": 7, "start": 9.0, "end": 13.0},
    ]
    stmts: list[Any] = [
        Dialogue("C01", "a"),
        Dialogue("C01", "b"),
        Change("C01", (("location", "=", "L04"),)),
        Dialogue("C01", "c"),
    ]
    bl: Blocking = plan_blocking(stmts, continuity, _scene_place(), timing, 20.0)
    tr = bl.tracks["C01"]
    assert tr["times"][0] < 20.0  # a walk started before the end
    assert tr["times"][-1] == 20.0
