"""Tests for A-9 procedural places (stages/generate/places.py)."""

from typing import Any

import numpy as np

from stages.generate.places import GENERATED, Place, _pick_template, _size_for, build_place


def _entry(name: str, description: str) -> dict[str, Any]:
    return {"id": "L01", "name": name, "description": description}


def test_place_is_non_empty_generated_cloud() -> None:
    place = build_place(_entry("the fort gate", "stone arch, iron chain, monsoon path"), 7)
    assert isinstance(place, Place)
    assert len(place.cloud) > 500
    assert (place.cloud.cls == GENERATED).all()
    assert (place.cloud.confidence == 1.0).all()
    assert place.bounds["min"][1] == 0.0  # floor at y=0
    assert place.bounds["max"][1] > 0


def test_place_is_deterministic() -> None:
    a = build_place(_entry("the rampart", "sea wall walk, wet battlements"), 11)
    b = build_place(_entry("the rampart", "sea wall walk, wet battlements"), 11)
    assert np.array_equal(a.cloud.positions, b.cloud.positions)
    assert np.array_equal(a.cloud.colors, b.cloud.colors)
    assert a.bounds == b.bounds


def test_different_locations_differ() -> None:
    a = build_place(_entry("the chapel ruin", "roofless nave, fig roots, low arch"), 3)
    b = build_place(_entry("Ade's hut", "one whitewashed room, oil lamp"), 3)
    assert not np.array_equal(a.cloud.positions, b.cloud.positions)
    assert a.bounds != b.bounds


def test_spots_inside_bounds() -> None:
    place = build_place(_entry("the passage", "steps into the hill, iron gate, channel"), 5)
    lo = np.array(place.bounds["min"])
    hi = np.array(place.bounds["max"])
    for spot in place.spots.values():
        p = np.array(spot)
        assert np.all(p[:2] >= lo[:2] - 1e-6) and np.all(p[:2] <= hi[:2] + 1e-6)
        assert lo[1] <= p[1] <= hi[1]


def test_template_picks_by_keyword() -> None:
    assert _pick_template("sea wall walk", "rampart").__name__ == "_wall_walk"
    assert _pick_template("one whitewashed room", "hut").__name__ == "_room"
    assert _pick_template("stone arch, iron chain", "gate").__name__ == "_gate"
    assert _pick_template("roofless nave", "chapel").__name__ == "_ruin"
    assert _pick_template("steps into the hill", "passage").__name__ == "_tunnel"
    assert _pick_template("something unknown", "odd").__name__ == "_room"


def test_size_knows_big_places() -> None:
    assert _size_for("sea wall walk", "rampart") > _size_for("one whitewashed room", "hut")
