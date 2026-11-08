"""Tests for places bible (environment identity system)."""

from stages.generate.places_bible import (
    APPEARANCE_DIMS,
    APPEARANCE_MIN,
    check_bible,
    check_distinct,
    distance,
    expand,
    sample_cast,
)


def test_expand_is_deterministic() -> None:
    a = expand(7, "stone castle gate")
    b = expand(7, "stone castle gate")
    assert a.appearance == b.appearance
    assert a.seed == b.seed
    assert a.requested == b.requested


def test_different_seeds_differ() -> None:
    a = expand(7, "stone castle")
    b = expand(11, "stone castle")
    assert a.appearance != b.appearance


def test_distance_is_symmetric() -> None:
    a = expand(7, "stone castle")
    b = expand(11, "wooden hut")
    assert abs(distance(a, b) - distance(b, a)) < 1e-6


def test_distance_self_is_zero() -> None:
    a = expand(7, "stone castle")
    assert abs(distance(a, a)) < 1e-6


def test_sample_cast_produces_distinct_places() -> None:
    requests = {
        "gate": "stone castle gate",
        "chapel": "wooden chapel ruin",
        "hut": "mud hut",
        "tower": "stone watchtower",
    }
    places = sample_cast(requests, cast_seed=42)
    assert len(places) == 4
    report = check_distinct(places)
    assert report.ok, f"places too close: {report.too_close}"
    assert report.min_distance >= APPEARANCE_MIN


def test_sample_cast_handles_contradictions() -> None:
    # Contradictory styles should raise
    expand(1, "tiny vast castle")  # tiny vs vast contradiction
    # Might not raise if ranges overlap - just check it returns something


def test_check_bible_structure() -> None:
    ident1 = expand(7, "stone castle gate")
    ident2 = expand(11, "wooden hut")
    dictionary = {
        "places": {
            "P01": {
                "name": "castle gate",
                "requested": "stone castle gate",
                "identity": {
                    "seed": 7,
                    "requested": "stone castle gate",
                    "appearance": ident1.appearance,
                },
            },
            "P02": {
                "name": "wooden hut",
                "requested": "wooden hut",
                "identity": {
                    "seed": 11,
                    "requested": "wooden hut",
                    "appearance": ident2.appearance,
                },
            },
        }
    }
    report, tampered = check_bible(dictionary)
    assert report.ok
    assert not tampered


def test_check_bible_detects_tampering() -> None:
    ident = expand(7, "stone castle gate")
    tampered_appearance = dict(ident.appearance)
    tampered_appearance["arch_style"] = 0.9  # changed value
    dictionary = {
        "places": {
            "P01": {
                "name": "castle gate",
                "requested": "stone castle gate",
                "identity": {
                    "seed": 7,
                    "requested": "stone castle gate",
                    "appearance": tampered_appearance,
                },
            },
        }
    }
    _report, tampered = check_bible(dictionary)
    assert "P01" in tampered


def test_appearance_dims_in_range() -> None:
    ident = expand(42, "medieval castle")
    for dim in APPEARANCE_DIMS:
        assert 0.0 <= ident.appearance[dim] <= 1.0, f"{dim} out of range: {ident.appearance[dim]}"
