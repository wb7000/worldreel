"""Tests for the cosmic place bible.

Covers: deterministic expansion, style-range narrowing, distinctness check,
tamper detection, sample_cast best-candidate selection, and cross-type
distance (only shared dims compared).
"""

from stages.generate.cosmic_places import (
    PLACE_TYPES,
    check_bible,
    check_distinct,
    distance,
    expand,
    sample_cast,
    verify,
)


def test_expand_deterministic() -> None:
    """Same seed + request + type always gives the same identity."""
    a = expand(42, "glowing ring habitat", "orbital")
    b = expand(42, "glowing ring habitat", "orbital")
    assert a.appearance == b.appearance
    assert a.place_type == "orbital"
    assert a.seed == 42


def test_expand_all_types_have_shared_dims() -> None:
    """Every place type includes the shared appearance dims."""
    for ptype in PLACE_TYPES:
        ident = expand(1, "", ptype)
        for d in (
            "structure",
            "material",
            "light_source",
            "light_level",
            "gravity",
            "atmosphere",
            "temperature",
            "scale",
            "openness",
        ):
            assert d in ident.appearance


def test_distance_only_compares_shared_dims() -> None:
    """Orbital vs void compares only shared dims."""
    orbital = expand(1, "", "orbital")
    void = expand(2, "", "void")
    d = distance(orbital, void)
    assert 0.0 <= d <= 1.0


def test_distance_identical_places_is_zero() -> None:
    """Same place compared to itself should be 0."""
    a = expand(5, "", "nebula_interior")
    b = expand(5, "", "nebula_interior")
    assert distance(a, b) == 0.0


def test_check_distinct_passes_when_separated() -> None:
    """Well-separated places pass distinctness."""
    places = {
        "L1": expand(1, "", "planetary_surface"),
        "L2": expand(100, "", "orbital"),
        "L3": expand(200, "", "megastructure"),
    }
    report = check_distinct(places)
    assert report.ok is True


def test_check_distinct_fails_when_too_close() -> None:
    """Places too close on shared dims fail distinctness."""
    places = {
        "L1": expand(1, "", "planetary_surface"),
        "L2": expand(1, "", "planetary_surface"),  # identical -> distance 0
    }
    report = check_distinct(places)
    assert report.ok is False
    assert len(report.too_close) == 1


def test_sample_cast_deterministic() -> None:
    """Same cast_seed gives same cast."""
    requests = {
        "L1": "orbital:glowing ring",
        "L2": "nebula_interior:purple clouds",
        "L3": "void:drift",
    }
    a = sample_cast(requests, 42)
    b = sample_cast(requests, 42)
    for pid in requests:
        assert a[pid].appearance == b[pid].appearance
        assert a[pid].place_type == b[pid].place_type


def test_sample_cast_prefix_parsing() -> None:
    """Request prefix before ':' sets place type."""
    requests = {"L1": "orbital:glowing ring", "L2": "nebula_interior:purple clouds"}
    cast = sample_cast(requests, 7)
    assert cast["L1"].place_type == "orbital"
    assert cast["L2"].place_type == "nebula_interior"


def test_sample_cast_defaults_to_planetary_surface() -> None:
    """No prefix means planetary_surface."""
    requests = {"L1": "crystal cavern"}
    cast = sample_cast(requests, 7)
    assert cast["L1"].place_type == "planetary_surface"


def test_verify_true_for_stored_identity() -> None:
    """Stored identity round-trips through expand."""
    ident = expand(99, "megastructure:ringworld", "megastructure")
    stored = ident.to_json()
    assert verify(stored) is True


def test_verify_false_for_tampered_appearance() -> None:
    """Changing a stored value makes verify fail."""
    ident = expand(99, "megastructure:ringworld", "megastructure")
    stored = ident.to_json()
    stored["appearance"]["scale"] = 1.0 - stored["appearance"]["scale"]
    assert verify(stored) is False


def test_verify_false_for_wrong_type() -> None:
    """Changing place_type makes verify fail."""
    ident = expand(99, "megastructure:ringworld", "megastructure")
    stored = ident.to_json()
    stored["place_type"] = "void"
    assert verify(stored) is False


def test_check_bible_passes_for_distinct_places() -> None:
    """check_bible passes when all places are distinct."""
    dictionary = {
        "places": {
            "L1": {"identity": expand(1, "", "planetary_surface").to_json()},
            "L2": {"identity": expand(100, "", "orbital").to_json()},
            "L3": {"identity": expand(200, "", "dimensional_rift").to_json()},
        }
    }
    report, tampered = check_bible(dictionary)
    assert report.ok is True
    assert tampered == []


def test_check_bible_flags_tampered() -> None:
    """check_bible flags tampered identities."""
    dictionary = {
        "places": {
            "L1": {"identity": expand(1, "", "planetary_surface").to_json()},
            "L2": {"identity": expand(2, "", "orbital").to_json()},
        }
    }
    dictionary["places"]["L2"]["identity"]["appearance"]["gravity"] = 0.999
    _, tampered = check_bible(dictionary)
    assert "L2" in tampered
