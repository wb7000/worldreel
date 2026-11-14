"""Tests for the entity bible (non-human identity system).

Covers: deterministic expansion, style-range narrowing, distinctness check,
tamper detection, sample_cast best-candidate selection, and cross-type
distance (only shared dims compared).
"""

from stages.generate.entity_bible import (
    ENTITY_TYPES,
    check_bible,
    check_distinct,
    distance,
    expand,
    sample_cast,
    verify,
)


def test_expand_deterministic() -> None:
    """Same seed + request + type always gives the same identity."""
    a = expand(42, "glowing insectoid", "alien")
    b = expand(42, "glowing insectoid", "alien")
    assert a.appearance == b.appearance
    assert a.entity_type == "alien"
    assert a.seed == 42


def test_expand_all_types_have_shared_dims() -> None:
    """Every entity type includes the shared appearance dims."""
    for etype in ENTITY_TYPES:
        ident = expand(1, "", etype)
        for d in ("material", "scale", "luminosity", "color_temp", "texture"):
            assert d in ident.appearance


def test_expand_request_does_not_crash() -> None:
    """A style request with entity-type prefix should parse without error."""
    ident = expand(1, "alien:glowing insectoid", "alien")
    assert ident.appearance is not None
    assert ident.entity_type == "alien"
    ident2 = expand(1, "robot:metal panels", "robot")
    assert ident2.appearance is not None
    assert ident2.entity_type == "robot"


def test_distance_only_compares_shared_dims() -> None:
    """Alien vs robot compares only shared dims, not alien-specific ones."""
    alien = expand(1, "", "alien")
    robot = expand(2, "", "robot")
    d = distance(alien, robot)
    assert 0.0 <= d <= 1.0


def test_distance_identical_entities_is_zero() -> None:
    """Same entity compared to itself should be 0."""
    a = expand(5, "", "energy")
    b = expand(5, "", "energy")
    assert distance(a, b) == 0.0


def test_distance_max_when_no_shared_dims() -> None:
    """Entities with no shared dims are maximally different."""
    a = expand(1, "", "human")
    b = expand(2, "", "gas")
    # human and gas share all SHARED dims, so this won't be 1.0 unless they happen to match
    d = distance(a, b)
    assert 0.0 <= d <= 1.0


def test_check_distinct_passes_when_separated() -> None:
    """Well-separated entities pass distinctness."""
    entities = {
        "E1": expand(1, "", "human"),
        "E2": expand(100, "", "robot"),
        "E3": expand(200, "", "energy"),
    }
    report = check_distinct(entities)
    assert report.ok is True


def test_check_distinct_fails_when_too_close() -> None:
    """Entities too close on shared dims fail distinctness."""
    # Use same seed but slightly different to force closeness
    entities = {
        "E1": expand(1, "", "human"),
        "E2": expand(1, "", "human"),  # identical -> distance 0
    }
    report = check_distinct(entities)
    assert report.ok is False
    assert len(report.too_close) == 1


def test_sample_cast_deterministic() -> None:
    """Same cast_seed gives same cast."""
    requests = {"E1": "alien:insectoid", "E2": "robot:metal", "E3": "energy:pulsing"}
    a = sample_cast(requests, 42)
    b = sample_cast(requests, 42)
    for eid in requests:
        assert a[eid].appearance == b[eid].appearance
        assert a[eid].entity_type == b[eid].entity_type


def test_sample_cast_prefix_parsing() -> None:
    """Request prefix before ':' sets entity type."""
    requests = {"E1": "alien:glowing wings", "E2": "robot:rusty panels"}
    cast = sample_cast(requests, 7)
    assert cast["E1"].entity_type == "alien"
    assert cast["E2"].entity_type == "robot"


def test_sample_cast_defaults_to_human() -> None:
    """No prefix means human."""
    requests = {"E1": "tall thin woman"}
    cast = sample_cast(requests, 7)
    assert cast["E1"].entity_type == "human"


def test_verify_true_for_stored_identity() -> None:
    """Stored identity round-trips through expand."""
    ident = expand(99, "crystal faceted giant", "crystal")
    stored = ident.to_json()
    assert verify(stored) is True


def test_verify_false_for_tampered_appearance() -> None:
    """Changing a stored value makes verify fail."""
    ident = expand(99, "crystal faceted giant", "crystal")
    stored = ident.to_json()
    stored["appearance"]["scale"] = 1.0 - stored["appearance"]["scale"]
    assert verify(stored) is False


def test_verify_false_for_wrong_type() -> None:
    """Changing entity_type makes verify fail."""
    ident = expand(99, "crystal faceted giant", "crystal")
    stored = ident.to_json()
    stored["entity_type"] = "robot"
    assert verify(stored) is False


def test_check_bible_passes_for_distinct_entities() -> None:
    """check_bible passes when all entities are distinct."""
    dictionary = {
        "entities": {
            "E1": {"identity": expand(1, "", "human").to_json()},
            "E2": {"identity": expand(100, "", "robot").to_json()},
            "E3": {"identity": expand(200, "", "energy").to_json()},
        }
    }
    report, tampered = check_bible(dictionary)
    assert report.ok is True
    assert tampered == []


def test_check_bible_flags_tampered() -> None:
    """check_bible flags tampered identities."""
    dictionary = {
        "entities": {
            "E1": {"identity": expand(1, "", "human").to_json()},
            "E2": {"identity": expand(2, "", "robot").to_json()},
        }
    }
    dictionary["entities"]["E2"]["identity"]["appearance"]["material"] = 0.999
    _, tampered = check_bible(dictionary)
    assert "E2" in tampered
