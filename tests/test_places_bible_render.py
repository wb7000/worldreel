"""Tests for places bible renderer."""

import numpy as np

from stages.generate.places_bible import expand
from stages.generate.places_bible_render import render_place


def test_render_is_deterministic() -> None:
    ident = expand(7, "stone castle gate")
    a = render_place(ident, seed=42)
    b = render_place(ident, seed=42)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.colors, b.colors)
    assert len(a) > 100  # non-empty


def test_different_identities_differ() -> None:
    a = render_place(expand(7, "stone castle gate"), seed=42)
    b = render_place(expand(11, "wooden hut"), seed=42)
    assert not np.array_equal(a.positions, b.positions)


def test_different_seeds_differ() -> None:
    ident = expand(7, "stone castle")
    a = render_place(ident, seed=7)
    b = render_place(ident, seed=11)
    assert not np.array_equal(a.positions, b.positions)


def test_cloud_has_valid_columns() -> None:
    ident = expand(7, "stone castle")
    cloud = render_place(ident, seed=42)
    assert cloud.positions.shape == (len(cloud), 3)
    assert cloud.colors.shape == (len(cloud), 3)
    assert cloud.cls.shape == (len(cloud),)
    assert cloud.confidence.shape == (len(cloud),)
    assert (cloud.confidence >= 0).all() and (cloud.confidence <= 1).all()
    assert np.all(np.isfinite(cloud.positions))
    assert np.all(np.isfinite(cloud.colors))
