import numpy as np
import pytest

from stages.generate.bible import expand
from stages.generate.face_projection import body_colour, face_mask, tps
from stages.generate.human_preset import (
    CLOTHES_FEMALE,
    CLOTHES_MALE,
    FACE_TARGETS,
    LIDS,
    STRENGTH,
    face_targets,
    human_preset,
    mpfb_age,
    phenotype,
)


def test_age_anchors() -> None:
    assert mpfb_age(25) == pytest.approx(0.5)
    assert mpfb_age(90) == pytest.approx(1.0)
    assert mpfb_age(11) == pytest.approx(0.1875)
    assert mpfb_age(120) == 1.0
    assert 0.5 < mpfb_age(60) < 1.0


def test_phenotype_in_range_and_seeded() -> None:
    for seed in range(20):
        p = phenotype(expand(seed, ""))
        for k in ("gender", "age", "height", "weight"):
            assert 0.0 <= p[k] <= 1.0
        assert sum(p["race"].values()) == pytest.approx(1.0)
    assert phenotype(expand(3, "")) == phenotype(expand(3, ""))


def test_face_targets_known_names_and_bounded() -> None:
    names = {n for pairs in FACE_TARGETS.values() for pair in pairs for n in pair}
    names |= {t for t, _ in LIDS}
    for seed in range(20):
        for t in face_targets(expand(seed, "")):
            assert t["target"] in names
            assert 0.0 < t["value"] <= max(STRENGTH, max(v for _, v in LIDS))


def test_preset_assets_and_no_logo_clothes() -> None:
    for seed in range(30):
        p = human_preset(expand(seed, ""))
        assert p["skin_mhmat"].endswith(".mhmat")
        assert all(c.endswith(".mhclo") for c in p["clothes"])
        suit = p["clothes"][0].removesuffix(".mhclo")
        assert suit in CLOTHES_MALE + CLOTHES_FEMALE
        assert suit not in ("male_casualsuit02", "male_casualsuit04", "female_casualsuit01")
        assert 1.5 <= p["worldreel"]["height_m"] <= 1.95
        assert all(0.0 <= c <= 1.0 for c in p["worldreel"]["hair_rgb"])


def test_tps_interpolates_and_keeps_affine() -> None:
    rng = np.random.default_rng(0)
    c = rng.uniform(0, 100, (9, 2))
    v = c + rng.normal(0, 3, (9, 2))
    assert np.abs(tps(c, v, c) - v).max() < 1e-8
    assert np.abs(tps(c, 2 * c + 5, np.array([[40.0, 60.0]])) - [85.0, 125.0]).max() < 1e-8


def test_face_mask_covers_face_not_corners() -> None:
    lm = np.array([[400, 450], [620, 450], [510, 560], [430, 660], [590, 660]], float)
    m = face_mask(lm, (1024, 1024))
    assert m[520, 510] > 0.95  # between eyes and mouth
    assert m[5, 5] < 0.01 and m[1000, 1000] < 0.01
    assert m[950, 510] < 0.05  # below the chin


def test_body_colour_keeps_texture_lightness() -> None:
    tex, port = (0.435, 0.251, 0.153), (0.686, 0.608, 0.58)
    out = body_colour(tex, port)

    def lum(c: tuple[float, float, float]) -> float:
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    assert abs(lum(out) - lum(tex)) < 0.06  # not the bright studio-lit portrait
