from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stages.audio.beds import ambience, music, tension_curve
from stages.audio.stems import MIX_GAINS, mixdown, render_stems, stem_tracks
from stages.audio.synth import SAMPLE_RATE, footstep, wav_bytes, wav_samples


def test_tension_curve_steps_and_glides() -> None:
    c = tension_curve([{"t": 0.0, "tension": 2}, {"t": 5.0, "tension": 8}], 10.0)
    assert len(c) == 10 * SAMPLE_RATE
    assert abs(c[0] - 2) < 1e-6 and abs(c[-1] - 8) < 1e-6
    mid = c[int(5.0 * SAMPLE_RATE)]
    assert 3 < mid < 7  # smoothed, not a hard jump


def test_beds_are_deterministic_bounded_and_rain_adds_highs() -> None:
    mood = [{"t": 0.0, "tension": 5}]
    assert np.array_equal(music(2.0, mood, seed=1), music(2.0, mood, seed=1))
    assert np.abs(music(2.0, [{"t": 0.0, "tension": 10}])).max() <= 1.0
    dry, rain = ambience(3.0, mood, "dry"), ambience(3.0, mood, "rain")

    def high_share(x: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
        return float(spec[f > 2000].sum() / spec.sum())

    assert high_share(rain) > 10 * high_share(dry)


def test_music_gets_louder_and_brighter_from_calm_to_intense() -> None:
    calm = music(10.0, [{"t": 0.0, "tension": 1}]).astype(np.float64)
    tense = music(10.0, [{"t": 0.0, "tension": 10}]).astype(np.float64)
    assert np.sqrt((tense**2).mean()) > np.sqrt((calm**2).mean())

    def centroid(x: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
        return float((f * spec).sum() / spec.sum())

    assert centroid(tense) > centroid(calm) + 50


def test_stems_are_separate_tracks_and_the_mix_is_their_weighted_sum() -> None:
    clip = wav_bytes(footstep({"surface": "wood"}))
    events = [
        {"t": 0.5, "type": "footstep", "attributes": {"clip": "c.wav"}},
        {"t": 1.0, "type": "footstep", "attributes": {"clip": "c.wav"}},
    ]
    stems = render_stems(2.0, events, [{"t": 0.0, "tension": 6}], {"c.wav": clip}, "rain")
    assert set(stems) == {"ambience", "music", "foley"}
    tracks, entries = stem_tracks(stems)
    assert {t["layer"] for t in tracks} == {"ambience", "music", "foley"}
    assert all(t["license"] == "synthetic" and t["path"] in entries for t in tracks)
    back = {t["layer"]: wav_samples(entries[t["path"]]) for t in tracks}
    expected = sum(MIX_GAINS[k] * back[k].astype(np.float64) for k in back)
    mix = mixdown(back)
    scale = 0.99 / np.abs(expected).max() if np.abs(expected).max() > 0.99 else 1.0
    assert np.abs(mix - expected * scale).max() < 1e-5
    # The footsteps land at their event times in the foley stem.
    foley = back["foley"]
    assert np.abs(foley[: int(0.45 * SAMPLE_RATE)]).max() < 1e-4
    assert np.abs(foley[int(0.5 * SAMPLE_RATE) : int(0.6 * SAMPLE_RATE)]).max() > 0.05


def test_export_writes_one_file_per_layer_plus_mix(tmp_path: Path) -> None:
    from formats.common import load_or_create_signing_key
    from formats.wreel import write_wreel
    from tools.export_stems import export
    from tools.make_sample_wreel import build_sample

    manifest, entries = build_sample()
    stems = render_stems(1.0, [], [{"t": 0.0, "tension": 4}], {})
    tracks, stem_entries = stem_tracks(stems)
    manifest["audio"]["tracks"] = tracks
    entries.update(stem_entries)
    world = tmp_path / "w.wreel"
    write_wreel(world, manifest, entries, load_or_create_signing_key(tmp_path / "k.pem"))
    rows = export(world, tmp_path / "stems")
    names = sorted(str(r["file"]) for r in rows)
    assert names == ["ambience.wav", "mix.wav", "music.wav"]
    listing = json.loads((tmp_path / "stems" / "stems.json").read_text(encoding="utf-8"))
    assert len(listing) == 3 and all((tmp_path / "stems" / r["file"]).is_file() for r in listing)
