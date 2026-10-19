from __future__ import annotations

import numpy as np

from formats.wscript import read_wscript
from stages.generate.voices import (
    centroid_cosines,
    delivery,
    nearest_centroid_accuracy,
    pitch_spread_semitones,
    script_lines,
)
from tools.make_sample_wscript import OUT


def test_every_line_gets_identity_plus_delivery_and_never_a_named_speaker() -> None:
    lines = script_lines(read_wscript(OUT))
    assert len(lines) == 101 and {ln.character for ln in lines} == {"C01", "C02", "C03", "C04"}
    for ln in lines:
        assert ln.expressive.startswith(ln.flat.split(" with a very close")[0])  # same identity
        assert "monotone" not in ln.expressive and "{" not in ln.text
        assert ln.expressive.startswith("A ")  # a description, not a named speaker ("Jon's ...")


def test_delivery_follows_tension_manner_and_cues() -> None:
    assert delivery("Fine.", None, 1)[0].startswith("calm")
    assert delivery("Fine.", None, 9)[0].startswith("urgent")
    words, emotions = delivery("No, no, no. Not the trees.", "under his breath", 5)
    assert "whispering" in words and "afraid" in words and "afraid" in emotions
    assert "questioning" in delivery("Where is the drone?", None, 4)[1]
    assert delivery("Fine.", None, 4)[1] == ["neutral"]


def test_pitch_spread_separates_flat_from_moving_pitch() -> None:
    sr = 16000
    t = np.arange(sr * 2) / sr
    flat = np.sin(2 * np.pi * 150 * t)
    glide = np.sin(2 * np.pi * np.cumsum(150 * 2 ** (np.sin(2 * np.pi * 1.5 * t) * 4 / 12)) / sr)
    assert pitch_spread_semitones(flat, sr) < 0.2 < 1.5 < pitch_spread_semitones(glide, sr)


def test_voice_identity_metrics() -> None:
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(3, 64))
    emb = np.concatenate([c + 0.3 * rng.normal(size=(10, 64)) for c in centres])
    labels = [f"C{k}" for k in range(3) for _ in range(10)]
    assert nearest_centroid_accuracy(emb, labels) == 1.0
    assert all(v < 0.6 for v in centroid_cosines(emb, labels).values())
    mixed = rng.normal(size=(30, 64))  # one voice for everyone = collapse
    assert nearest_centroid_accuracy(mixed, labels) < 0.6


def test_reference_set_minimises_the_closest_pair_and_guard_refuses_others(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import hashlib

    import pytest

    from stages.generate.voices import (
        UndesignedReferenceError,
        best_reference_set,
        chatterbox_params,
        check_reference,
    )

    e = np.eye(4)
    cands = {"A": np.stack([e[0], e[1]]), "B": np.stack([e[0], e[2]])}  # A0 == B0: avoid that pair
    choice, worst = best_reference_set(cands)
    assert worst == 0.0 and not (choice["A"] == 0 and choice["B"] == 0)
    ok = tmp_path / "ok.wav"
    ok.write_bytes(b"designed")
    record = {"A": {"file": "ok.wav", "sha256": hashlib.sha256(b"designed").hexdigest()}}
    check_reference(str(ok), record)
    bad = tmp_path / "someone.wav"
    bad.write_bytes(b"a real person's recording")
    with pytest.raises(UndesignedReferenceError):
        check_reference(str(bad), record)
    calm, tense = chatterbox_params(2, ("neutral",)), chatterbox_params(9, ("afraid",))
    assert calm[0] < 0.5 < tense[0] and tense[1] == 0.3
