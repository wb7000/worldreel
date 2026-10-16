from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from formats.common import FormatError, dump_json, make_signature
from formats.wreel import (
    MANIFEST,
    POINT_STRIDE,
    SIGNATURE,
    PointCloud,
    read_wreel,
    validate_wreel,
    write_wreel,
)

Sample = tuple[dict[str, Any], dict[str, bytes]]


def _write(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey, name: str = "s.wreel") -> Path:
    path = tmp_path / name
    write_wreel(path, sample[0], sample[1], key)
    return path


def _rewrite_zip(src: Path, dst: Path, changes: dict[str, bytes | None]) -> None:
    """Copy a zip, replacing (bytes) or dropping (None) entries, without re-signing."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            if info.filename in changes:
                data = changes[info.filename]
                if data is not None:
                    zout.writestr(info.filename, data)
            else:
                zout.writestr(info, zin.read(info))
        for name, data in changes.items():
            if data is not None and name not in zin.namelist():
                zout.writestr(name, data)


def _write_unvalidated(
    path: Path, manifest: dict[str, Any], entries: dict[str, bytes], key: Ed25519PrivateKey
) -> None:
    write_wreel(path, manifest, entries, key, validate=False)


# --------------------------------------------------------------------------- round trip


def test_round_trip_preserves_everything(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    manifest, entries = sample
    path = _write(tmp_path, sample, key)
    doc = read_wreel(path)
    assert doc.report.ok, doc.report.issues
    assert doc.signature_status == "valid"
    assert doc.entries == entries
    expected = {k: v for k, v in manifest.items() if k != "files"}
    assert {k: v for k, v in doc.manifest.items() if k != "files"} == expected
    assert set(doc.manifest["files"]) == set(entries)


def test_write_is_deterministic_and_rewrite_is_byte_identical(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    first = _write(tmp_path, sample, key, "a.wreel")
    second = _write(tmp_path, sample, key, "b.wreel")
    assert first.read_bytes() == second.read_bytes()
    doc = read_wreel(first)
    third = tmp_path / "c.wreel"
    write_wreel(third, doc.manifest, doc.entries, key)
    assert third.read_bytes() == first.read_bytes()


def test_container_layout(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    path = _write(tmp_path, sample, key)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert names[:2] == [MANIFEST, SIGNATURE]
    assert names[2:] == sorted(names[2:])


def test_point_cloud_pack_unpack() -> None:
    rng = np.random.default_rng(0)
    cloud = PointCloud(
        rng.normal(size=(50, 3)).astype(np.float32),
        rng.integers(0, 256, size=(50, 3)).astype(np.uint8),
        rng.integers(0, 5, size=50).astype(np.uint8),
        rng.uniform(size=50).astype(np.float32),
    )
    data = cloud.pack()
    assert len(data) == 50 * POINT_STRIDE
    assert PointCloud.unpack(data) == cloud


def test_point_record_layout_is_little_endian_20_bytes() -> None:
    cloud = PointCloud(
        np.array([[1.0, 2.0, 3.0]], np.float32),
        np.array([[10, 20, 30]], np.uint8),
        np.array([2], np.uint8),
        np.array([0.5], np.float32),
    )
    data = cloud.pack()
    assert data[:4] == np.float32(1.0).astype("<f4").tobytes()
    assert data[12:16] == bytes([10, 20, 30, 2])
    assert data[16:20] == np.float32(0.5).astype("<f4").tobytes()


# --------------------------------------------------------------------------- required sections


def test_audio_slot_is_required(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    manifest, entries = sample
    del manifest["audio"]
    with pytest.raises(FormatError):
        write_wreel(tmp_path / "x.wreel", manifest, entries, key)


def test_empty_audio_slot_is_valid(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    manifest, entries = sample
    manifest["audio"] = {"tracks": [], "events": [], "mood": []}
    assert read_wreel(_write(tmp_path, (manifest, entries), key)).report.ok


@pytest.mark.parametrize("field", ["ai_label", "notice", "rights", "identity_policy"])
def test_provenance_fields_are_required(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey, field: str
) -> None:
    manifest, entries = sample
    del manifest["provenance"][field]
    with pytest.raises(FormatError):
        write_wreel(tmp_path / "x.wreel", manifest, entries, key)


def test_unknown_ai_label_rejected(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    manifest, entries = sample
    manifest["provenance"]["ai_label"] = "Real footage"
    with pytest.raises(FormatError):
        write_wreel(tmp_path / "x.wreel", manifest, entries, key)


def test_person_needs_identity_policy(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    manifest, entries = sample
    del manifest["actors"][0]["identity"]
    with pytest.raises(FormatError):
        write_wreel(tmp_path / "x.wreel", manifest, entries, key)


def test_people_with_identity_policy_none_rejected(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    manifest, entries = sample
    manifest["provenance"]["identity_policy"] = "none"
    path = tmp_path / "x.wreel"
    _write_unvalidated(path, manifest, entries, key)
    assert "identity_policy" in validate_wreel(path).codes()


def test_interaction_rules_cannot_carry_code(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    manifest, entries = sample
    manifest["interaction"]["rules"][0]["action"]["script"] = "fetch('https://evil')"
    with pytest.raises(FormatError):
        write_wreel(tmp_path / "x.wreel", manifest, entries, key)


# --------------------------------------------------------------------------- integrity


def test_tampered_entry_detected(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    good = _write(tmp_path, sample, key)
    bad = tmp_path / "bad.wreel"
    other_track = b'{"times": [0], "positions": [[0,0,0]], "rotations": [[0,0,0,1]]}\n'
    _rewrite_zip(good, bad, {"actors/a1/track.json": other_track})
    report = validate_wreel(bad)
    assert "hash_mismatch" in report.codes()
    assert not report.ok


def test_tampered_manifest_breaks_signature(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    good = _write(tmp_path, sample, key)
    with zipfile.ZipFile(good) as zf:
        manifest = json.loads(zf.read(MANIFEST))
    manifest["provenance"]["ai_label"] = "AI-reconstructed"
    bad = tmp_path / "bad.wreel"
    _rewrite_zip(good, bad, {MANIFEST: dump_json(manifest)})
    doc = read_wreel(bad)
    assert doc.signature_status == "invalid"
    assert "signature_invalid" in doc.report.codes()


def test_missing_signature_reported(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    good = _write(tmp_path, sample, key)
    bad = tmp_path / "bad.wreel"
    _rewrite_zip(good, bad, {SIGNATURE: None})
    doc = read_wreel(bad)
    assert doc.signature_status == "missing"
    assert not doc.report.ok


def test_extra_unlisted_entry_rejected(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    good = _write(tmp_path, sample, key)
    bad = tmp_path / "bad.wreel"
    _rewrite_zip(good, bad, {"extra/payload.bin": b"\x00" * 10})
    assert "unlisted_entry" in validate_wreel(bad).codes()


def test_unsafe_zip_path_is_unreadable(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    good = _write(tmp_path, sample, key)
    bad = tmp_path / "bad.wreel"
    _rewrite_zip(good, bad, {"../escape.txt": b"x"})
    with pytest.raises(FormatError):
        read_wreel(bad)
    assert "unreadable" in validate_wreel(bad).codes()


def test_not_a_zip_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "junk.wreel"
    path.write_bytes(b"not a zip")
    assert "unreadable" in validate_wreel(path).codes()


# --------------------------------------------------------------------------- semantics


def test_unknown_actor_reference(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    manifest, entries = sample
    manifest["pov_presets"][2]["actor_id"] = "ghost"
    path = tmp_path / "x.wreel"
    _write_unvalidated(path, manifest, entries, key)
    assert "bad_reference" in validate_wreel(path).codes()


def test_point_count_mismatch(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    manifest, entries = sample
    manifest["scene"]["layers"][0]["lods"][0]["count"] += 1
    path = tmp_path / "x.wreel"
    _write_unvalidated(path, manifest, entries, key)
    assert "bad_point_chunk" in validate_wreel(path).codes()


def test_confidence_out_of_range(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    manifest, entries = sample
    path0 = manifest["scene"]["layers"][0]["lods"][0]["path"]
    cloud = PointCloud.unpack(entries[path0])
    cloud.confidence[0] = 1.5
    entries[path0] = cloud.pack()
    path = tmp_path / "x.wreel"
    _write_unvalidated(path, manifest, entries, key)
    assert "bad_confidence" in validate_wreel(path).codes()


def test_track_times_must_increase(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    manifest, entries = sample
    track = json.loads(entries["actors/a1/track.json"])
    track["times"][2] = track["times"][1]
    entries["actors/a1/track.json"] = dump_json(track)
    path = tmp_path / "x.wreel"
    _write_unvalidated(path, manifest, entries, key)
    assert "bad_times" in validate_wreel(path).codes()


def test_non_unit_quaternion(tmp_path: Path, sample: Sample, key: Ed25519PrivateKey) -> None:
    manifest, entries = sample
    manifest["cameras"][0]["keyframes"][0]["rotation"] = [0.0, 0.0, 0.0, 2.0]
    path = tmp_path / "x.wreel"
    _write_unvalidated(path, manifest, entries, key)
    assert "bad_quaternion" in validate_wreel(path).codes()


# --------------------------------------------------------------------------- versioning


def test_newer_minor_version_reads_with_warning(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    manifest, entries = sample
    manifest["format_version"] = "1.7"
    manifest["future_feature"] = {"anything": True}
    path = _write(tmp_path, (manifest, entries), key)
    report = validate_wreel(path)
    assert report.ok
    assert {"newer_minor_version", "unknown_field"} <= report.codes()


def test_unknown_field_rejected_in_current_version(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    manifest, entries = sample
    manifest["future_feature"] = {}
    with pytest.raises(FormatError):
        write_wreel(tmp_path / "x.wreel", manifest, entries, key)


def test_newer_major_version_unreadable(
    tmp_path: Path, sample: Sample, key: Ed25519PrivateKey
) -> None:
    good = _write(tmp_path, sample, key)
    with zipfile.ZipFile(good) as zf:
        manifest = json.loads(zf.read(MANIFEST))
    manifest["format_version"] = "2.0"
    raw = dump_json(manifest)
    bad = tmp_path / "bad.wreel"
    _rewrite_zip(good, bad, {MANIFEST: raw, SIGNATURE: make_signature(MANIFEST, raw, key)})
    with pytest.raises(FormatError, match="major version 2"):
        read_wreel(bad)
