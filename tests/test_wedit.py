from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from formats.common import FormatError
from formats.wedit import PROVENANCE_TXT, read_wedit, validate_wedit, write_wedit
from tests.conftest import FIXED_TIME


def make_bundle() -> tuple[dict[str, Any], dict[str, bytes]]:
    manifest: dict[str, Any] = {
        "format": "wedit",
        "format_version": "1.0",
        "level": "flat",
        "provenance": {
            "ai_label": "AI-generated-and-edited",
            "notice": "AI-generated scene, edited. Not a recording.",
            "tool": "worldreel",
            "tool_version": "0.1.0",
            "created_at": FIXED_TIME,
            "source": {"kind": "synthetic"},
            "identity_policy": "synthetic-only",
            "rights": {"upload_rights": "synthetic", "takedown": {"status": "none"}},
        },
        "source_wreel": {"sha256": "a" * 64, "title": "Courtyard"},
        "camera_path": "director",
        "video": {
            "path": "video/main.mov",
            "fps": 24,
            "width": 1280,
            "height": 720,
            "duration": 12.0,
        },
        "stems": [
            {
                "id": "dlg_C01",
                "layer": "dialogue",
                "character_id": "C01",
                "path": "stems/dlg_C01.wav",
                "sample_rate": 48000,
                "channels": 1,
                "license": "synthetic",
            },
            {
                "id": "music",
                "layer": "music",
                "path": "stems/music.wav",
                "sample_rate": 48000,
                "channels": 2,
                "license": "CC0-1.0",
            },
        ],
        "render_layers": [
            {"id": "depth", "kind": "depth", "path": "layers/depth/", "pattern": "%06d.exr"}
        ],
        "timeline": [{"format": "otio", "path": "timeline/main.otio"}],
        "changes": [],
        "files": {},
    }
    entries = {
        "video/main.mov": b"placeholder video",
        "stems/dlg_C01.wav": b"RIFF....",
        "stems/music.wav": b"RIFF....",
        "layers/depth/000001.exr": b"exr",
        "layers/depth/000002.exr": b"exr",
        "timeline/main.otio": b"{}",
    }
    return manifest, entries


@pytest.mark.parametrize("as_zip", [False, True])
def test_wedit_round_trip(tmp_path: Path, key: Ed25519PrivateKey, as_zip: bool) -> None:
    manifest, entries = make_bundle()
    path = tmp_path / "cut.wedit"
    write_wedit(path, manifest, entries, key, as_zip=as_zip)
    doc = read_wedit(path)
    assert doc.report.ok, doc.report.issues
    assert doc.signature_status == "valid"
    assert {k: v for k, v in doc.entries.items() if k != PROVENANCE_TXT} == entries
    assert b"AI-generated-and-edited" in doc.entries[PROVENANCE_TXT]


def test_directory_form_is_plain_files(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    manifest, entries = make_bundle()
    path = tmp_path / "cut.wedit"
    write_wedit(path, manifest, entries, key)
    assert (path / "stems" / "music.wav").read_bytes() == b"RIFF...."
    assert (path / PROVENANCE_TXT).exists()


def test_missing_stem_refused(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    manifest, entries = make_bundle()
    del entries["stems/music.wav"]
    with pytest.raises(FormatError, match="missing_file"):
        write_wedit(tmp_path / "x.wedit", manifest, entries, key)


def test_provenance_required(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    manifest, entries = make_bundle()
    manifest["provenance"]["ai_label"] = "none"
    with pytest.raises(FormatError):
        write_wedit(tmp_path / "x.wedit", manifest, entries, key)


def test_edit_in_folder_detected(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    manifest, entries = make_bundle()
    path = tmp_path / "cut.wedit"
    write_wedit(path, manifest, entries, key)
    (path / "stems" / "music.wav").write_bytes(b"changed")
    assert "hash_mismatch" in validate_wedit(path).codes()


def test_refuses_to_write_into_non_empty_directory(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    manifest, entries = make_bundle()
    path = tmp_path / "cut.wedit"
    path.mkdir()
    (path / "keep.txt").write_text("user file")
    with pytest.raises(FormatError, match="not empty"):
        write_wedit(path, manifest, entries, key)
    assert (path / "keep.txt").read_text() == "user file"
