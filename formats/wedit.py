"""Read, write and validate `.wedit` edit bundles (spec/wedit.md).

Phase 1 covers the container and manifest. Rendering the video, stems and layers is Phase 6.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from formats.common import (
    FormatError,
    Report,
    SignatureStatus,
    check_files_table,
    check_unique_ids,
    check_version,
    dump_json,
    files_table,
    load_json,
    make_signature,
    read_container,
    schema_check,
    verify_signature,
    write_container,
)

FORMAT = "wedit"
SUPPORTED_VERSION = (1, 1)
FORMAT_VERSION = "1.1"  # 1.1: timeline format "mlt" (D-6.1)
MANIFEST = "wedit.json"
SIGNATURE = "signature.json"
PROVENANCE_TXT = "PROVENANCE.txt"
SCHEMA = "wedit-manifest-1.schema.json"


@dataclass
class Wedit:
    manifest: dict[str, Any]
    entries: dict[str, bytes]
    signature_status: SignatureStatus = "missing"
    report: Report = field(default_factory=Report)


def provenance_text(provenance: Mapping[str, Any]) -> bytes:
    """Plain-text copy of the AI label for people who open the folder directly."""
    lines = [
        provenance["ai_label"],
        provenance["notice"],
        f"Tool: {provenance['tool']} {provenance['tool_version']}",
        f"Created: {provenance['created_at']}",
        "The edit bundle was derived from a WorldReel scene. This label must stay with the files.",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_wedit(
    path: Path,
    manifest: Mapping[str, Any],
    entries: Mapping[str, bytes],
    key: Ed25519PrivateKey,
    *,
    as_zip: bool = False,
    validate: bool = True,
) -> bytes:
    """Write a signed bundle (directory by default). PROVENANCE.txt is always generated."""
    data = dict(entries)
    data[PROVENANCE_TXT] = provenance_text(manifest["provenance"])
    media_types = {p: i["media_type"] for p, i in dict(manifest.get("files", {})).items()}
    full = dict(manifest)
    full["files"] = files_table(data, media_types)
    manifest_bytes = dump_json(full)
    data[MANIFEST] = manifest_bytes
    data[SIGNATURE] = make_signature(MANIFEST, manifest_bytes, key)
    if validate:
        report = Report()
        _validate_entries(data, report)
        if not report.ok:
            raise FormatError(
                "refusing to write invalid .wedit:\n" + "\n".join(map(str, report.errors))
            )
    write_container(path, data, [MANIFEST, SIGNATURE], as_zip=as_zip)
    return manifest_bytes


def read_wedit(path: Path) -> Wedit:
    all_entries = read_container(path)
    report = Report()
    status = _validate_entries(all_entries, report)
    manifest = load_json(all_entries[MANIFEST], MANIFEST)
    data = {k: v for k, v in all_entries.items() if k not in (MANIFEST, SIGNATURE)}
    return Wedit(manifest=manifest, entries=data, signature_status=status, report=report)


def validate_wedit(path: Path) -> Report:
    try:
        return read_wedit(path).report
    except FormatError as exc:
        report = Report()
        report.error("unreadable", str(exc))
        return report


def _validate_entries(all_entries: Mapping[str, bytes], rep: Report) -> SignatureStatus:
    if MANIFEST not in all_entries:
        raise FormatError("wedit.json is missing")
    manifest_bytes = all_entries[MANIFEST]
    status = verify_signature(MANIFEST, manifest_bytes, all_entries.get(SIGNATURE), rep)
    manifest = load_json(manifest_bytes, MANIFEST)
    if not isinstance(manifest, dict):
        raise FormatError("wedit.json must be a JSON object")
    version = check_version(manifest, FORMAT, SUPPORTED_VERSION, rep)
    schema_check(manifest, SCHEMA, rep, lenient_unknown=version[1] > SUPPORTED_VERSION[1])
    data = {k: v for k, v in all_entries.items() if k not in (MANIFEST, SIGNATURE)}
    if isinstance(manifest.get("files"), dict):
        check_files_table(manifest["files"], data, rep)
    if not rep.ok:
        return status

    if PROVENANCE_TXT not in data:
        rep.warning("provenance_txt_missing", "PROVENANCE.txt is missing")
    if manifest["level"] == "round-trip":
        rep.warning(
            "round_trip_unimplemented", "round-trip change records are not applied until Phase 9"
        )

    def need(path: str, where: str) -> None:
        if path not in data:
            rep.error("missing_file", f"referenced file {path!r} is not in the bundle", where)

    need(manifest["video"]["path"], "video")
    check_unique_ids(manifest["stems"], "stems", rep)
    for stem in manifest["stems"]:
        need(stem["path"], f"stems.{stem['id']}")
    check_unique_ids(manifest["render_layers"], "render_layers", rep)
    for layer in manifest["render_layers"]:
        if not any(p.startswith(layer["path"]) for p in data):
            rep.error(
                "empty_layer", f"no files under {layer['path']}", f"render_layers.{layer['id']}"
            )
    for timeline in manifest["timeline"]:
        need(timeline["path"], "timeline")
    if "script" in manifest:
        need(manifest["script"]["path"], "script")
    check_unique_ids(manifest["changes"], "changes", rep)
    return status
