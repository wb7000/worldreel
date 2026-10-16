"""Frozen dataset versions: a named, immutable list of file ids + hashes (docs/DATA.md).

Freeze one before every training run; `verify` proves it can be reproduced exactly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pipeline.cache import file_sha256
from stages.ingest.store import DataStore, DataStoreError, now

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass
class VerifyReport:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def freeze(store: DataStore, name: str, description: str = "") -> dict[str, Any]:
    if not _NAME.match(name):
        raise DataStoreError("version names use letters, digits, '.', '_' and '-' (max 64)")
    store.init()
    path = store.versions / f"{name}.json"
    if path.exists():
        raise DataStoreError(f"version {name!r} already exists; versions are immutable")
    files = [
        {
            "id": r.id,
            "hash": r.hash,
            "license": r.license,
            "source": r.source,
            "media_type": r.media_type,
        }
        for r in sorted(store.active(), key=lambda r: r.id)
    ]
    if not files:
        raise DataStoreError("no active files to freeze")
    version = {
        "name": name,
        "created_at": now(),
        "description": description,
        "selection": "all files with status=active",
        "count": len(files),
        "files": files,
    }
    path.write_text(json.dumps(version, indent=2) + "\n", encoding="utf-8")
    return version


def load(store: DataStore, name: str) -> dict[str, Any]:
    path = store.versions / f"{name}.json"
    if not path.exists():
        raise DataStoreError(f"no version named {name!r}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def list_versions(store: DataStore) -> list[str]:
    return sorted(p.stem for p in store.versions.glob("*.json")) if store.versions.exists() else []


def versions_using(store: DataStore, file_id: str) -> list[str]:
    return [
        n for n in list_versions(store) if any(f["id"] == file_id for f in load(store, n)["files"])
    ]


def verify(store: DataStore, name: str) -> VerifyReport:
    """Every file of the version is present with the recorded hash."""
    version = load(store, name)
    report = VerifyReport(name)
    rows = {r.id: r for r in store.rows()}
    removed = store.removed_ids()
    for entry in version["files"]:
        fid = entry["id"]
        if fid in removed:
            report.errors.append(f"{fid} was removed; this version can no longer be reproduced")
            continue
        row = rows.get(fid)
        if row is None:
            report.errors.append(f"{fid} is not in the manifest")
            continue
        if row.status == "excluded":
            report.warnings.append(f"{fid} has since been excluded ({row.notes})")
        path = store.root / row.path
        if not path.exists():
            report.errors.append(f"{fid}: file missing at {row.path}")
        elif file_sha256(path) != entry["hash"]:
            report.errors.append(f"{fid}: file content changed (hash mismatch)")
    return report
