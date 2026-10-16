"""The data store: drop-folder ingest and the manifest (docs/DATA.md).

    data/
      inbox/          drop files here, each with a `<file>.info.json` sidecar
      needs_info/     files waiting for a source + license note
      raw/            originals after ingestion, never edited
      processed/      stage outputs, one folder per file id
      rejected/       duplicates and unreadable files (with a reason file)
      manifest.csv    one row per file
      removals.csv    tombstones for removed files (retrain bookkeeping)
      versions/       frozen dataset versions

Sidecar `<file>.info.json`: {"source": "...", "license": "...", "upload_rights": "owner",
"notes": "..."}. `source` and `license` are required.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

from pipeline.cache import file_sha256
from pipeline.stage import Item
from tools.toolpaths import find_tool

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
INFO_SUFFIX = ".info.json"
UPLOAD_RIGHTS = {"unknown", "owner", "licensed", "public-domain", "synthetic"}
STATUSES = {"needs_info", "active", "excluded"}

# Reconstruction makes a derivative work; a No-Derivatives licence does not allow that.
_NO_DERIVATIVES = ("-nd", "noderivatives", "no-derivatives", "noderivs")

_MAGIC = {
    b"\x1aE\xdf\xa3": "matroska/webm",
    b"RIFF": "avi/webp",
    b"\x89PNG": "png",
    b"\xff\xd8\xff": "jpeg",
}


@dataclass
class Row:
    id: str
    hash: str
    source: str
    license: str
    date_added: str
    status: str
    notes: str
    original_name: str
    path: str
    bytes: int
    media_type: str
    upload_rights: str = "unknown"
    takedown_status: str = "none"

    def item(self, root: Path) -> Item:
        return Item(
            id=self.id,
            sha256=self.hash,
            path=root / self.path,
            source=self.source,
            license=self.license,
            upload_rights=self.upload_rights,
        )


COLUMNS = [f.name for f in fields(Row)]


@dataclass
class IngestReport:
    added: list[str]
    needs_info: list[str]
    duplicates: list[str]
    rejected: list[str]

    def summary(self) -> str:
        return (
            f"added {len(self.added)}, needs info {len(self.needs_info)}, "
            f"duplicates {len(self.duplicates)}, rejected {len(self.rejected)}"
        )


class DataStoreError(Exception):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def media_type(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT:
        return "image"
    return None


def check_integrity(path: Path) -> str | None:
    """None if the file looks like a readable video/image, otherwise the reason it is not."""
    if path.stat().st_size == 0:
        return "empty file"
    ffprobe = find_tool("ffprobe")
    if ffprobe is not None:
        proc = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return f"ffprobe cannot read it: {proc.stderr.strip()[-200:] or 'no streams'}"
        return None
    head = path.read_bytes()[:12]
    if head[4:8] == b"ftyp" or any(head.startswith(m) for m in _MAGIC):
        return None
    return "unrecognised file header (install ffmpeg for a proper check)"


def read_info(path: Path) -> tuple[dict[str, str] | None, str]:
    """Read the sidecar. Returns (info, problem); info is None when it is missing or unusable."""
    sidecar = path.with_name(path.name + INFO_SUFFIX)
    if not sidecar.exists():
        return None, f"missing {sidecar.name} with source and license"
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"{sidecar.name} is not valid JSON: {exc}"
    info = {k: str(v).strip() for k, v in raw.items() if v is not None}
    missing = [k for k in ("source", "license") if not info.get(k)]
    if missing:
        return None, f"{sidecar.name} is missing {', '.join(missing)}"
    if any(tag in info["license"].lower().replace(" ", "") for tag in _NO_DERIVATIVES):
        return (
            None,
            f"license {info['license']!r} forbids derivatives, which reconstruction creates",
        )
    rights = info.get("upload_rights", "unknown")
    if rights not in UPLOAD_RIGHTS:
        return None, f"upload_rights must be one of {sorted(UPLOAD_RIGHTS)}"
    return info, ""


class DataStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.inbox = root / "inbox"
        self.needs_info_dir = root / "needs_info"
        self.raw = root / "raw"
        self.processed = root / "processed"
        self.rejected = root / "rejected"
        self.versions = root / "versions"
        self.manifest_path = root / "manifest.csv"
        self.removals_path = root / "removals.csv"

    def init(self) -> None:
        for d in (
            self.inbox,
            self.needs_info_dir,
            self.raw,
            self.processed,
            self.rejected,
            self.versions,
        ):
            d.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._write_rows([])

    # ------------------------------------------------------------------ manifest

    def rows(self) -> list[Row]:
        if not self.manifest_path.exists():
            return []
        with self.manifest_path.open(newline="", encoding="utf-8") as fh:
            return [
                Row(**{**r, "bytes": int(r["bytes"])})  # type: ignore[arg-type]
                for r in csv.DictReader(fh)
            ]

    def _write_rows(self, rows: list[Row]) -> None:
        tmp = self.manifest_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in sorted(rows, key=lambda r: (r.date_added, r.id)):
                writer.writerow(asdict(row))
        os.replace(tmp, self.manifest_path)

    def get(self, file_id: str) -> Row:
        for row in self.rows():
            if row.id == file_id:
                return row
        raise DataStoreError(f"no file with id {file_id!r}")

    def active(self) -> Iterator[Row]:
        return (r for r in self.rows() if r.status == "active")

    # ------------------------------------------------------------------ ingest

    def ingest(self) -> IngestReport:
        """Process everything in inbox/ and retry everything in needs_info/."""
        self.init()
        report = IngestReport([], [], [], [])
        rows = self.rows()
        by_hash = {r.hash: r for r in rows}

        for path in sorted(self.needs_info_dir.iterdir()):
            if path.name.endswith(INFO_SUFFIX) or not path.is_file():
                continue
            row = next(
                (
                    r
                    for r in rows
                    if r.status == "needs_info" and r.path == f"needs_info/{path.name}"
                ),
                None,
            )
            info, problem = read_info(path)
            if row is None:
                continue
            if info is None:
                row.notes = problem
                continue
            self._activate(row, path, info)
            report.added.append(row.id)

        for path in sorted(self.inbox.iterdir()):
            if not path.is_file() or path.name.endswith(INFO_SUFFIX) or path.name.startswith("."):
                continue
            kind = media_type(path)
            if kind is None:
                self._reject(path, f"unsupported file type {path.suffix!r}")
                report.rejected.append(path.name)
                continue
            unreadable = check_integrity(path)
            if unreadable:
                self._reject(path, unreadable)
                report.rejected.append(path.name)
                continue
            digest = file_sha256(path)
            if digest in by_hash:
                self._reject(path, f"duplicate of {by_hash[digest].id}")
                report.duplicates.append(path.name)
                continue
            file_id = f"f{digest[:12]}"
            row = Row(
                id=file_id,
                hash=digest,
                source="",
                license="",
                date_added=now(),
                status="needs_info",
                notes="",
                original_name=path.name,
                path="",
                bytes=path.stat().st_size,
                media_type=kind,
            )
            info, problem = read_info(path)
            if info is None:
                target = self.needs_info_dir / f"{file_id}{path.suffix.lower()}"
                self._move(path, target)
                row.path = f"needs_info/{target.name}"
                row.notes = problem
                report.needs_info.append(file_id)
            else:
                self._activate(row, path, info)
                report.added.append(file_id)
            rows.append(row)
            by_hash[digest] = row
        self._write_rows(rows)
        return report

    def _activate(self, row: Row, path: Path, info: dict[str, str]) -> None:
        target = self.raw / f"{row.id}{path.suffix.lower()}"
        sidecar = path.with_name(path.name + INFO_SUFFIX)
        self._move(path, target)
        sidecar.unlink(missing_ok=True)
        row.path = f"raw/{target.name}"
        row.source = info["source"]
        row.license = info["license"]
        row.upload_rights = info.get("upload_rights", "unknown")
        row.notes = info.get("notes", "")
        row.status = "active"

    def _move(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise DataStoreError(f"refusing to overwrite {dst}")
        shutil.move(str(src), str(dst))
        sidecar = src.with_name(src.name + INFO_SUFFIX)
        if sidecar.exists() and dst.parent == self.needs_info_dir:
            shutil.move(str(sidecar), str(dst.with_name(dst.name + INFO_SUFFIX)))

    def _reject(self, path: Path, reason: str) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        target = self.rejected / f"{stamp}_{path.name}"
        self._move(path, target)
        sidecar = path.with_name(path.name + INFO_SUFFIX)
        if sidecar.exists():
            shutil.move(str(sidecar), str(target.with_name(target.name + INFO_SUFFIX)))
        target.with_name(target.name + ".reason.txt").write_text(reason + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ changes

    def set_info(
        self,
        file_id: str,
        source: str,
        license: str,
        upload_rights: str = "unknown",
        notes: str = "",
    ) -> None:
        """Provide the missing note for a file in needs_info/ and ingest it."""
        row = self.get(file_id)
        if row.status != "needs_info":
            raise DataStoreError(f"{file_id} is {row.status}, not needs_info")
        path = self.root / row.path
        path.with_name(path.name + INFO_SUFFIX).write_text(
            json.dumps(
                {
                    "source": source,
                    "license": license,
                    "upload_rights": upload_rights,
                    "notes": notes,
                }
            ),
            encoding="utf-8",
        )
        self.ingest()
        if self.get(file_id).status != "active":
            raise DataStoreError(f"{file_id} still needs info: {self.get(file_id).notes}")

    def exclude(self, file_id: str, reason: str) -> None:
        """Mark a bad file as excluded. The file stays in raw/ so old versions still reproduce."""
        rows = self.rows()
        for row in rows:
            if row.id == file_id:
                row.status = "excluded"
                row.notes = f"excluded {now()}: {reason}"
                self._write_rows(rows)
                return
        raise DataStoreError(f"no file with id {file_id!r}")

    def remove(self, file_id: str, reason: str, used_in_versions: list[str]) -> bool:
        """Truly delete a file (wrong licence, takedown). Returns True if a retrain is needed."""
        rows = self.rows()
        row = next((r for r in rows if r.id == file_id), None)
        if row is None:
            raise DataStoreError(f"no file with id {file_id!r}")
        (self.root / row.path).unlink(missing_ok=True)
        shutil.rmtree(self.processed / file_id, ignore_errors=True)
        self._write_rows([r for r in rows if r.id != file_id])
        retrain = bool(used_in_versions)
        new_file = not self.removals_path.exists()
        with self.removals_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            if new_file:
                writer.writerow(
                    ["id", "hash", "removed_at", "reason", "used_in_versions", "retrain_needed"]
                )
            writer.writerow(
                [
                    file_id,
                    row.hash,
                    now(),
                    reason,
                    ";".join(used_in_versions),
                    "yes" if retrain else "no",
                ]
            )
        return retrain

    def removed_ids(self) -> set[str]:
        if not self.removals_path.exists():
            return set()
        with self.removals_path.open(newline="", encoding="utf-8") as fh:
            return {r["id"] for r in csv.DictReader(fh)}
