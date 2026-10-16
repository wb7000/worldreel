"""Shared building blocks for WorldReel file formats: issues, versions, hashing, safe containers,
JSON-schema validation and Ed25519 signing."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "spec" / "schemas"
SCHEMA_BASE_URI = "https://github.com/wb7000/worldreel/spec/schemas/"

ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
MAX_ENTRY_BYTES = 4 * 1024**3
_SAFE_PATH = re.compile(r"^[A-Za-z0-9._/-]{1,255}$")


class FormatError(Exception):
    """A file cannot be read at all (not a container, unsafe paths, unsupported major version)."""


# --------------------------------------------------------------------------- issues


@dataclass(frozen=True)
class Issue:
    level: Literal["error", "warning"]
    code: str
    message: str
    where: str = ""

    def __str__(self) -> str:
        loc = f" [{self.where}]" if self.where else ""
        return f"{self.level.upper()} {self.code}{loc}: {self.message}"


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)

    def error(self, code: str, message: str, where: str = "") -> None:
        self.issues.append(Issue("error", code, message, where))

    def warning(self, code: str, message: str, where: str = "") -> None:
        self.issues.append(Issue("warning", code, message, where))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}


# --------------------------------------------------------------------------- versions


def parse_version(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)", text)
    if not match:
        raise FormatError(f"format_version {text!r} is not MAJOR.MINOR")
    return int(match.group(1)), int(match.group(2))


def check_version(
    manifest: Mapping[str, Any], fmt: str, supported: tuple[int, int], report: Report
) -> tuple[int, int]:
    """Check `format` and `format_version`. Raises FormatError for an unreadable major version."""
    if manifest.get("format") != fmt:
        raise FormatError(f"not a {fmt} manifest (format={manifest.get('format')!r})")
    version = parse_version(str(manifest.get("format_version", "")))
    if version[0] != supported[0]:
        raise FormatError(
            f"{fmt} major version {version[0]} is not supported (this reader: {supported[0]}.x)"
        )
    if version[1] > supported[1]:
        report.warning(
            "newer_minor_version",
            f"file is {fmt} {version[0]}.{version[1]}; reader knows {supported[0]}.{supported[1]}"
            "; unknown fields are ignored",
        )
    return version


# --------------------------------------------------------------------------- hashing / json


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_json(obj: Any) -> bytes:
    """Stable JSON encoding used by every writer (2-space indent, UTF-8, trailing newline)."""
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def load_json(data: bytes, what: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"{what} is not valid UTF-8 JSON: {exc}") from exc


def files_table(entries: Mapping[str, bytes], media_types: Mapping[str, str]) -> dict[str, Any]:
    return {
        path: {
            "sha256": sha256_hex(data),
            "bytes": len(data),
            "media_type": media_types.get(path, guess_media_type(path)),
        }
        for path, data in sorted(entries.items())
    }


def guess_media_type(path: str) -> str:
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "json": "application/json",
        "pts": "application/x-wreel-points",
        "txt": "text/plain",
        "wsl": "text/x-wsl",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "mov": "video/quicktime",
        "mp4": "video/mp4",
        "png": "image/png",
        "exr": "image/x-exr",
        "otio": "application/json",
        "vtt": "text/vtt",
    }.get(suffix, "application/octet-stream")


def check_files_table(
    table: Mapping[str, Any], entries: Mapping[str, bytes], report: Report
) -> None:
    """Every entry is listed with the right hash and size, and nothing listed is missing."""
    for path in sorted(set(entries) - set(table)):
        report.error("unlisted_entry", "entry is not listed in the files table", path)
    for path, info in sorted(table.items()):
        data = entries.get(path)
        if data is None:
            report.error("missing_entry", "listed file is missing from the container", path)
            continue
        if len(data) != info.get("bytes"):
            report.error(
                "size_mismatch", f"{len(data)} bytes, table says {info.get('bytes')}", path
            )
        if sha256_hex(data) != info.get("sha256"):
            report.error("hash_mismatch", "SHA-256 does not match the files table", path)


# --------------------------------------------------------------------------- schemas


@cache
def _registry() -> Registry[Any]:
    registry: Registry[Any] = Registry()
    for schema_file in sorted(SCHEMA_DIR.glob("*.schema.json")):
        contents = json.loads(schema_file.read_text(encoding="utf-8"))
        registry = registry.with_resource(contents["$id"], Resource.from_contents(contents))
    return registry


@cache
def _validator(schema_name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=_registry())


def schema_check(
    instance: Any, schema_name: str, report: Report, *, lenient_unknown: bool = False
) -> None:
    """Validate against a JSON schema. With `lenient_unknown`, unknown fields become warnings
    (used for files written by a newer minor version)."""
    for err in sorted(_validator(schema_name).iter_errors(instance), key=lambda e: e.json_path):
        where = err.json_path
        if lenient_unknown and err.validator == "additionalProperties":
            report.warning("unknown_field", err.message, where)
        else:
            report.error("schema", err.message, where)


# --------------------------------------------------------------------------- containers


def is_safe_path(path: str) -> bool:
    if not _SAFE_PATH.match(path) or path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        return False
    return ".." not in path.split("/") and "" not in path.split("/")


def read_container(path: Path) -> dict[str, bytes]:
    """Read a ZIP or directory container into memory. Rejects unsafe or oversized entries."""
    entries: dict[str, bytes] = {}
    if path.is_dir():
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = file.relative_to(path).as_posix()
            if not is_safe_path(rel):
                raise FormatError(f"unsafe entry path {rel!r}")
            entries[rel] = file.read_bytes()
        return entries
    if not zipfile.is_zipfile(path):
        raise FormatError(f"{path} is neither a ZIP container nor a directory")
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not is_safe_path(info.filename):
                raise FormatError(f"unsafe entry path {info.filename!r}")
            if info.flag_bits & 0x1:
                raise FormatError(f"encrypted entry {info.filename!r} is not allowed")
            if info.file_size > MAX_ENTRY_BYTES:
                raise FormatError(f"entry {info.filename!r} is too large")
            if info.filename in entries:
                raise FormatError(f"duplicate entry {info.filename!r}")
            entries[info.filename] = zf.read(info)
    return entries


def write_container(
    path: Path, entries: Mapping[str, bytes], first: Iterable[str], *, as_zip: bool
) -> None:
    """Write entries deterministically: `first` in order, then the rest sorted by path.
    A directory target must not exist or must be empty (writers never delete files)."""
    head = list(first)
    order = head + sorted(p for p in entries if p not in head)
    for name in order:
        if not is_safe_path(name):
            raise FormatError(f"unsafe entry path {name!r}")
    if as_zip:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with zipfile.ZipFile(tmp, "w") as zf:
            for name in order:
                info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, entries[name], compresslevel=6)
        os.replace(tmp, path)
        return
    if path.exists() and any(path.iterdir()):
        raise FormatError(f"target directory {path} is not empty")
    for name in order:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entries[name])


# --------------------------------------------------------------------------- signing


SignatureStatus = Literal["valid", "invalid", "missing"]


def generate_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def default_key_path() -> Path:
    env = os.environ.get("WORLDREEL_SIGNING_KEY")
    return Path(env) if env else Path.home() / ".worldreel" / "signing_key.pem"


def load_or_create_signing_key(path: Path | None = None) -> Ed25519PrivateKey:
    """Load the local signing key, creating one on first use. Never stored in the repository."""
    key_path = path or default_key_path()
    if key_path.exists():
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise FormatError(f"{key_path} is not an Ed25519 private key")
        return key
    key = generate_signing_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return key


def public_key_b64(key: Ed25519PrivateKey) -> str:
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def make_signature(signed_name: str, data: bytes, key: Ed25519PrivateKey) -> bytes:
    return dump_json(
        {
            "alg": "Ed25519",
            "public_key": public_key_b64(key),
            "signature": base64.b64encode(key.sign(data)).decode("ascii"),
            "signed": signed_name,
        }
    )


def verify_signature(
    signed_name: str, data: bytes, signature_json: bytes | None, report: Report
) -> SignatureStatus:
    if signature_json is None:
        report.error("signature_missing", "signature.json is missing")
        return "missing"
    try:
        sig = load_json(signature_json, "signature.json")
        if sig.get("alg") != "Ed25519" or sig.get("signed") != signed_name:
            raise ValueError("unexpected alg or signed target")
        public = Ed25519PublicKey.from_public_bytes(base64.b64decode(sig["public_key"]))
        public.verify(base64.b64decode(sig["signature"]), data)
    except (InvalidSignature, ValueError, KeyError, TypeError, FormatError) as exc:
        report.error(
            "signature_invalid", f"signature does not verify: {str(exc) or 'bad signature'}"
        )
        return "invalid"
    return "valid"


def check_unique_ids(items: Iterable[Mapping[str, Any]], what: str, report: Report) -> set[str]:
    seen: set[str] = set()
    for item in items:
        ident = item.get("id")
        if not isinstance(ident, str):
            continue
        if ident in seen:
            report.error("duplicate_id", f"duplicate {what} id {ident!r}", what)
        seen.add(ident)
    return seen
