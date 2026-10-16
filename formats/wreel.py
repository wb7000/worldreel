"""Read, write and validate `.wreel` interactive-world files (spec/wreel.md)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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

FORMAT = "wreel"
SUPPORTED_VERSION = (1, 0)
FORMAT_VERSION = "1.0"
MANIFEST = "manifest.json"
SIGNATURE = "signature.json"
SCHEMA = "wreel-manifest-1.schema.json"

PROVENANCE_CLASSES = ("observed", "inferred", "guessed", "retrieved", "generated")
POINT_DTYPE = np.dtype(
    [("position", "<f4", (3,)), ("rgb", "u1", (3,)), ("cls", "u1"), ("conf", "<f4")]
)
POINT_STRIDE = 20
assert POINT_DTYPE.itemsize == POINT_STRIDE

QUAT_TOLERANCE = 1e-3


# --------------------------------------------------------------------------- point chunks


@dataclass
class PointCloud:
    """Columns of a `.pts` chunk. `cls` holds provenance class codes (spec §10.1)."""

    positions: npt.NDArray[np.float32]
    colors: npt.NDArray[np.uint8]
    cls: npt.NDArray[np.uint8]
    confidence: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        n = len(self.positions)
        shapes = (
            self.positions.shape == (n, 3),
            self.colors.shape == (n, 3),
            self.cls.shape == (n,),
            self.confidence.shape == (n,),
        )
        if not all(shapes):
            raise ValueError("point columns must have shapes (N,3), (N,3), (N,), (N,)")

    def __len__(self) -> int:
        return len(self.positions)

    def pack(self) -> bytes:
        records = np.empty(len(self), dtype=POINT_DTYPE)
        records["position"] = self.positions
        records["rgb"] = self.colors
        records["cls"] = self.cls
        records["conf"] = self.confidence
        return records.tobytes()

    @classmethod
    def unpack(cls, data: bytes) -> PointCloud:
        if len(data) % POINT_STRIDE:
            raise FormatError(f"point chunk size {len(data)} is not a multiple of {POINT_STRIDE}")
        records = np.frombuffer(data, dtype=POINT_DTYPE)
        return cls(
            positions=records["position"].copy(),
            colors=records["rgb"].copy(),
            cls=records["cls"].copy(),
            confidence=records["conf"].copy(),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointCloud):
            return NotImplemented
        return self.pack() == other.pack()


# --------------------------------------------------------------------------- document


@dataclass
class Wreel:
    """An in-memory `.wreel`: the manifest (including `files`) and every data entry."""

    manifest: dict[str, Any]
    entries: dict[str, bytes]
    signature_status: SignatureStatus = "missing"
    report: Report = field(default_factory=Report)

    def points(self, path: str) -> PointCloud:
        return PointCloud.unpack(self.entries[path])

    def track(self, path: str) -> dict[str, Any]:
        data: dict[str, Any] = load_json(self.entries[path], path)
        return data


def build_manifest_bytes(manifest: Mapping[str, Any], entries: Mapping[str, bytes]) -> bytes:
    """Fill in the `files` table and encode the manifest exactly as it will be signed."""
    media_types = {
        path: info["media_type"] for path, info in dict(manifest.get("files", {})).items()
    }
    full = dict(manifest)
    full["files"] = files_table(entries, media_types)
    return dump_json(full)


def write_wreel(
    path: Path,
    manifest: Mapping[str, Any],
    entries: Mapping[str, bytes],
    key: Ed25519PrivateKey,
    *,
    validate: bool = True,
) -> bytes:
    """Write a signed, deterministic `.wreel`. Returns the manifest bytes that were signed.
    By default refuses to write a file that would not validate."""
    manifest_bytes = build_manifest_bytes(manifest, entries)
    all_entries = dict(entries)
    all_entries[MANIFEST] = manifest_bytes
    all_entries[SIGNATURE] = make_signature(MANIFEST, manifest_bytes, key)
    if validate:
        report = Report()
        _validate_entries(all_entries, report)
        if not report.ok:
            raise FormatError(
                "refusing to write invalid .wreel:\n" + "\n".join(map(str, report.errors))
            )
    write_container(path, all_entries, [MANIFEST, SIGNATURE], as_zip=True)
    return manifest_bytes


def read_wreel(path: Path) -> Wreel:
    """Read and validate. Raises FormatError if the file is unreadable; otherwise the returned
    object's `report` lists every problem (check `report.ok`)."""
    all_entries = read_container(path)
    report = Report()
    status = _validate_entries(all_entries, report)
    manifest = load_json(all_entries[MANIFEST], MANIFEST)
    data = {k: v for k, v in all_entries.items() if k not in (MANIFEST, SIGNATURE)}
    return Wreel(manifest=manifest, entries=data, signature_status=status, report=report)


def validate_wreel(path: Path) -> Report:
    try:
        return read_wreel(path).report
    except FormatError as exc:
        report = Report()
        report.error("unreadable", str(exc))
        return report


# --------------------------------------------------------------------------- validation


def _validate_entries(all_entries: Mapping[str, bytes], rep: Report) -> SignatureStatus:
    """Validate a full set of container entries into `rep`; returns the signature status."""
    if MANIFEST not in all_entries:
        raise FormatError("manifest.json is missing")
    manifest_bytes = all_entries[MANIFEST]
    status = verify_signature(MANIFEST, manifest_bytes, all_entries.get(SIGNATURE), rep)
    manifest = load_json(manifest_bytes, MANIFEST)
    if not isinstance(manifest, dict):
        raise FormatError("manifest.json must be a JSON object")
    version = check_version(manifest, FORMAT, SUPPORTED_VERSION, rep)
    schema_check(manifest, SCHEMA, rep, lenient_unknown=version[1] > SUPPORTED_VERSION[1])
    data = {k: v for k, v in all_entries.items() if k not in (MANIFEST, SIGNATURE)}
    if isinstance(manifest.get("files"), dict):
        check_files_table(manifest["files"], data, rep)
    if rep.ok:
        # Semantic checks assume the schema holds.
        _semantic_checks(manifest, data, rep)
    return status


def _semantic_checks(m: dict[str, Any], data: Mapping[str, bytes], rep: Report) -> None:
    actor_ids = check_unique_ids(m["actors"], "actors", rep)
    camera_ids = check_unique_ids(m["cameras"], "cameras", rep)
    pov_ids = check_unique_ids(m["pov_presets"], "pov_presets", rep)
    check_unique_ids(m["regions"], "regions", rep)
    check_unique_ids(m["scene"]["layers"], "scene.layers", rep)
    check_unique_ids(m["scene"]["lighting"]["lights"], "scene.lighting.lights", rep)
    check_unique_ids(m["audio"]["tracks"], "audio.tracks", rep)
    check_unique_ids(m["interaction"]["rules"], "interaction.rules", rep)

    def ref(value: str | None, known: set[str], what: str, where: str) -> None:
        if value is not None and value not in known:
            rep.error("bad_reference", f"unknown {what} {value!r}", where)

    def need_file(path: str, where: str) -> None:
        if path not in data:
            rep.error("missing_file", f"referenced file {path!r} is not in the container", where)

    for layer in m["scene"]["layers"]:
        _check_layer(layer, data, rep, f"scene.layers.{layer['id']}")

    bounds = m["scene"]["bounds"]
    if any(lo > hi for lo, hi in zip(bounds["min"], bounds["max"], strict=True)):
        rep.error("bad_bounds", "scene bounds min > max", "scene.bounds")

    for actor in m["actors"]:
        where = f"actors.{actor['id']}"
        _check_layer(actor["geometry"], data, rep, where + ".geometry")
        need_file(actor["track"]["path"], where + ".track")
        if actor["track"]["path"] in data:
            _check_track(data[actor["track"]["path"]], rep, where + ".track")

    directors = [c for c in m["cameras"] if c["kind"] == "director"]
    if len(directors) > 1:
        rep.error("multiple_director_cameras", "at most one camera may have kind 'director'")
    for cam in m["cameras"]:
        times = [k["t"] for k in cam["keyframes"]]
        _check_times(times, rep, f"cameras.{cam['id']}")
        for k in cam["keyframes"]:
            _check_quat(k["rotation"], rep, f"cameras.{cam['id']}")

    for pov in m["pov_presets"]:
        where = f"pov_presets.{pov['id']}"
        ref(pov.get("camera_id"), camera_ids, "camera", where)
        ref(pov.get("actor_id"), actor_ids, "actor", where)

    for region in m["regions"]:
        b = region["bounds"]
        if any(lo > hi for lo, hi in zip(b["min"], b["max"], strict=True)):
            rep.error("bad_bounds", "region bounds min > max", f"regions.{region['id']}")

    audio = m["audio"]
    for track in audio["tracks"]:
        need_file(track["path"], f"audio.tracks.{track['id']}")
        ref(track.get("actor_id"), actor_ids, "actor", f"audio.tracks.{track['id']}")
    for i, event in enumerate(audio["events"]):
        ref(event.get("actor_id"), actor_ids, "actor", f"audio.events[{i}]")
    for sub in audio.get("subtitles", []):
        need_file(sub["path"], "audio.subtitles")
    _check_times([e["t"] for e in audio["mood"]], rep, "audio.mood", strict=False)

    for rule in m["interaction"]["rules"]:
        where = f"interaction.rules.{rule['id']}"
        ref(rule["trigger"].get("actor_id"), actor_ids, "actor", where)
        ref(rule["action"].get("preset_id"), pov_ids, "POV preset", where)

    people = [a for a in m["actors"] if a["type"] == "person"]
    if people and m["provenance"]["identity_policy"] == "none":
        rep.error(
            "identity_policy",
            "scene contains people but provenance.identity_policy is 'none'",
            "provenance",
        )


def _check_layer(
    layer: Mapping[str, Any], data: Mapping[str, bytes], rep: Report, where: str
) -> None:
    levels = [lod["level"] for lod in layer["lods"]]
    if levels != list(range(len(levels))):
        rep.error("bad_lod_levels", f"LOD levels must be 0..n in order, got {levels}", where)
    for lod in layer["lods"]:
        path = lod["path"]
        if path not in data:
            rep.error("missing_file", f"referenced file {path!r} is not in the container", where)
            continue
        expected = lod["count"] * POINT_STRIDE
        if len(data[path]) != expected:
            rep.error(
                "bad_point_chunk",
                f"{path}: {len(data[path])} bytes, expected count*{POINT_STRIDE} = {expected}",
                where,
            )
            continue
        cloud = PointCloud.unpack(data[path])
        if len(cloud) and int(cloud.cls.max()) >= len(PROVENANCE_CLASSES):
            rep.error("bad_point_class", f"{path}: provenance class code out of range", where)
        conf = cloud.confidence
        if len(cloud) and (not np.all(np.isfinite(conf)) or conf.min() < 0 or conf.max() > 1):
            rep.error("bad_confidence", f"{path}: confidence outside [0, 1]", where)
        if len(cloud) and not np.all(np.isfinite(cloud.positions)):
            rep.error("bad_positions", f"{path}: non-finite positions", where)


def _check_track(raw: bytes, rep: Report, where: str) -> None:
    try:
        track = load_json(raw, where)
        times, positions, rotations = track["times"], track["positions"], track["rotations"]
    except (FormatError, KeyError, TypeError) as exc:
        rep.error("bad_track", f"track is not a valid track object: {exc}", where)
        return
    if not (len(times) == len(positions) == len(rotations)) or not times:
        rep.error(
            "bad_track", "times/positions/rotations must be non-empty and equal length", where
        )
        return
    _check_times(times, rep, where)
    for p in positions:
        if len(p) != 3:
            rep.error("bad_track", "positions must be [x, y, z]", where)
            return
    for q in rotations:
        _check_quat(q, rep, where)


def _check_times(times: list[float], rep: Report, where: str, *, strict: bool = True) -> None:
    pairs = itertools.pairwise(times)
    if any((b <= a) if strict else (b < a) for a, b in pairs):
        rule = "strictly increasing" if strict else "non-decreasing"
        rep.error("bad_times", f"times must be {rule}", where)


def _check_quat(q: list[float], rep: Report, where: str) -> None:
    if len(q) != 4 or abs(math.sqrt(sum(c * c for c in q)) - 1.0) > QUAT_TOLERANCE:
        rep.error("bad_quaternion", f"rotation {q} is not a unit quaternion [x, y, z, w]", where)
