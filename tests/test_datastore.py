from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

import stages.ingest.store as store_mod
from stages.ingest.store import COLUMNS, DataStore, DataStoreError
from tests.media import HAS_FFMPEG, make_clip
from tools import datasets
from tools.data import main as data_cli

VideoMaker = Callable[[Path, str], Path]
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    s = DataStore(tmp_path / "data")
    s.init()
    return s


@pytest.fixture
def video(monkeypatch: pytest.MonkeyPatch) -> VideoMaker:
    """Returns a function that writes a distinct valid video into a folder."""
    counter = {"n": 0}

    def write(folder: Path, name: str) -> Path:
        counter["n"] += 1
        path = folder / name
        if HAS_FFMPEG:
            make_clip(path, seconds=0.5 + 0.1 * counter["n"], audio=False)
        else:
            path.write_bytes(FAKE_MP4 + bytes([counter["n"]]))
        return path

    if not HAS_FFMPEG:
        monkeypatch.setattr(store_mod, "find_tool", lambda name: None)
    return write


def info(path: Path, **fields: str) -> None:
    data = {"source": "own recording", "license": "CC-BY-4.0", "upload_rights": "owner", **fields}
    path.with_name(path.name + ".info.json").write_text(json.dumps(data))


def test_ingest_with_info(store: DataStore, video: VideoMaker) -> None:
    clip = video(store.inbox, "Holiday Clip.MP4")
    info(clip)
    report = store.ingest()
    assert len(report.added) == 1
    row = store.get(report.added[0])
    assert row.status == "active"
    assert row.path == f"raw/{row.id}.mp4"
    assert (store.root / row.path).exists()
    assert row.original_name == "Holiday Clip.MP4"
    assert (row.source, row.license, row.upload_rights) == ("own recording", "CC-BY-4.0", "owner")
    assert list(store.inbox.iterdir()) == []
    header = store.manifest_path.read_text().splitlines()[0].split(",")
    assert header == COLUMNS
    assert {"id", "hash", "source", "license", "date_added", "status", "notes"} <= set(header)


def test_missing_info_is_held_then_released(store: DataStore, video: VideoMaker) -> None:
    video(store.inbox, "clip.mp4")
    report = store.ingest()
    fid = report.needs_info[0]
    row = store.get(fid)
    assert row.status == "needs_info"
    assert "source and license" in row.notes
    assert (store.root / row.path).parent == store.needs_info_dir
    store.set_info(fid, source="filmed it myself", license="CC0-1.0", upload_rights="owner")
    row = store.get(fid)
    assert row.status == "active"
    assert (store.root / row.path).exists()
    assert list(store.needs_info_dir.iterdir()) == []


def test_info_added_later_by_sidecar(store: DataStore, video: VideoMaker) -> None:
    video(store.inbox, "clip.mp4")
    fid = store.ingest().needs_info[0]
    held = store.root / store.get(fid).path
    info(held)
    assert store.ingest().added == [fid]


def test_no_derivatives_license_is_held(store: DataStore, video: VideoMaker) -> None:
    clip = video(store.inbox, "clip.mp4")
    info(clip, license="CC-BY-ND-4.0")
    fid = store.ingest().needs_info[0]
    assert "derivatives" in store.get(fid).notes


def test_duplicates_and_bad_files_are_rejected(store: DataStore, video: VideoMaker) -> None:
    clip = video(store.inbox, "a.mp4")
    info(clip)
    store.ingest()
    copy = store.inbox / "copy.mp4"
    copy.write_bytes((store.root / store.get(store.rows()[0].id).path).read_bytes())
    (store.inbox / "broken.mp4").write_bytes(b"garbage")
    (store.inbox / "notes.txt").write_text("hello")
    (store.inbox / "empty.mov").write_bytes(b"")
    report = store.ingest()
    assert report.duplicates == ["copy.mp4"]
    assert sorted(report.rejected) == ["broken.mp4", "empty.mov", "notes.txt"]
    reasons = {p.name.split("_", 1)[1]: p.read_text() for p in store.rejected.glob("*.reason.txt")}
    assert "duplicate of" in reasons["copy.mp4.reason.txt"]
    assert len(store.rows()) == 1


def test_exclude_keeps_file_for_old_versions(store: DataStore, video: VideoMaker) -> None:
    for name in ("a.mp4", "b.mp4"):
        info(video(store.inbox, name))
    store.ingest()
    datasets.freeze(store, "v1")
    bad = store.rows()[0].id
    store.exclude(bad, "blurry")
    assert store.get(bad).status == "excluded"
    report = datasets.verify(store, "v1")
    assert report.ok
    assert any("excluded" in w for w in report.warnings)
    v2 = datasets.freeze(store, "v2")
    assert bad not in {f["id"] for f in v2["files"]}


def test_freeze_and_verify(store: DataStore, video: VideoMaker) -> None:
    for name in ("a.mp4", "b.mp4"):
        info(video(store.inbox, name))
    store.ingest()
    version = datasets.freeze(store, "train-2026-10", "first set")
    assert version["count"] == 2
    assert datasets.verify(store, "train-2026-10").ok
    with pytest.raises(DataStoreError, match="immutable"):
        datasets.freeze(store, "train-2026-10")
    with pytest.raises(DataStoreError):
        datasets.freeze(store, "../escape")
    row = store.rows()[0]
    (store.root / row.path).write_bytes(b"changed")
    report = datasets.verify(store, "train-2026-10")
    assert not report.ok
    assert "hash mismatch" in report.errors[0]


def test_remove_records_retrain_need(store: DataStore, video: VideoMaker) -> None:
    for name in ("a.mp4", "b.mp4"):
        info(video(store.inbox, name))
    store.ingest()
    datasets.freeze(store, "v1")
    victim = store.rows()[0]
    used = datasets.versions_using(store, victim.id)
    assert used == ["v1"]
    assert store.remove(victim.id, "takedown request", used) is True
    assert not (store.root / victim.path).exists()
    assert victim.id not in {r.id for r in store.rows()}
    assert victim.id in store.removed_ids()
    removals = store.removals_path.read_text()
    assert "takedown request" in removals and ",yes" in removals
    report = datasets.verify(store, "v1")
    assert not report.ok
    assert "can no longer be reproduced" in report.errors[0]


def test_freeze_with_nothing_active(store: DataStore) -> None:
    with pytest.raises(DataStoreError, match="no active files"):
        datasets.freeze(store, "empty")


def test_cli_remove_requires_confirmation(
    store: DataStore, video: VideoMaker, capsys: pytest.CaptureFixture[str]
) -> None:
    info(video(store.inbox, "a.mp4"))
    assert data_cli(["--root", str(store.root), "ingest"]) == 0
    fid = store.rows()[0].id
    assert data_cli(["--root", str(store.root), "remove", fid, "--reason", "x"]) == 2
    assert store.get(fid).status == "active"
    assert data_cli(["--root", str(store.root), "remove", fid, "--reason", "x", "--yes"]) == 0
    assert "removed" in capsys.readouterr().out
