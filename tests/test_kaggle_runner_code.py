"""The Kaggle runner must not push a job against stale code (D-9.26).

Kaggle jobs import repo modules from the mounted code dataset. A module that
was committed but never uploaded makes the job die with ModuleNotFoundError
*after* the queue wait and the multi-GB model download -- 4 minutes of GPU and
wall-clock time for nothing. `run` now uploads the code when the tracked files
changed since the last push, and skips the upload when they did not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.kaggle_runner as kr


def test_code_fingerprint_changes_with_the_tracked_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    a = root / "a.py"
    b = root / "b.py"
    a.write_text("x = 1\n", encoding="utf-8")
    b.write_text("y = 2\n", encoding="utf-8")

    monkeypatch.setattr(kr, "REPO_ROOT", root)
    monkeypatch.setattr(kr, "tracked_code_files", lambda: [a, b])
    first = kr.code_fingerprint()
    assert first == kr.code_fingerprint(), "the fingerprint must be stable"
    b.write_text("y = 3\n", encoding="utf-8")
    second = kr.code_fingerprint()
    assert first != second, "a changed tracked file must change the fingerprint"


def test_ensure_code_pushed_skips_an_unchanged_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_push(message: str) -> None:
        calls.append(message)

    state = tmp_path / "code_state.json"
    monkeypatch.setattr(kr, "CODE_STATE", state)
    monkeypatch.setattr(kr, "code_fingerprint", lambda: "abc123")
    monkeypatch.setattr(kr, "push_code", fake_push)

    kr.ensure_code_pushed()
    assert calls == ["code update"], "a first run must upload the code"
    assert json.loads(state.read_text(encoding="utf-8"))["fingerprint"] == "abc123"

    kr.ensure_code_pushed()
    assert calls == ["code update"], "an unchanged tree must not upload again"

    monkeypatch.setattr(kr, "code_fingerprint", lambda: "different")
    kr.ensure_code_pushed()
    assert calls == ["code update", "code update"], "a changed tree must upload again"


def test_ensure_code_pushed_survives_a_corrupt_state_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    state = tmp_path / "code_state.json"
    state.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(kr, "CODE_STATE", state)
    monkeypatch.setattr(kr, "code_fingerprint", lambda: "abc")
    monkeypatch.setattr(kr, "push_code", lambda message: calls.append(message))
    kr.ensure_code_pushed()
    assert calls == ["code update"]
