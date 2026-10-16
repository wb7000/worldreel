"""Phase 2 acceptance (brief §15): dropping a file in inbox/ produces a manifest row and processed
outputs; a frozen dataset version can be reproduced."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formats.wreel import read_wreel
from tests.media import HAS_FFMPEG, make_clip
from tools.data import main as data_cli


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg for a real video")
def test_drop_ingest_process_freeze_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("WORLDREEL_SIGNING_KEY", str(tmp_path / "key.pem"))
    root = tmp_path / "data"
    cli = ["--root", str(root)]
    assert data_cli([*cli, "init"]) == 0

    clip = make_clip(root / "inbox" / "street.mp4", seconds=2.0)
    clip.with_name("street.mp4.info.json").write_text(
        json.dumps(
            {
                "source": "test clip generated with ffmpeg",
                "license": "CC0-1.0",
                "upload_rights": "synthetic",
            }
        )
    )
    assert data_cli([*cli, "ingest"]) == 0
    rows = (root / "manifest.csv").read_text().splitlines()
    assert len(rows) == 2 and ",active," in rows[1]
    file_id = rows[1].split(",")[0]

    assert data_cli([*cli, "process"]) == 0  # default config: real frames, mock reconstruction
    processed = root / "processed" / file_id
    assert len(list((processed / "frames" / "frames").glob("*.png"))) == 10
    world = read_wreel(processed / "package" / "world.wreel")
    assert world.report.ok and world.signature_status == "valid"
    assert world.manifest["provenance"]["ai_label"] == "AI-reconstructed"
    assert (
        world.manifest["provenance"]["source"]["sha256"]
        == world.manifest["generation"]["inputs"][0]["sha256"]
    )
    assert world.manifest["provenance"]["rights"]["license"] == "CC0-1.0"
    assert {s["name"] for s in world.manifest["generation"]["stages"]} >= {
        "frames",
        "camera",
        "package",
    }

    capsys.readouterr()
    assert data_cli([*cli, "process"]) == 0
    assert capsys.readouterr().out.count("cached") == 8  # resumable: nothing recomputed

    assert data_cli([*cli, "freeze", "v1"]) == 0
    assert data_cli([*cli, "verify", "v1"]) == 0
    assert "reproducible" in capsys.readouterr().out
