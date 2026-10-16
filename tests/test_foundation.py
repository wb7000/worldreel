"""Phase 0 checks: repository structure, hygiene rules and the hardware tool."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import stages
from tools import hwcheck

ROOT = Path(__file__).resolve().parent.parent

BRIEF_STAGES = (
    "ingest",
    "frames",
    "camera",
    "reconstruct",
    "actors",
    "fill",
    "audio",
    "package",
    "export",
    "generate",
    "safety",
)


def test_stage_list_matches_brief() -> None:
    assert stages.STAGES == BRIEF_STAGES


@pytest.mark.parametrize("name", BRIEF_STAGES)
def test_each_stage_is_a_documented_package(name: str) -> None:
    module = importlib.import_module(f"stages.{name}")
    assert name == module.STAGE_NAME
    assert module.__doc__ and module.__doc__.strip()


@pytest.mark.parametrize(
    "path",
    [
        "CLAUDE.md",
        "DECISIONS.md",
        "README.md",
        ".gitignore",
        "spec/README.md",
        "spec/wreel.md",
        "spec/wedit.md",
        "spec/wscript.md",
        "spec/schemas/wreel-manifest-1.schema.json",
        "spec/schemas/wedit-manifest-1.schema.json",
        "spec/schemas/wscript-header-1.schema.json",
        "docs/ARCHITECTURE.md",
        "docs/MODELS.md",
        "docs/DATA.md",
        "docs/ROADMAP.md",
        "docs/ALGORITHMS.md",
        "player/package.json",
        ".github/workflows/ci.yml",
    ],
)
def test_required_files_exist(path: str) -> None:
    assert (ROOT / path).is_file(), path


def test_gitignore_blocks_data_media_and_weights() -> None:
    lines = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    for pattern in ("data/*", "*.safetensors", "*.ckpt", "*.pt", "*.mp4", "*.wav", "*.wreel"):
        assert pattern in lines, pattern


def test_claude_md_states_hard_rules() -> None:
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8").lower()
    assert "never commit data" in text
    assert "safety hooks" in text
    assert "do not invent" in text


def test_hardware_is_recorded_in_decisions() -> None:
    text = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    assert "Hardware of the primary dev machine" in text


def test_parse_nvidia_smi() -> None:
    gpus = hwcheck.parse_nvidia_smi("NVIDIA GeForce RTX 3060 Laptop GPU, 6144\n\n")
    assert len(gpus) == 1
    assert gpus[0].vram_mb == 6144
    assert gpus[0].cuda is True


def test_parse_nvidia_smi_handles_garbage() -> None:
    assert hwcheck.parse_nvidia_smi("") == []
    assert hwcheck.parse_nvidia_smi("No devices were found") == []


def test_hwcheck_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "hw.json"
    assert hwcheck.main(["--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert {"cpu", "ram_gb", "gpus", "has_cuda_gpu", "tools"} <= data.keys()
    assert data["has_cuda_gpu"] == any(g["cuda"] for g in data["gpus"])
