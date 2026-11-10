"""Which implementation runs for each stage, and the pipeline configuration file.

Config (TOML, default `pipeline/default.toml`):

    seed = 0
    max_workers = 4
    [impl]            # "mock" or "real" per stage
    frames = "real"
    [stage.frames]    # options passed to that stage
    fps = 5
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.stage import Stage
from stages.actors.mock import MockActorsStage
from stages.actors.oracle import OracleActorsStage
from stages.actors.sam2_stage import Sam2ActorsStage
from stages.audio.mock import MockAudioStage
from stages.camera.classical import ClassicalCameraStage
from stages.camera.mock import MockCameraStage
from stages.export.mock import MockExportStage
from stages.fill.mock import MockFillStage
from stages.frames.images import ImageSequenceFramesStage
from stages.frames.mock import MockFramesStage
from stages.frames.real import FramesStage
from stages.generate.mock import MockGenerateStage
from stages.generate.real import GenerateStage
from stages.package.real import PackageStage
from stages.reconstruct.mock import MockReconstructStage
from stages.reconstruct.planesweep import PlaneSweepReconstructStage

DEFAULT_CONFIG = Path(__file__).with_name("default.toml")

IMPLEMENTATIONS: dict[str, dict[str, type]] = {
    "frames": {"mock": MockFramesStage, "real": FramesStage, "images": ImageSequenceFramesStage},
    "camera": {"mock": MockCameraStage, "real": ClassicalCameraStage},
    "reconstruct": {"mock": MockReconstructStage, "real": PlaneSweepReconstructStage},
    "actors": {"mock": MockActorsStage, "oracle": OracleActorsStage, "real": Sam2ActorsStage},
    "fill": {"mock": MockFillStage},
    "audio": {"mock": MockAudioStage},
    "package": {"mock": PackageStage, "real": PackageStage},
    "export": {"mock": MockExportStage},
    "generate": {"mock": MockGenerateStage, "real": GenerateStage},
}

VIDEO_PIPELINE = ("frames", "actors", "camera", "reconstruct", "fill", "audio", "package", "export")


@dataclass
class PipelineConfig:
    seed: int = 0
    max_workers: int = 4
    impl: dict[str, str] = field(default_factory=dict)
    stage: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(
        cls, path: Path | None = None, overrides: Mapping[str, str] | None = None
    ) -> PipelineConfig:
        with (path or DEFAULT_CONFIG).open("rb") as fh:
            raw = tomllib.load(fh)
        cfg = cls(
            seed=int(raw.get("seed", 0)),
            max_workers=int(raw.get("max_workers", 4)),
            impl={str(k): str(v) for k, v in raw.get("impl", {}).items()},
            stage={str(k): dict(v) for k, v in raw.get("stage", {}).items()},
        )
        cfg.impl.update(overrides or {})
        return cfg

    @classmethod
    def all_mock(cls) -> PipelineConfig:
        return cls(impl=dict.fromkeys(IMPLEMENTATIONS, "mock"))


def build_stages(names: tuple[str, ...], impl: Mapping[str, str]) -> list[Stage]:
    stages: list[Stage] = []
    for name in names:
        choice = impl.get(name, "mock")
        options = IMPLEMENTATIONS[name]
        if choice not in options:
            raise ValueError(
                f"stage {name!r} has no {choice!r} implementation (available: {sorted(options)})"
            )
        stages.append(options[choice]())
    return stages
