"""Mock `actors`: finds no people (stages/actors/common.py describes the output files)."""

from __future__ import annotations

from typing import ClassVar

from pipeline.io import ACTORS_JSON, write_json
from pipeline.stage import Impl, StageContext


class MockActorsStage:
    name: ClassVar[str] = "actors"
    version: ClassVar[str] = "0.2.0"
    impl: ClassVar[Impl] = "mock"
    depends_on: ClassVar[tuple[str, ...]] = ("frames",)

    def run(self, ctx: StageContext) -> None:
        write_json(
            ctx.out / ACTORS_JSON, {"backend": "none", "frame_count": 0, "tracks": [], "mock": True}
        )
