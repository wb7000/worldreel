"""Mock `actors`: detects nothing. Segmentation and tracking models arrive in Phase 3."""

from __future__ import annotations

from typing import ClassVar

from pipeline.io import ACTORS_JSON, write_json
from pipeline.stage import Impl, StageContext


class MockActorsStage:
    name: ClassVar[str] = "actors"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "mock"
    depends_on: ClassVar[tuple[str, ...]] = ("frames", "camera")

    def run(self, ctx: StageContext) -> None:
        write_json(ctx.out / ACTORS_JSON, {"actors": [], "mock": True})
