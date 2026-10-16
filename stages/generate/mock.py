"""Mock `generate`: ignores the script and returns the canned synthetic courtyard scene.
Script-to-scene generation is Phase 8. Not part of the video pipeline graph."""

from __future__ import annotations

from typing import ClassVar

from formats.common import load_or_create_signing_key
from formats.wreel import write_wreel
from pipeline.io import WORLD_WREEL
from pipeline.stage import SOURCE, Impl, StageContext
from tools.make_sample_wreel import build_sample


class MockGenerateStage:
    name: ClassVar[str] = "generate"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "mock"
    depends_on: ClassVar[tuple[str, ...]] = (SOURCE,)

    def run(self, ctx: StageContext) -> None:
        manifest, entries = build_sample(seed=ctx.seed, created_at=ctx.config.get("created_at"))
        write_wreel(ctx.out / WORLD_WREEL, manifest, entries, load_or_create_signing_key())
