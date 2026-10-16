"""Mock `audio`: an empty audio section with a neutral mood. Stem separation and sound events
arrive in Phase 5; the slot exists now so packaging always has one."""

from __future__ import annotations

from typing import ClassVar

from pipeline.io import AUDIO_JSON, write_json
from pipeline.stage import Impl, StageContext


class MockAudioStage:
    name: ClassVar[str] = "audio"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "mock"
    depends_on: ClassVar[tuple[str, ...]] = ("frames",)

    def run(self, ctx: StageContext) -> None:
        write_json(
            ctx.out / AUDIO_JSON,
            {"tracks": [], "events": [], "mood": [{"t": 0.0, "tension": 5}], "mock": True},
        )
