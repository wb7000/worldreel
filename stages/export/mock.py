"""Mock `export`: records what an export would contain. Real `.wedit` / MP4 / glTF export is
Phase 6."""

from __future__ import annotations

from typing import ClassVar

from formats.wreel import read_wreel
from pipeline.io import WORLD_WREEL, write_json
from pipeline.stage import Impl, StageContext, StageError


class MockExportStage:
    name: ClassVar[str] = "export"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "mock"
    depends_on: ClassVar[tuple[str, ...]] = ("package",)

    def run(self, ctx: StageContext) -> None:
        world = read_wreel(ctx.inputs["package"] / WORLD_WREEL)
        if not world.report.ok:
            raise StageError("package produced an invalid .wreel")
        write_json(
            ctx.out / "export.json",
            {
                "status": "mock: no files exported until Phase 6",
                "would_export": ["wedit", "mp4", "gltf"],
                "ai_label": world.manifest["provenance"]["ai_label"],
                "cameras": [c["id"] for c in world.manifest["cameras"]],
            },
        )
