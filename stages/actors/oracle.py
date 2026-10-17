"""`actors` from ground truth: exact person masks of a synthetic scene (benchmarks only).

Config: `scene` = path of the Blender scene folder (with gt.json). Lets the benchmark measure
everything downstream of detection (3D lifting, static-scene masking, packaging) exactly; the
real detector/tracker (stages/actors/sam2_stage.py) is measured separately against the same
ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from pipeline.stage import Impl, StageContext, StageError
from stages.actors.common import write_tracks
from tools.synth import actor_masks


class OracleActorsStage:
    name: ClassVar[str] = "actors"
    version: ClassVar[str] = "0.1.0"
    impl: ClassVar[Impl] = "oracle"
    depends_on: ClassVar[tuple[str, ...]] = ("frames",)

    def run(self, ctx: StageContext) -> None:
        scene = ctx.config.get("scene")
        if not scene or not (Path(scene) / "gt.json").exists():
            raise StageError("oracle actors need config 'scene' pointing at a synthetic scene")
        gt = json.loads((Path(scene) / "gt.json").read_text(encoding="utf-8"))
        labels = np.stack([actor_masks(Path(scene), i) for i in range(len(gt["frames"]))])
        write_tracks(ctx.out, labels, "oracle-ground-truth", ["person"] * len(gt.get("actors", [])))
