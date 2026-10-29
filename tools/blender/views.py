"""Render a finished MPFB human from the D-8.0 camera spread (A-14 v2, D-8.11):

    blender -b out_dir/human_portrait.blend -P tools/blender/views.py -- out_dir

Distances 1 / 2 / 3 m and yaws 0-40 degrees, camera at eye height looking at the eyes,
854 x 480, 50 degree vertical field of view (as `eval.face_metrics`). Writes
out_dir/views/d{d}_y{yaw}.png for `eval.human_faces`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

DISTANCES = (1.0, 2.0, 3.0)
YAWS = (0, 10, 20, 30, 40)


def main(out: Path) -> None:
    face_z = json.loads((out / "build.json").read_text())["face_z"]
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 854, 480
    scene.eevee.taa_render_samples = 16
    cam = bpy.data.objects.new("viewcam", bpy.data.cameras.new("viewcam"))
    bpy.context.collection.objects.link(cam)
    cam.data.sensor_fit = "VERTICAL"
    cam.data.angle = math.radians(50.0)
    scene.camera = cam
    target = Vector((0.0, 0.0, face_z))
    (out / "views").mkdir(exist_ok=True)
    for d in DISTANCES:
        for yaw in YAWS:
            th = math.radians(yaw)
            off = Vector((math.sin(th), -math.cos(th), 0.0)) * d  # the human faces -Y
            cam.location = target + off
            cam.rotation_euler = (-off).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = str(out / "views" / f"d{d:.0f}_y{yaw}.png")
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--") + 1]).resolve())
