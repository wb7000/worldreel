"""Render a short camera orbit of a finished MPFB human for the realism-pass test (D-8.12):

    blender -b out_dir/human_portrait.blend -P tools/blender/orbit.py -- out_dir [frames]

Head and shoulders, camera at eye height 0.9 m away swinging from -25 to +25 degrees yaw,
832 x 480 (Wan 480p landscape), frames (default 33 = 4n + 1 as Wan needs). Writes
out_dir/orbit/f000.png ...
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]


def main(out: Path, frames: int) -> None:
    face_z = json.loads((out / "build.json").read_text())["face_z"]
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 832, 480
    cam = bpy.data.objects.new("orbitcam", bpy.data.cameras.new("orbitcam"))
    bpy.context.collection.objects.link(cam)
    cam.data.lens = 50.0
    scene.camera = cam
    target = Vector((0.0, -0.03, face_z - 0.06))
    (out / "orbit").mkdir(exist_ok=True)
    for i in range(frames):
        yaw = math.radians(-25 + 50 * i / (frames - 1))
        off = Vector((math.sin(yaw), -math.cos(yaw), 0.05)) * 0.9  # the human faces -Y
        cam.location = target + off
        cam.rotation_euler = (-off).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(out / "orbit" / f"f{i:03d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1 :]
    main(Path(args[0]).resolve(), int(args[1]) if len(args) > 1 else 33)
