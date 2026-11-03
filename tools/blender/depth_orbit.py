"""Render a depth-only camera orbit of a built MPFB human (D-8.13): structure for the video model.

    blender -b out_dir/human.blend -P tools/blender/depth_orbit.py -- out_dir [frames]

Same camera path as `orbit.py` (+-25 degrees yaw, 832 x 480). Hair is hidden, nothing of the
portrait is used: near = bright, far = dark on black. Writes out_dir/depth/f000.png ...
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

NEAR, FAR = 0.66, 0.92  # metres from the camera: nose tip to behind the ears, for facial relief


def _depth_material() -> object:
    mat = bpy.data.materials.new("depth")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    cam = nt.nodes.new("ShaderNodeCameraData")
    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value, rng.inputs["From Max"].default_value = NEAR, FAR
    rng.inputs["To Min"].default_value, rng.inputs["To Max"].default_value = 1.0, 0.05
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(cam.outputs["View Z Depth"], rng.inputs["Value"])
    nt.links.new(rng.outputs["Result"], emit.inputs["Strength"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    out.is_active_output = True
    return mat


def main(out: Path, frames: int) -> None:
    face_z = json.loads((out / "build.json").read_text())["face_z"]
    scene = bpy.context.scene
    human = bpy.data.objects["Human"]
    hair = ("short", "long", "ponytail", "bob", "afro", "braid", "eyebrow", "eyelashes")
    for ob in human.children:
        if any(h in ob.name for h in hair):
            ob.hide_render = True
    for name in ("strands", "stubble", "moustache", "beard"):
        ob = bpy.data.objects.get(name)
        if ob is not None:
            ob.hide_render = True
    for name in ("key", "fill", "rim"):
        bpy.data.objects[name].hide_render = True
    depth = _depth_material()
    for ob in bpy.data.objects:
        if ob.type == "MESH":
            for slot in ob.material_slots:
                slot.link = "OBJECT"
                slot.material = depth
    scene.world.use_nodes = False
    scene.world.color = (0.0, 0.0, 0.0)
    scene.view_settings.view_transform = "Standard"
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x, scene.render.resolution_y = 832, 480
    cam = bpy.data.objects.new("depthcam", bpy.data.cameras.new("depthcam"))
    bpy.context.collection.objects.link(cam)
    cam.data.lens = 50.0
    scene.camera = cam
    target = Vector((0.0, -0.03, face_z - 0.06))
    (out / "depth").mkdir(exist_ok=True)
    for i in range(frames):
        yaw = math.radians(-25 + 50 * i / (frames - 1))
        off = Vector((math.sin(yaw), -math.cos(yaw), 0.05)) * 0.9  # the human faces -Y
        cam.location = target + off
        cam.rotation_euler = (-off).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(out / "depth" / f"f{i:03d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1 :]
    main(Path(args[0]).resolve(), int(args[1]) if len(args) > 1 else 33)
