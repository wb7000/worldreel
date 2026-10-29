"""Build one MPFB2 human from a preset and render a look-test (A-14 v2, D-8.11). Runs inside
Blender 4.2+ with the MPFB extension installed:

    blender -b --factory-startup -P tools/blender/human.py -- build preset.json out_dir
    blender -b out_dir/human.blend -P tools/blender/human.py -- portrait aligned.png out_dir \
        [--iris R,G,B]

`build` makes the human, scales it to the bible height, colours hair and eyes, renders
out_dir/face.png through the fixed orthographic face camera and saves out_dir/human.blend.
`portrait` projects the aligned portrait (RGBA in face.png's pixels, from
`stages.generate.face_projection`) onto the skin from that camera and renders body.png and
head.png (three-point light, EEVEE). All outputs are git-ignored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]
from bpy_extras.object_utils import world_to_camera_view  # type: ignore[import-not-found]
from mathutils import Vector  # type: ignore[import-not-found]

FACE_PX = 1024


def _mpfb():  # type: ignore[no-untyped-def]
    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.mpfb")
    from bl_ext.user_default.mpfb.services.humanservice import (  # type: ignore[import-not-found]
        HumanService,
    )

    return HumanService


def _linear(rgb: list[float]) -> tuple[float, float, float, float]:
    r, g, b = (c**2.2 for c in rgb)  # sRGB 0-1 -> approx. linear
    return (r, g, b, 1.0)


def _light(name: str, loc: tuple[float, float, float], energy: float, size: float,
           target: Vector) -> None:  # fmt: skip
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.size = energy, size
    ob = bpy.data.objects.new(name, data)
    ob.location = loc
    ob.rotation_euler = (target - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(ob)


def _look(cam: object, target: Vector, offset: Vector) -> None:
    cam.location = target + offset  # type: ignore[attr-defined]
    cam.rotation_euler = (-offset).to_track_quat("-Z", "Y").to_euler()  # type: ignore[attr-defined]


def _camera(name: str, ortho: float | None = None) -> object:
    cam = bpy.data.objects.get(name)
    if cam is None:
        cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        bpy.context.collection.objects.link(cam)
    if ortho:
        cam.data.type, cam.data.ortho_scale = "ORTHO", ortho
    return cam


def _tint(ob: object, rgb: list[float]) -> None:
    """Hair colour: the texture's brightness modulates the target colour (keeps strands)."""
    for slot in ob.material_slots:  # type: ignore[attr-defined]
        nt = slot.material.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        link = bsdf.inputs["Base Color"].links[0]
        bw = nt.nodes.new("ShaderNodeRGBToBW")
        rng = nt.nodes.new("ShaderNodeMapRange")
        rng.inputs["From Max"].default_value = 0.35
        rng.inputs["To Min"].default_value, rng.inputs["To Max"].default_value = 0.3, 1.0
        comb = nt.nodes.new("ShaderNodeCombineColor")
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs["B"].default_value = _linear(rgb)
        nt.links.new(link.from_socket, bw.inputs["Color"])
        nt.links.new(bw.outputs["Val"], rng.inputs["Value"])
        for ch in ("Red", "Green", "Blue"):
            nt.links.new(rng.outputs["Result"], comb.inputs[ch])
        nt.links.new(comb.outputs["Color"], mix.inputs["A"])
        nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])


def _iris(human: object, rgb: list[float]) -> None:
    for ob in human.children:  # type: ignore[attr-defined]
        for slot in ob.material_slots:
            for n in slot.material.node_tree.nodes:
                if n.type == "GROUP" and "IrisMajorColor" in n.inputs:
                    n.inputs["IrisMajorColor"].default_value = _linear(rgb)
                    n.inputs["IrisMinorColor"].default_value = _linear([0.6 * c for c in rgb])


def _scene_setup(target: Vector) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.world = scene.world or bpy.data.worlds.new("world")
    scene.world.color = (0.18, 0.18, 0.19)
    scene.view_settings.view_transform = "AgX"
    _light("key", (1.5, -2.0, 2.4), 300.0, 1.5, target)
    _light("fill", (-2.0, -1.5, 1.6), 120.0, 2.0, target)
    _light("rim", (0.5, 2.0, 2.2), 200.0, 1.0, target)


def _render(cam: object, path: Path, w: int, h: int) -> None:
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def build(preset: Path, out: Path) -> None:
    human_service = _mpfb()
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    info = json.loads(preset.read_text())
    settings = human_service.get_default_deserialization_settings()
    human = human_service.deserialize_from_dict(info, settings)
    bpy.context.view_layer.update()
    top = max((human.matrix_world @ Vector(c)).z for c in human.bound_box)
    extra = info.get("worldreel", {})
    if extra.get("height_m"):  # the bible's height in metres; parented parts follow
        human.scale = human.scale * (extra["height_m"] / top)
        bpy.context.view_layer.update()
        top = max((human.matrix_world @ Vector(c)).z for c in human.bound_box)
    hair = info.get("hair", "").removesuffix(".mhclo")
    for ob in human.children:
        if hair and ob.name.endswith("." + hair) and extra.get("hair_rgb"):
            _tint(ob, extra["hair_rgb"])
        if "eyebrow" in ob.name and extra.get("hair_rgb"):
            _tint(ob, extra["hair_rgb"])
    k = top / 1.75
    face_z = top - 0.115 * k  # roughly the eyes
    _scene_setup(Vector((0.0, 0.0, face_z)))
    cam = _camera("facecam", ortho=0.30 * k)
    _look(cam, Vector((0.0, 0.0, face_z - 0.02 * k)), Vector((0.0, -1.0, 0.0)))
    _render(cam, out / "face.png", FACE_PX, FACE_PX)
    (out / "build.json").write_text(json.dumps({"height_m": round(top, 3), "face_z": face_z}))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "human.blend"))


def _project(human: object, cam: object, image: Path) -> None:
    """UV map 'portrait' = each vertex seen through the face camera; the skin group mixes the
    portrait in by its alpha times a facing term (fades out towards the ears)."""
    mods = [(m, m.show_viewport) for m in human.modifiers]  # type: ignore[attr-defined]
    for m, _ in mods:
        m.show_viewport = False
    deps = bpy.context.evaluated_depsgraph_get()
    ev = human.evaluated_get(deps)  # type: ignore[attr-defined]
    world = [ev.matrix_world @ v.co for v in ev.data.vertices]  # shape keys applied
    for m, shown in mods:
        m.show_viewport = shown
    scene = bpy.context.scene
    proj = [world_to_camera_view(scene, cam, co) for co in world]
    mesh = human.data  # type: ignore[attr-defined]
    uv = mesh.uv_layers.get("portrait") or mesh.uv_layers.new(name="portrait", do_init=False)
    for loop in mesh.loops:
        p = proj[loop.vertex_index]
        uv.data[loop.index].uv = (p.x, p.y)
    g = bpy.data.node_groups[human.material_slots[0].material.name]  # type: ignore[attr-defined]
    # after the group's colour tint (texture -> Mix with colorMixIn), so the tint never
    # flattens the portrait's detail
    tex = g.nodes["Mix"]
    link = next(lk for lk in g.links if lk.from_node == tex)
    uvn = g.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "portrait"
    img = g.nodes.new("ShaderNodeTexImage")
    img.image = bpy.data.images.load(str(image))
    img.extension = "CLIP"
    geo = g.nodes.new("ShaderNodeNewGeometry")
    dot = g.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = (0.0, -1.0, 0.0)  # towards the face camera
    face = g.nodes.new("ShaderNodeMapRange")
    face.inputs["From Min"].default_value, face.inputs["From Max"].default_value = 0.25, 0.65
    fac = g.nodes.new("ShaderNodeMath")
    fac.operation = "MULTIPLY"
    mix = g.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    g.links.new(uvn.outputs["UV"], img.inputs["Vector"])
    g.links.new(geo.outputs["Normal"], dot.inputs[0])
    g.links.new(dot.outputs["Value"], face.inputs["Value"])
    g.links.new(face.outputs["Result"], fac.inputs[0])
    g.links.new(img.outputs["Alpha"], fac.inputs[1])
    g.links.new(fac.outputs["Value"], mix.inputs["Factor"])
    g.links.new(link.from_socket, mix.inputs["A"])
    g.links.new(img.outputs["Color"], mix.inputs["B"])
    g.links.new(mix.outputs["Result"], link.to_socket)


def portrait(image: Path, out: Path, iris: list[float] | None, skin: list[float] | None) -> None:
    human = bpy.data.objects["Human"]
    build_info = json.loads((out / "build.json").read_text())
    top, face_z = build_info["height_m"], build_info["face_z"]
    _project(human, bpy.data.objects["facecam"], image)
    for ob in human.children:  # the portrait carries its own eyebrows
        if "eyebrow" in ob.name:
            ob.hide_render = True
    if iris:
        _iris(human, iris)
    if skin:  # body skin toward the portrait's colour (the group's own colorMixIn input)
        node = human.material_slots[0].material.node_tree.nodes[
            human.material_slots[0].material.name
        ]
        node.inputs["colorMixIn"].default_value = _linear(skin[:3])
        node.inputs["colorMixInStrength"].default_value = skin[3]
    body = _camera("bodycam")
    body.data.lens = 50.0
    _look(body, Vector((0.0, 0.0, top / 2)), Vector((0.35, -1.0, 0.05)) * (2.35 * top))
    _render(body, out / "body.png", 768, 1024)
    head = _camera("headcam")
    head.data.lens = 85.0
    _look(head, Vector((0.0, 0.0, face_z - 0.03)), Vector((0.3, -0.85, 0.05)))
    _render(head, out / "head.png", 768, 1024)
    _look(head, Vector((0.0, 0.0, face_z - 0.03)), Vector((0.0, -0.9, 0.0)))
    _render(head, out / "front.png", 768, 1024)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "human_portrait.blend"))


def main(argv: list[str]) -> None:
    # Blender resolves relative paths against its own folders, not the working directory
    if argv[0] == "build":
        out = Path(argv[2]).resolve()
        out.mkdir(parents=True, exist_ok=True)
        build(Path(argv[1]).resolve(), out)
    elif argv[0] == "portrait":
        iris = None
        if "--iris" in argv:
            iris = [float(c) for c in argv[argv.index("--iris") + 1].split(",")]
        skin = None
        if "--skin" in argv:
            skin = [float(c) for c in argv[argv.index("--skin") + 1].split(",")]
        portrait(Path(argv[1]).resolve(), Path(argv[2]).resolve(), iris, skin)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :])
