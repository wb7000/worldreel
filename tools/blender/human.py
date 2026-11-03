"""Build one MPFB2 human from a preset and render a look-test (A-14 v2, D-8.11). Runs inside
Blender 4.2+ with the MPFB extension installed:

    blender -b --factory-startup -P tools/blender/human.py -- build preset.json out_dir
    blender -b out_dir/human.blend -P tools/blender/human.py -- portrait aligned.png out_dir \
        [--eye NAME] [--skin R,G,B,STRENGTH]

`build` makes the human, scales it to the bible height, colours hair and eyes, renders
out_dir/face.png through the fixed orthographic face camera and saves out_dir/human.blend.
`portrait` projects the aligned portrait (RGBA in face.png's pixels, from
`stages.generate.face_projection`) onto the skin from that camera and renders body.png and
head.png (three-point light, EEVEE). All outputs are git-ignored.
"""

from __future__ import annotations

import json
import math
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
        # a broad white sheen (roughness 0.7) lifted black hair to grey (D-8.11c)
        bsdf.inputs["Roughness"].default_value = 0.65
        bsdf.inputs["Specular IOR Level"].default_value = 0.2


HAIR_ASSETS = "datafiles/assets/geometry_nodes/procedural_hair_node_assets.blend"


def _hair_groups() -> dict[str, object]:
    path = bpy.utils.resource_path("LOCAL") + "/" + HAIR_ASSETS
    want = ("Generate Hair Curves", "Frizz Hair Curves", "Curl Hair Curves", "Hair Curves Noise",
            "Set Hair Curve Profile", "Displace Hair Curves", "Shrinkwrap Hair Curves",
            "Clump Hair Curves")  # fmt: skip
    missing = [n for n in want if n not in bpy.data.node_groups]
    if missing:
        with bpy.data.libraries.load(path) as (_src, dst):
            dst.node_groups = missing
    return {n: bpy.data.node_groups[n] for n in want}


def _strand_material(rgb: list[float]) -> object:
    """Per-strand brightness variation (Curves Info random) around the hair colour."""
    mat = bpy.data.materials.new("strands")  # one per strand object
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    # thin strands catch white highlights and read grey: low specular, matte
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Specular IOR Level"].default_value = 0.08
    info = nt.nodes.new("ShaderNodeHairInfo")
    rng = nt.nodes.new("ShaderNodeMapRange")
    # fine strands scatter a lot of light: darker on average than the hair's albedo colour, or
    # auburn rendered bright orange (D-8.11c)
    rng.inputs["To Min"].default_value, rng.inputs["To Max"].default_value = 0.45, 0.85
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    # strands multiply-scatter and saturate: auburn rendered bright red, so mute the colour
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    mix.inputs["A"].default_value = _linear([lum + 0.7 * (c - lum) for c in rgb])
    comb = nt.nodes.new("ShaderNodeCombineColor")
    nt.links.new(info.outputs["Random"], rng.inputs["Value"])
    for ch in ("Red", "Green", "Blue"):
        nt.links.new(rng.outputs["Result"], comb.inputs[ch])
    nt.links.new(comb.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    return mat


def hairline(azimuth_deg: float) -> float:
    """Hairline height above the eyes (in units of k = height / 1.75 m) around the head:
    0 deg = front, 90 = over the ears, 180 = nape."""
    a = abs(azimuth_deg)
    if a <= 60:
        return 0.065
    if a <= 110:
        return 0.065 - (a - 60) / 50 * 0.015  # down to 0.05 above the ears
    return 0.05 - (a - 110) / 70 * 0.12  # down to the nape (-0.07)


def _scalp_group(human: object, face_z: float, k: float) -> None:
    """Vertex group 'scalp' on the body mesh, above a hairline that runs around the head
    (D-8.11c: a front / side / back split put roots over the ears down to the jaw)."""
    mods = [(m, m.show_viewport) for m in human.modifiers]  # type: ignore[attr-defined]
    for m, _ in mods:
        m.show_viewport = False
    ev = human.evaluated_get(bpy.context.evaluated_depsgraph_get())  # type: ignore[attr-defined]
    world = [ev.matrix_world @ v.co for v in ev.data.vertices]
    for m, shown in mods:
        m.show_viewport = shown
    # body vertices only: MakeHuman's hidden helper geometry includes a bob-shaped
    # `helper-hair` shell, and roots on it hung the hair down to the jaw (D-8.11c)
    body_g = human.vertex_groups["body"].index  # type: ignore[attr-defined]
    real = [any(g.group == body_g for g in v.groups) for v in human.data.vertices]  # type: ignore[attr-defined]
    skull = [co for co, r in zip(world, real, strict=True) if r and co.z > face_z]
    xc = sum(c.x for c in skull) / len(skull)
    yc = sum(c.y for c in skull) / len(skull)
    idx = []
    for i, co in enumerate(world):
        if not real[i] or co.z < face_z - 0.1 * k:
            continue
        x, y = co.x - xc, co.y - yc
        if math.hypot(x, y) > 0.13 * k:
            continue  # shoulders / not the head
        azimuth = math.degrees(math.atan2(abs(x), -y))  # the human faces -Y
        ear = abs(x) > 0.06 * k and -0.03 * k < y < 0.035 * k and co.z < face_z + 0.045 * k
        if not ear and co.z > face_z + hairline(azimuth) * k:
            idx.append(i)
    vg = human.vertex_groups.get("scalp") or human.vertex_groups.new(name="scalp")  # type: ignore[attr-defined]
    vg.add(idx, 1.0, "REPLACE")


def _strand_hair(human: object, rgb: list[float], curl: float, length: float, k: float,
                 seed: int, vgroup: str = "scalp", density: float = 300000.0,
                 comb: tuple[float, float] = (1.0, -0.3), offset: float | None = None,
                 radius: float = 0.0002, name: str = "strands",
                 wiggle: float | None = None) -> None:  # fmt: skip
    """Real strands from the scalp (Blender's procedural hair node assets): MakeHuman hair is a
    few large cards and read as a plastic helmet (owner, D-8.11c). Length from the bible,
    curl from the portrait, a backward / downward lie, frizz and noise for a natural look."""
    g = _hair_groups()
    curves = bpy.data.hair_curves.new(name)
    ob = bpy.data.objects.new(name, curves)
    bpy.context.collection.objects.link(ob)
    ob.matrix_world = human.matrix_world.copy()  # type: ignore[attr-defined]
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    out = nodes.new("NodeGroupOutput")

    def group(name: str) -> object:
        n = nodes.new("GeometryNodeGroup")
        n.node_tree = g[name]
        return n

    scalp = nodes.new("GeometryNodeInputNamedAttribute")
    scalp.data_type = "FLOAT"
    scalp.inputs["Name"].default_value = vgroup
    gen = group("Generate Hair Curves")
    gen.inputs[1].default_value = human  # the "Surface" object input
    gen.inputs["Hair Length"].default_value = length
    gen.inputs["Density"].default_value = density / (k * k)
    gen.inputs["Control Points"].default_value = 10
    gen.inputs["Seed"].default_value = seed
    gen.inputs["Hair Material"].default_value = _strand_material(rgb)
    links.new(scalp.outputs[0], gen.inputs["Density Mask"])
    lie = group("Displace Hair Curves")  # comb back and down, more towards the tips
    lie.inputs["Factor"].default_value = 1.0
    lie.inputs["Shape"].default_value = 0.4
    lie.inputs["Displace Vector"].default_value = (0.0, comb[0] * length, comb[1] * length)
    links.new(gen.outputs[0], lie.inputs["Geometry"])
    # lay the combed strands onto the head with some volume (straight out they read as a
    # porcupine); curls add their own volume
    wrap = group("Shrinkwrap Hair Curves")
    wrap.inputs[2].default_value = human  # the "Surface" object input
    wrap.inputs["Offset Distance"].default_value = (
        offset if offset is not None else (0.003 + 0.004 * curl) * k
    )
    wrap.inputs["Smoothing Steps"].default_value = 2
    links.new(lie.outputs[0], wrap.inputs["Geometry"])
    last = wrap
    frizz = group("Frizz Hair Curves")
    # curls as strong frizz + noise: the Curl asset either clumps strands to shared guides
    # (spikes) or, with per-strand guides, does nothing visible (D-8.11c)
    frizz.inputs["Factor"].default_value = 0.4 + 0.2 * curl
    frizz.inputs["Distance"].default_value = (0.0015 + 0.0012 * curl) * k
    if wiggle is not None:  # short facial hair: wobble relative to length (fixed 2-4 mm made
        frizz.inputs["Distance"].default_value = wiggle * length  # 2.5 mm stubble a beard)
    links.new(last.outputs[0], frizz.inputs["Geometry"])
    last = frizz
    if wiggle is None:  # head hair: locks, not straw (owner: "hair fucked up", D-8.11e)
        clump = group("Clump Hair Curves")
        clump.inputs["Existing Guide Map"].default_value = False
        clump.inputs["Guide Distance"].default_value = 0.006 * k  # small locks; 10 cm = spikes
        clump.inputs["Factor"].default_value = 0.55
        clump.inputs["Tip Spread"].default_value = 0.001 * k
        clump.inputs["Seed"].default_value = seed
        links.new(last.outputs[0], clump.inputs["Geometry"])
        last = clump
    noise = group("Hair Curves Noise")
    noise.inputs["Factor"].default_value = 0.5 + 0.3 * curl
    noise.inputs["Distance"].default_value = (0.002 + 0.002 * curl) * k
    noise.inputs["Scale"].default_value = 30.0 + 40.0 * curl
    if wiggle is not None:
        noise.inputs["Distance"].default_value = 0.5 * wiggle * length
    links.new(last.outputs[0], noise.inputs["Geometry"])
    prof = group("Set Hair Curve Profile")
    prof.inputs["Replace Radius"].default_value = True
    prof.inputs["Radius"].default_value = radius * k
    prof.inputs["Factor Min"].default_value = 0.25  # tapered tips (blunt ends read as straw)
    prof.inputs["Factor Min"].default_value = 0.25  # tapered tips (blunt ends read as straw)
    links.new(noise.outputs[0], prof.inputs["Geometry"])
    links.new(prof.outputs[0], out.inputs[0])
    mod = ob.modifiers.new("strands", "NODES")
    mod.node_group = tree


def _scene_setup(target: Vector) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.world = scene.world or bpy.data.worlds.new("world")
    scene.world.color = (0.18, 0.18, 0.19)
    scene.view_settings.view_transform = "AgX"
    # look-test light: two soft keys mirrored left / right, so neither half of the face reads
    # as a different person (owner, D-8.11d); scene lighting comes from the world later (8.4)
    _light("key", (1.0, -2.2, 2.3), 250.0, 3.0, target)
    _light("fill", (-1.0, -2.2, 2.3), 250.0, 3.0, target)
    _light("rim", (0.0, 2.0, 2.2), 200.0, 1.0, target)  # centred: off to one side it lit one cheek


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
            hair_ob = ob
        if "eyebrow" in ob.name and extra.get("hair_rgb"):
            _tint(ob, extra["hair_rgb"])
    k = top / 1.75
    face_z = top - 0.115 * k  # roughly the eyes
    if hair and extra.get("hair_rgb"):
        _scalp_group(human, face_z, k)
        length = float(extra.get("hair_len_m", 0.015)) * k
        _strand_hair(human, extra["hair_rgb"], float(extra.get("curl", 0.0)), length, k, len(hair))
        if extra.get("strands_only"):  # short hair: the strands replace the card helmet
            hair_ob.hide_render = hair_ob.hide_viewport = True
    _scene_setup(Vector((0.0, 0.0, face_z)))
    cam = _camera("facecam", ortho=0.30 * k)
    _look(cam, Vector((0.0, 0.0, face_z - 0.02 * k)), Vector((0.0, -1.0, 0.0)))
    _render(cam, out / "face.png", FACE_PX, FACE_PX)
    (out / "build.json").write_text(json.dumps({"height_m": round(top, 3), "face_z": face_z}))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "human.blend"))


def _camera_uv(ob: object, cam: object) -> None:
    """UV map 'portrait' = each vertex as the face camera sees it (shape keys applied)."""
    mods = [(m, m.show_viewport) for m in ob.modifiers]  # type: ignore[attr-defined]
    for m, _ in mods:
        m.show_viewport = False
    deps = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(deps)  # type: ignore[attr-defined]
    world = [ev.matrix_world @ v.co for v in ev.data.vertices]
    for m, shown in mods:
        m.show_viewport = shown
    scene = bpy.context.scene
    proj = [world_to_camera_view(scene, cam, co) for co in world]
    mesh = ob.data  # type: ignore[attr-defined]
    uv = mesh.uv_layers.get("portrait") or mesh.uv_layers.new(name="portrait", do_init=False)
    for loop in mesh.loops:
        p = proj[loop.vertex_index]
        uv.data[loop.index].uv = (p.x, p.y)


def _mask_group(human: object, cam: object, mask: Path, name: str) -> int:
    """Vertex group from a mask in the face camera's pixels (body vertices facing the camera);
    returns how many vertices got a weight."""
    img = bpy.data.images.load(str(mask))
    w, h = img.size
    px = list(img.pixels)  # RGBA floats, bottom row first
    mods = [(m, m.show_viewport) for m in human.modifiers]  # type: ignore[attr-defined]
    for m, _ in mods:
        m.show_viewport = False
    ev = human.evaluated_get(bpy.context.evaluated_depsgraph_get())  # type: ignore[attr-defined]
    verts = [
        (ev.matrix_world @ v.co, ev.matrix_world.to_3x3() @ v.normal) for v in ev.data.vertices
    ]
    for m, shown in mods:
        m.show_viewport = shown
    body_g = human.vertex_groups["body"].index  # type: ignore[attr-defined]
    real = [any(g.group == body_g for g in v.groups) for v in human.data.vertices]  # type: ignore[attr-defined]
    vg = human.vertex_groups.get(name) or human.vertex_groups.new(name=name)  # type: ignore[attr-defined]
    scene = bpy.context.scene
    n = 0
    for i, (co, nor) in enumerate(verts):
        if not real[i] or nor.y > -0.2:  # only the front of the face
            continue
        p = world_to_camera_view(scene, cam, co)
        x, y = int(p.x * (w - 1)), int(p.y * (h - 1))
        if 0 <= x < w and 0 <= y < h:
            v = px[(y * w + x) * 4]
            if v > 0.15:
                vg.add([i], float(v), "REPLACE")
                n += 1
    return n


def _project(human: object, cam: object, image: Path) -> None:
    """The skin group mixes the portrait in by its alpha times a facing term (fades out
    towards the ears); the eyeballs take the portrait's eyes."""
    _camera_uv(human, cam)
    g = bpy.data.node_groups[human.material_slots[0].material.name]  # type: ignore[attr-defined]
    # after the group's colour tint (texture -> Mix with colorMixIn), so the tint never
    # flattens the portrait's detail
    tex = g.nodes["Mix"]
    link = next(lk for lk in g.links if lk.from_node == tex)
    uvn = g.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "portrait"
    img = g.nodes.new("ShaderNodeTexImage")
    img.image = bpy.data.images.load(str(image))
    # the eyeballs keep their own textured eye (chosen by `eye_texture`): projecting the
    # portrait onto them (D-8.11b) looked glassy and misplaced, because the eyeball is larger
    # than the visible eye opening (owner look-test round, D-8.11c)
    img.extension = "CLIP"
    geo = g.nodes.new("ShaderNodeNewGeometry")
    dot = g.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = (0.0, -1.0, 0.0)  # towards the face camera
    face = g.nodes.new("ShaderNodeMapRange")
    # keep the portrait down the nose sides (fading early showed the body texture as grey
    # patches there, D-8.11c); fade only where the surface turns well away from the camera
    face.inputs["From Min"].default_value, face.inputs["From Max"].default_value = 0.0, 0.3
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


def _eye_texture(human: object, name: str) -> None:
    """Swap the textured eye material's image for the chosen CC0 eye texture."""
    for ob in human.children:  # type: ignore[attr-defined]
        if not ob.name.endswith(".high-poly"):
            continue
        for slot in ob.material_slots:
            for n in slot.material.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image and n.image.filepath.endswith("_eye.png"):
                    path = Path(bpy.path.abspath(n.image.filepath)).with_name(f"{name}_eye.png")
                    n.image = bpy.data.images.load(str(path))


Beard = list[tuple[Path, list[float], float, float, str]]  # mask, sRGB, length, density, piece


def portrait(image: Path, out: Path, eye: str | None, skin: list[float] | None,
             beard: Beard | None = None) -> None:  # fmt: skip
    human = bpy.data.objects["Human"]
    build_info = json.loads((out / "build.json").read_text())
    top, face_z = build_info["height_m"], build_info["face_z"]
    _project(human, bpy.data.objects["facecam"], image)
    for ob in human.children:  # the portrait carries its own eyebrows
        if "eyebrow" in ob.name:
            ob.hide_render = True
    if eye:
        _eye_texture(human, eye)
    k = top / 1.75
    for path, rgb, length, density, piece in beard or []:  # moustache / stubble / beard strands
        if _mask_group(human, bpy.data.objects["facecam"], path, piece) > 10:
            # combed down, lying close on the skin; moustache back towards the lip, not over it
            comb = (-0.5, -0.5) if piece == "moustache" else (-0.2, -1.0)
            _strand_hair(human, rgb, 0.0, length * k, k, 7, vgroup=piece, density=density,
                         comb=comb, offset=0.0004 * k, radius=0.00007, name=piece,
                         wiggle=0.15)  # fmt: skip
    # skin sheen: with the large soft key light the coat layer and broad specular laid a grey
    # veil over dark and red details (moustache, nose sides, under the eyes) (D-8.11c)
    skin_node = human.material_slots[0].material.node_tree.nodes[
        human.material_slots[0].material.name
    ]
    skin_node.inputs["Clearcoat"].default_value = 0.0
    skin_node.inputs["Roughness"].default_value = 0.6
    group = bpy.data.node_groups[human.material_slots[0].material.name]
    for n in group.nodes:
        if n.type == "BSDF_PRINCIPLED":
            n.inputs["Specular IOR Level"].default_value = 0.3
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
        eye = argv[argv.index("--eye") + 1] if "--eye" in argv else None
        skin = None
        if "--skin" in argv:
            skin = [float(c) for c in argv[argv.index("--skin") + 1].split(",")]
        beard = []
        for i, arg in enumerate(argv):
            if arg == "--beard":
                path, rgb, length, density, piece = argv[i + 1].split("@")
                beard.append((Path(path).resolve(), [float(c) for c in rgb.split(",")],
                              float(length), float(density), piece))  # fmt: skip
        portrait(Path(argv[1]).resolve(), Path(argv[2]).resolve(), eye, skin, beard)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :])
