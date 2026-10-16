"""Synthetic scene generator. Runs INSIDE Blender (tested with 3.6 LTS):

    blender --background --factory-startup --python tools/blender/gen_scene.py -- \
        --out <dir> --seed 1 [--frames 24 --fps 12 --width 320 --height 240 --engine EEVEE]

Writes to <dir>:
    rgb/0001.png ...        rendered frames
    depth/0001.npy ...      float32 (H, W) planar depth in metres along the camera's -Z; 0 = sky
    gt.json                 camera intrinsics + per-frame pose, actor tracks, lights (WorldReel
                            Y-up coordinates, spec/wreel.md §2)

Use tools/synth.py rather than calling this directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]
import numpy as np
from mathutils import Vector  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import (
    actor_quat_zup_to_yup,
    blender_actor_yaw,
    intrinsics,
    quat_zup_to_yup,
    vec_zup_to_yup,
)

GENERATOR_VERSION = "0.1.0"
CLIP_END = 200.0
# Pixels with no surface come back as a huge value (EEVEE: 65504, the float16 maximum;
# Cycles: ~1e10). Anything at or beyond the far clip plane cannot be a real surface.
SKY_DEPTH = CLIP_END * 0.999
# Blender's depth pass stores distance along the camera axis (planar), verified 2026-10-16 on
# Blender 3.6.23 with EEVEE and Cycles (see DECISIONS D-2.5). Set True if a version changes it.
DEPTH_IS_RADIAL = False


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--actors", type=int, default=2)
    p.add_argument("--boxes", type=int, default=5)
    p.add_argument("--probe", action="store_true", help="depth-semantics probe scene (one plane)")
    return p.parse_args(argv)


def material(name: str, rgb: tuple[float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    return mat


def add_mesh(kind: str, name: str, mat: bpy.types.Material, **kw: object) -> bpy.types.Object:
    getattr(bpy.ops.mesh, f"primitive_{kind}_add")(**kw)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_render(scene: bpy.types.Scene, args: argparse.Namespace, out: Path) -> None:
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    scene.frame_start = 1
    scene.frame_end = args.frames
    if args.engine == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = args.samples
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = args.samples
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"  # plain sRGB, no Filmic tone curve
    scene.render.filepath = str(out / "rgb") + os.sep  # frames become rgb/0001.png
    scene.view_layers[0].use_pass_z = True

    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    depth_out = tree.nodes.new("CompositorNodeOutputFile")
    depth_out.base_path = str(out / "depth_exr")
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_depth = "32"
    depth_out.file_slots[0].path = ""
    socket = layers.outputs.get("Depth") or layers.outputs.get("Z")
    tree.links.new(socket, depth_out.inputs[0])


def build_scene(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], bpy.types.Object, list[dict[str, object]]]:
    rng = random.Random(args.seed)
    scene = bpy.context.scene
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    sky = (0.55, 0.68, 0.85)
    world.node_tree.nodes["Background"].inputs[0].default_value = (*sky, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 30.0
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "AUTO"
    cam_data.clip_start = 0.05
    cam_data.clip_end = CLIP_END
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    actors: list[dict[str, object]] = []
    if args.probe:
        # A single wall facing the camera at exactly 5 m: planar depth is 5 everywhere on it.
        add_mesh(
            "plane",
            "probe_wall",
            material("wall", (0.8, 0.8, 0.8)),
            size=40,
            location=(0, 5, 0),
            rotation=(math.pi / 2, 0, 0),
        )
        cam.location = (0, 0, 0)
        look_at(cam, Vector((0, 1, 0)))
        cam.keyframe_insert("location", frame=1)
        cam.keyframe_insert("rotation_euler", frame=1)
        return actors, cam, []

    add_mesh("plane", "ground", material("ground", (0.42, 0.40, 0.36)), size=30, location=(0, 0, 0))
    for i in range(args.boxes):
        size = (rng.uniform(0.6, 2.0), rng.uniform(0.6, 2.0), rng.uniform(0.8, 3.0))
        loc = (rng.uniform(-7, 7), rng.uniform(-7, 7), size[2] / 2)
        color = (rng.uniform(0.2, 0.9), rng.uniform(0.2, 0.9), rng.uniform(0.2, 0.9))
        box = add_mesh("cube", f"box_{i}", material(f"box_{i}", color), size=1.0, location=loc)
        box.scale = size

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0, math.radians(rng.uniform(0, 360)))
    scene.collection.objects.link(sun)
    sun_dir_blender = sun.rotation_euler.to_matrix() @ Vector((0, 0, -1))
    lights = [
        {
            "id": "sun",
            "type": "directional",
            "color": [1.0, 1.0, 1.0],
            "intensity": sun_data.energy,
            "direction": list(vec_zup_to_yup(tuple(sun_dir_blender))),
        }
    ]

    for a in range(args.actors):
        root = bpy.data.objects.new(f"actor_{a}", None)
        scene.collection.objects.link(root)
        shirt = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        body = add_mesh(
            "cylinder",
            f"actor_{a}_body",
            material(f"shirt_{a}", shirt),
            radius=0.22,
            depth=1.45,
            location=(0, 0, 0.725),
        )
        head = add_mesh(
            "uv_sphere",
            f"actor_{a}_head",
            material(f"skin_{a}", (0.8, 0.65, 0.55)),
            radius=0.13,
            location=(0, 0, 1.62),
        )
        for part in (body, head):
            part.parent = root
        radius, phase = rng.uniform(2.0, 5.0), rng.uniform(0, 2 * math.pi)
        speed = rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)  # m/s along the circle
        for f in range(1, args.frames + 1):
            ang = phase + speed * (f - 1) / args.fps / radius
            root.location = (radius * math.cos(ang), radius * math.sin(ang), 0.0)
            tangent = (-math.sin(ang) * speed, math.cos(ang) * speed)
            # Blender actors face their local +Y (see convert.actor_quat_zup_to_yup).
            root.rotation_euler = (0, 0, blender_actor_yaw(*tangent))
            root.keyframe_insert("location", frame=f)
            root.keyframe_insert("rotation_euler", frame=f)
        actors.append(
            {"id": f"a{a}", "object": root.name, "type": "person", "height": 1.75, "radius": 0.22}
        )

    # Camera orbits slowly around the centre, looking slightly down.
    cam_radius, cam_height, start = (
        rng.uniform(10, 14),
        rng.uniform(2.5, 5.0),
        rng.uniform(0, 2 * math.pi),
    )
    for f in range(1, args.frames + 1):
        ang = start + 0.6 * (f - 1) / max(1, args.frames - 1)
        cam.location = (cam_radius * math.cos(ang), cam_radius * math.sin(ang), cam_height)
        look_at(cam, Vector((0, 0, 0.8)))
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
    return actors, cam, lights


def convert_depth(out: Path, args: argparse.Namespace, k: dict[str, float]) -> list[str]:
    exr_dir = out / "depth_exr"
    depth_dir = out / "depth"
    depth_dir.mkdir(exist_ok=True)
    names = []
    v, u = np.mgrid[0 : args.height, 0 : args.width].astype(np.float64)
    ray_scale = np.sqrt(1 + ((u - k["cx"]) / k["fx"]) ** 2 + ((v - k["cy"]) / k["fy"]) ** 2)
    for exr in sorted(exr_dir.glob("*.exr")):
        img = bpy.data.images.load(str(exr))
        buf = np.empty(args.width * args.height * img.channels, dtype=np.float32)
        img.pixels.foreach_get(buf)
        z = buf.reshape(args.height, args.width, img.channels)[::-1, :, 0].astype(np.float64)
        bpy.data.images.remove(img)
        if DEPTH_IS_RADIAL:
            z = z / ray_scale
        z[z >= SKY_DEPTH] = 0.0
        name = f"{int(exr.stem):04d}.npy"
        np.save(depth_dir / name, z.astype(np.float32))
        names.append(name)
        exr.unlink()
    exr_dir.rmdir()
    return names


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    if args.probe:
        args.frames = 1
    actors, cam, lights = build_scene(args)
    setup_render(scene, args, out)
    bpy.ops.render.render(animation=True)

    cd = cam.data
    k = intrinsics(
        cd.lens, cd.sensor_width, cd.sensor_height, cd.sensor_fit, args.width, args.height
    )
    depth_files = convert_depth(out, args, k)

    frames, tracks = [], {a["id"]: {"times": [], "positions": [], "rotations": []} for a in actors}
    for f in range(1, args.frames + 1):
        scene.frame_set(f)
        t = round((f - 1) / args.fps, 6)
        mw = cam.matrix_world
        q = mw.to_quaternion()
        frames.append(
            {
                "frame": f,
                "t": t,
                "rgb": f"rgb/{f:04d}.png",
                "depth": f"depth/{f:04d}.npy",
                "camera": {
                    "position": list(vec_zup_to_yup(tuple(mw.translation))),
                    "rotation": list(quat_zup_to_yup((q.w, q.x, q.y, q.z))),
                },
            }
        )
        for a in actors:
            obj = bpy.data.objects[a["object"]]
            aq = obj.matrix_world.to_quaternion()
            tr = tracks[a["id"]]
            tr["times"].append(t)
            tr["positions"].append(list(vec_zup_to_yup(tuple(obj.matrix_world.translation))))
            tr["rotations"].append(list(actor_quat_zup_to_yup((aq.w, aq.x, aq.y, aq.z))))

    gt = {
        "generator": "tools/blender/gen_scene.py",
        "generator_version": GENERATOR_VERSION,
        "blender_version": bpy.app.version_string,
        "engine": args.engine,
        "seed": args.seed,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "probe": args.probe,
        "coordinates": "WorldReel: right-handed, +Y up, metres; cameras look along -Z",
        "depth": "planar metres along camera -Z, float32 .npy, 0 = no surface (sky)",
        "intrinsics": k,
        "frames": frames,
        "depth_files": depth_files,
        "actors": [
            {**{k2: v for k2, v in a.items() if k2 != "object"}, "track": tracks[a["id"]]}
            for a in actors
        ],
        "lights": lights,
        "background": [0.55, 0.68, 0.85],
    }
    (out / "gt.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"WORLDREEL_GEN_OK {out}")


if __name__ == "__main__":
    main()
