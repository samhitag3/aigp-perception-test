from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-out", type=str, default="data/isaac_synth_raw")
    parser.add_argument("--num-frames", type=int, default=1000)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")

    # default to AI Grand Prix gate dimension
    parser.add_argument("--gate-width", type=float, default=2.7)
    parser.add_argument("--gate-height", type=float, default=2.7)
    parser.add_argument("--bar-thickness", type=float, default=0.6)
    parser.add_argument("--bar-depth", type=float, default=0.26)

    parser.add_argument("--num-clutter", type=int, default=20)
    parser.add_argument("--occlusion-prob", type=float, default=0.18)
    parser.add_argument("--far-gate-prob", type=float, default=0.20)
    parser.add_argument("--edge-case-prob", type=float, default=0.20)

    return parser.parse_args()


args = parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(launch_config={"headless": args.headless})

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.semantics import add_update_semantics  # noqa: E402
from omni.replicator.core import BackendDispatch, Writer, WriterRegistry  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdShade  # noqa: E402


class GateRawWriter(Writer):
    """Write RGB plus raw semantic ID arrays with idToLabels metadata."""

    def __init__(self, output_dir: str, frame_padding: int = 6):
        self._frame_id = 0
        self._frame_padding = int(frame_padding)
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self.annotators = [
            rep.AnnotatorRegistry.get_annotator("rgb"),
            rep.AnnotatorRegistry.get_annotator(
                "semantic_segmentation",
                init_params={"colorize": False},
            ),
        ]

    def write(self, data):
        frame = f"{self._frame_id:0{self._frame_padding}d}"

        rgb = data["rgb"]
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        self._backend.write_image(f"rgb/rgb_{frame}.png", rgb)

        sem = data["semantic_segmentation"]["data"]
        height, width = sem.shape[:2]
        sem = sem.view(np.uint32).reshape(height, width)

        self._backend.write_image(f"semantic/semantic_{frame}.png", sem)

        npy_buf = io.BytesIO()
        np.save(npy_buf, sem)
        self._backend.write_blob(
            f"semantic/semantic_{frame}.npy",
            npy_buf.getvalue(),
        )

        labels = data["semantic_segmentation"]["info"].get("idToLabels", {})
        json_buf = io.BytesIO()
        json_buf.write(
            json.dumps({str(k): v for k, v in labels.items()}, indent=2).encode("utf-8")
        )
        self._backend.write_blob(
            f"semantic/semantic_labels_{frame}.json",
            json_buf.getvalue(),
        )

        self._frame_id += 1

    def on_final_frame(self):
        self._backend.sync_pending_paths()


def make_material(stage, path: str, color):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse.Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material, diffuse


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def define_cube(stage, path: str, translate, scale, material=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()

    xform = UsdGeom.Xformable(prim)
    translate_op = xform.AddTranslateOp()
    scale_op = xform.AddScaleOp()
    translate_op.Set(Gf.Vec3d(*translate))
    scale_op.Set(Gf.Vec3f(*scale))

    if material is not None:
        bind_material(prim, material)

    return prim, translate_op, scale_op


def create_gate(stage, args, gate_material):
    root = stage.DefinePrim("/World/Gate", "Xform")
    xform = UsdGeom.Xformable(root)
    translate_op = xform.AddTranslateOp()
    rotate_op = xform.AddRotateXYZOp()

    w = args.gate_width
    h = args.gate_height
    t = args.bar_thickness
    d = args.bar_depth

    bars = [
        ("Left", (-w / 2 + t / 2, 0.0, 0.0), (t, d, h)),
        ("Right", (w / 2 - t / 2, 0.0, 0.0), (t, d, h)),
        ("Top", (0.0, 0.0, h / 2 - t / 2), (w, d, t)),
        ("Bottom", (0.0, 0.0, -h / 2 + t / 2), (w, d, t)),
    ]

    for name, local_pos, local_scale in bars:
        prim, _, _ = define_cube(
            stage,
            f"/World/Gate/{name}",
            local_pos,
            local_scale,
            gate_material,
        )
        add_update_semantics(prim, "gate", "class")

    return {
        "root": root,
        "translate": translate_op,
        "rotate": rotate_op,
    }


def create_scene(stage, args):
    looks = stage.DefinePrim("/World/Looks", "Scope")

    gate_mat, gate_color_input = make_material(stage, "/World/Looks/GateMat", (1.0, 0.35, 0.02))
    floor_mat, floor_color_input = make_material(stage, "/World/Looks/FloorMat", (0.35, 0.35, 0.35))
    wall_mat, _ = make_material(stage, "/World/Looks/WallMat", (0.55, 0.58, 0.62))
    clutter_mat, clutter_color_input = make_material(stage, "/World/Looks/ClutterMat", (0.2, 0.2, 0.2))

    define_cube(stage, "/World/Floor", (0.0, 2.0, -0.05), (12.0, 14.0, 0.10), floor_mat)
    define_cube(stage, "/World/BackWall", (0.0, 6.0, 2.0), (12.0, 0.12, 4.0), wall_mat)
    define_cube(stage, "/World/LeftWall", (-6.0, 2.0, 2.0), (0.12, 14.0, 4.0), wall_mat)
    define_cube(stage, "/World/RightWall", (6.0, 2.0, 2.0), (0.12, 14.0, 4.0), wall_mat)

    gate = create_gate(stage, args, gate_mat)

    clutter = []
    for idx in range(args.num_clutter):
        prim, translate_op, scale_op = define_cube(
            stage,
            f"/World/Clutter/Box_{idx:03d}",
            (50.0, 50.0, 50.0),
            (0.2, 0.2, 0.2),
            clutter_mat,
        )
        clutter.append((translate_op, scale_op))

    dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_intensity = dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float)
    dome_color = dome.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f)

    sun = stage.DefinePrim("/World/Sun", "DistantLight")
    sun_intensity = sun.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float)
    sun_angle = sun.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float)

    return {
        "gate": gate,
        "gate_color": gate_color_input,
        "floor_color": floor_color_input,
        "clutter_color": clutter_color_input,
        "clutter": clutter,
        "dome_intensity": dome_intensity,
        "dome_color": dome_color,
        "sun_intensity": sun_intensity,
        "sun_angle": sun_angle,
    }


def set_camera_look_at(transform_op, eye, target):
    eye_v = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
    target_v = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
    up_v = Gf.Vec3d(0.0, 0.0, 1.0)

    view = Gf.Matrix4d(1.0)
    view.SetLookAt(eye_v, target_v, up_v)
    transform_op.Set(view.GetInverse())


def create_camera(stage, args):
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(8.0)
    camera.CreateHorizontalApertureAttr(12.0)
    camera.CreateVerticalApertureAttr(12.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))

    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    transform_op = xform.AddTransformOp()

    set_camera_look_at(transform_op, (0.0, -4.0, 1.4), (0.0, 0.0, 1.2))
    return camera, transform_op


def random_color(rng, lo=0.15, hi=1.0):
    return Gf.Vec3f(
        rng.uniform(lo, hi),
        rng.uniform(lo, hi),
        rng.uniform(lo, hi),
    )


def randomize_frame(state, camera_transform_op, rng, args):
    far_case = rng.random() < args.far_gate_prob
    edge_case = rng.random() < args.edge_case_prob

    gate_x = rng.uniform(-0.35, 0.35)
    gate_y = rng.uniform(-0.20, 0.35)
    gate_z = rng.uniform(0.95, 1.35)

    gate_rx = rng.uniform(-10.0, 10.0)
    gate_ry = rng.uniform(-18.0, 18.0)
    gate_rz = rng.uniform(-35.0, 35.0)

    state["gate"]["translate"].Set(Gf.Vec3d(gate_x, gate_y, gate_z))
    state["gate"]["rotate"].Set(Gf.Vec3f(gate_rx, gate_ry, gate_rz))

    if far_case:
        distance = rng.uniform(5.0, 9.0)
    else:
        distance = rng.uniform(1.8, 5.0)

    cam_x = gate_x + rng.uniform(-1.4, 1.4)
    cam_y = gate_y - distance
    cam_z = rng.uniform(0.45, 2.4)

    if edge_case:
        target_x = gate_x + rng.uniform(-0.8, 0.8)
        target_z = gate_z + rng.uniform(-0.55, 0.55)
    else:
        target_x = gate_x + rng.uniform(-0.18, 0.18)
        target_z = gate_z + rng.uniform(-0.18, 0.18)

    set_camera_look_at(
        camera_transform_op,
        (cam_x, cam_y, cam_z),
        (target_x, gate_y, target_z),
    )

    gate_colors = [
        (1.0, 0.32, 0.02),
        (0.05, 0.35, 1.0),
        (0.95, 0.95, 0.08),
        (0.95, 0.10, 0.12),
        (0.85, 0.85, 0.85),
    ]
    state["gate_color"].Set(Gf.Vec3f(*rng.choice(gate_colors)))
    state["floor_color"].Set(random_color(rng, 0.20, 0.65))
    state["clutter_color"].Set(random_color(rng, 0.05, 0.95))

    state["dome_intensity"].Set(rng.uniform(80.0, 900.0))
    state["dome_color"].Set(random_color(rng, 0.65, 1.0))
    state["sun_intensity"].Set(rng.uniform(100.0, 1400.0))
    state["sun_angle"].Set(rng.uniform(0.2, 3.0))

    for translate_op, scale_op in state["clutter"]:
        if rng.random() > 0.55:
            translate_op.Set(Gf.Vec3d(50.0, 50.0, 50.0))
            continue

        if rng.random() < args.occlusion_prob:
            x = gate_x + rng.uniform(-0.75, 0.75)
            y = gate_y - rng.uniform(0.25, 1.40)
            z = gate_z + rng.uniform(-0.55, 0.55)
            sx = rng.uniform(0.06, 0.28)
            sy = rng.uniform(0.06, 0.28)
            sz = rng.uniform(0.15, 0.65)
        else:
            x = rng.uniform(-4.8, 4.8)
            y = rng.uniform(-1.5, 5.5)
            z = rng.uniform(0.15, 1.6)
            sx = rng.uniform(0.08, 0.55)
            sy = rng.uniform(0.08, 0.55)
            sz = rng.uniform(0.08, 1.2)

        translate_op.Set(Gf.Vec3d(x, y, z))
        scale_op.Set(Gf.Vec3f(sx, sy, sz))


def main():
    raw_out = Path(args.raw_out)
    raw_out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    rng = random.Random(args.seed)
    rep.set_global_seed(args.seed)

    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)

    stage = omni.usd.get_context().get_stage()
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

    state = create_scene(stage, args)
    camera, camera_transform_op = create_camera(stage, args)

    WriterRegistry.register(GateRawWriter)
    writer = WriterRegistry.get("GateRawWriter")
    writer.initialize(output_dir=str(raw_out), frame_padding=6)

    render_product = rep.create.render_product(
        camera.GetPath(),
        (args.width, args.height),
        name="GateCamera",
    )
    writer.attach([render_product])

    meta = {
        "generator": "isaac_generate_gate_synth.py",
        "isaac_sim_version": "4.5.0",
        "semantic_class": "gate",
        "width": args.width,
        "height": args.height,
        "num_frames_requested": args.num_frames,
        "seed": args.seed,
        "gate_width": args.gate_width,
        "gate_height": args.gate_height,
        "bar_thickness": args.bar_thickness,
        "bar_depth": args.bar_depth,
    }
    (raw_out / "meta.json").write_text(json.dumps(meta, indent=2))

    for frame_idx in range(args.num_frames):
        randomize_frame(state, camera_transform_op, rng, args)
        rep.orchestrator.step()

        if (frame_idx + 1) % 100 == 0:
            print(f"rendered {frame_idx + 1}/{args.num_frames}")

    writer.detach()
    render_product.destroy()
    rep.orchestrator.wait_until_complete()
    simulation_app.update()
    simulation_app.close()

    print(f"wrote raw Isaac dataset to {raw_out}")


if __name__ == "__main__":
    main()
