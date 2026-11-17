"""Adapt the Vurtego art asset to Small V4 Pro dimensions, Y-up -> Z-up.

Original art: RoboHoloclone12, CC BY 4.0; see docs/ASSETS.md. Output: one OBJ per
part, the label texture, and manifest.json with measured dimensions. Physics lives
in pogo.py; meshes are appearance only.
"""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import trimesh

SOURCE_SHA256 = "1687a9389e671a7fa38b147bd3b2a620c97294dac3ecdb9a4cf309128a18df18"
MOVING_PARTS = {"Slide_Shaft_Silver_MAT_0", "Grip_Tip_Rubber_Mat_0"}
PEDAL = "Traction_Pegs_Rubber_Mat1_0"
CYLINDER = "Cylinder_Base_blinn1_0"
TIP = "Grip_Tip_Rubber_Mat_0"
GRIPS = ["Grip_003_Rubber_Mat_0", "Grip_005_Rubber_Mat_0"]
# glTF: X across handles, Y up. Simulation: X forward, Y left, Z up.
AXES = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def extract_archive(archive: Path, output: Path):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            "Unexpected archive: review provenance before importing a new revision"
        )
    original = output / "original"
    original.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if (
                not (original / entry.filename)
                .resolve()
                .is_relative_to(original.resolve())
            ):
                raise ValueError("Archive contains a path outside the destination")
        z.extractall(original)
    if archive.resolve() != (output / "source.zip").resolve():
        shutil.copyfile(archive, output / "source.zip")
    return original


def _numbers(values):
    return " ".join(f"{v:.10g}" for v in values)


def _obj(mesh, path):
    """Preserve per-vertex UVs/normals, without relying on an external MTL loader."""
    lines = ["# Adapted from Vurtego Pogo Stick Model by RoboHoloclone12, CC BY 4.0"]
    lines += ["v " + _numbers(v) for v in mesh.vertices]
    lines += ["vt " + _numbers(v) for v in mesh.visual.uv]
    lines += ["vn " + _numbers(v) for v in mesh.vertex_normals]
    lines += [
        "f " + " ".join(f"{i + 1}/{i + 1}/{i + 1}" for i in face) for face in mesh.faces
    ]
    path.write_text("\n".join(lines) + "\n")


def prepare(original: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    scene = trimesh.load(original / "scene.gltf", force="scene", process=False)
    meshes = {}
    for name in scene.graph.nodes_geometry:
        transform, geom = scene.graph[name]
        mesh = scene.geometry[geom].copy()
        mesh.apply_transform(transform)
        meshes[name] = mesh
    required = MOVING_PARTS | {PEDAL, CYLINDER, *GRIPS}
    if not required.issubset(meshes):
        raise ValueError(f"Missing named parts: {required - meshes.keys()}")
    # Frame origin at the top of the traction pegs, on the cylinder axis.
    pedal_top = float(meshes[PEDAL].bounds[1, 1])
    origin = np.array([0.0, pedal_top, 0.0])
    transformed = {}
    material_ids = {
        p["name"]: p["primitives"][0]["material"]
        for p in json.loads((original / "scene.gltf").read_text())["meshes"]
    }
    # Start with the v2 cross-section: keep rubber grips exactly circular and
    # unchanged for the settled-grasp fixture. Overall axial scaling alone would
    # leave the pedals BELOW the tip at 432 mm compression, so locate each part
    # from dimensional anchors instead. See ASSETS.md for the drawing convention.
    for name, mesh in meshes.items():
        transform = np.eye(4)
        transform[:3, :3] = AXES * 5.2
        transform[:3, 3] = -AXES @ origin * 5.2
        mesh.apply_transform(transform)
        transformed[name] = mesh

    bar = transformed["Handles_Silver_MAT_0"]
    transverse = 0.022225 / np.ptp(bar.bounds[:, 0])
    grip_radius = np.ptp(transformed[GRIPS[0]].bounds[:, 0]) / 2
    tube_width = np.ptp(transformed[CYLINDER].bounds[:, 0])
    tip_insertion = (
        transformed["Slide_Shaft_Silver_MAT_0"].bounds[0, 2]
        - transformed[TIP].bounds[0, 2]
    )
    grip_z = 0.762 - grip_radius  # 30 in drawing anchor: upper grip surface
    tip_bottom = -(1.2446 - 0.762)  # 49 in extended, 19 in below pedal top
    tube_top = (
        0.762
        - np.ptp(transformed["Handle_Base_Rubber_Mat1_0"].bounds[:, 2]) * transverse
        + 0.02
    )

    def resize_z(mesh, low, high):
        lo, hi = mesh.bounds[:, 2]
        mesh.vertices[:, 2] = low + (mesh.vertices[:, 2] - lo) * (high - low) / (
            hi - lo
        )

    for name, mesh in transformed.items():
        old_z = mesh.bounds[:, 2].copy()
        if name in GRIPS:
            mesh.vertices[:, 1] *= transverse  # length/span, not cross-section
            mesh.vertices[:, 2] += grip_z - old_z.mean()
        else:
            mesh.apply_scale(transverse)
            if name == "Handles_Silver_MAT_0":
                mesh.vertices[:, 2] += grip_z - mesh.bounds[:, 2].mean()
            elif name == "Handle_Base_Rubber_Mat1_0":
                mesh.vertices[:, 2] += 0.762 - mesh.bounds[1, 2]
            elif name == CYLINDER:
                # 76.2 mm bore + estimated 2 mm wall; bar OD is independent.
                mesh.vertices[:, :2] *= 0.0802 / np.ptp(mesh.bounds[:, 0])
                resize_z(mesh, -0.0254, tube_top)
            elif name == PEDAL:
                resize_z(mesh, -0.0254, 0.0)  # diagram's 1 in pedal thickness
            elif name == TIP:
                mesh.vertices[:, 2] += tip_bottom - mesh.bounds[0, 2]
            elif name == "Slide_Shaft_Silver_MAT_0":
                # Preserve insertion into the tip; at full stroke the shaft
                # remains inside the tube, below the cap (15 mm allowance).
                resize_z(
                    mesh,
                    tip_bottom + tip_insertion * transverse,
                    tube_top - 0.432 - 0.015,
                )
            else:  # small decorative buttons on the cylinder
                mesh.vertices[:, :2] *= 0.0802 / (tube_width * transverse)
        # In-place vertex edits invalidate trimesh's cached normals/bounds.
        _obj(mesh, output / (name + ".obj"))
    shutil.copyfile(
        original / "textures/blinn1_baseColor.png", output / "basecolor.png"
    )
    shutil.copyfile(original / "license.txt", output / "LICENSE.txt")
    bounds = {n: m.bounds for n, m in transformed.items()}
    all_vertices = np.concatenate([m.vertices for m in transformed.values()])
    all_bounds = np.array([all_vertices.min(0), all_vertices.max(0)])
    grips = {
        "left" if transformed[n].centroid[1] > 0 else "right": bounds[n].mean(0)
        for n in GRIPS
    }
    cyl = bounds[CYLINDER]
    peg = bounds[PEDAL]
    tip = bounds[TIP]
    inner_y = float(cyl[1, 1]) + 0.004
    pedal_half = np.array(
        [
            (peg[1, 0] - peg[0, 0]) / 2,
            (peg[1, 1] - inner_y) / 2,
            (peg[1, 2] - peg[0, 2]) / 2,
        ]
    )
    pedal_centres = {
        side: np.array([0.0, sign * (inner_y + pedal_half[1]), -pedal_half[2]])
        for side, sign in [("left", 1), ("right", -1)]
    }
    metadata = {
        "source_archive_sha256": SOURCE_SHA256,
        "source": "https://sketchfab.com/3d-models/vurtego-pogo-stick-model-01b521f3c511437da40224ecaa7658fc",
        "author": "RoboHoloclone12",
        "license": "CC-BY-4.0",
        "adaptation": "small-v4-pro-v3",
        "equivalent_axial_scale": 1.2446 / scene.extents[1],
        "transverse_scale": float(5.2 * transverse),
        "rubber_grip_cross_section_scale": 5.2,
        "handle_to_pedal_drawing_m": 0.762,
        "metal_bar_od_m": float(np.ptp(bar.bounds[:, 0])),
        "tube_wall_estimate_m": 0.002,
        "source_scene_extent_m": scene.extents.tolist(),
        "uncompressed_height_m": float(np.ptp(all_bounds[:, 2])),
        "overall_width_m": float(np.ptp(all_bounds[:, 1])),
        "pedal_depth_m": float(2 * pedal_half[0]),
        "pedal_width_each_m": float(2 * pedal_half[1]),
        "grip_span_m": float(grips["left"][1] - grips["right"][1]),
        "grip_height_above_pedal_m": float(grips["left"][2]),
        "pedal_top_above_tip_bottom_m": float(-tip[0, 2]),
        "mesh_count": len(meshes),
        "triangles": sum(len(m.faces) for m in meshes.values()),
        "animations_in_archive": 0,
        "moving_parts": sorted(MOVING_PARTS),
        "grips": {k: v.tolist() for k, v in grips.items()},
        "pedal_centres": {k: v.tolist() for k, v in pedal_centres.items()},
        "pedal_half_size": pedal_half.tolist(),
        "bounds": {k: v.tolist() for k, v in bounds.items()},
        "materials": {n: int(i) for n, i in material_ids.items()},
    }
    metadata["generated_files_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.suffix in [".obj", ".png"]
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
