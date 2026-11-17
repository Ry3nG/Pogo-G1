"""Studio render of exported poses in Blender (clay look, transparent PNG with a
soft contact shadow): stills in Cycles, clips in EEVEE.

    B=blender
    $B -b --factory-startup -P scripts/render/blender_scene.py -- \
        artifacts/render/scene.npz hero apex OUT/hero.png
    $B -b --factory-startup -P scripts/render/blender_scene.py -- \
        artifacts/render/scene.npz hop touchdown,compressed,takeoff,rising,apex,landing OUT/hop-{}.png

"hero" renders one pose in three-quarter view at 2000 x 3000; "hop" renders
several poses with one fixed side camera framed on all of them, 1200 px tall,
so they share scale and ground line ({} becomes 1, 2, ...). Optional trailing
arguments: samples (default 384) and a resolution percentage for test renders.
Blender 4.5 LTS (portable tarball); OPTIX, else CUDA; denoising on the CPU.

Clips (scene.npz from export_poses.py clip/replay; FRAMES is "all" or "a:b"):

    $B -b --factory-startup -P scripts/render/blender_scene.py -- \
        artifacts/render/opening/scene.npz opening all artifacts/render/opening/png
    for r in 1012-0002-vel-obs-s0 1015-1133-step-steer-s0 1025-1621-centroidal-s0; do
      $B -b --factory-startup -P scripts/render/blender_scene.py -- \
          artifacts/render/turn/$r/scene.npz turn all artifacts/render/turn/$r/png
    done
    for l in N U S; do
      $B -b --factory-startup -P scripts/render/blender_scene.py -- \
          artifacts/render/nus-$l-hold/scene.npz nus all artifacts/render/nus-$l-hold/png
    done

"opening" is a three-quarter view that follows the rider's position and
heading, framed on its recent height, at 1920 x 1080. "terrain" is the same
view, a little higher and framed on the local ground, for scenes exported with
a terrain mesh (export_poses.py SET="terrain=..."): the terrain is drawn in a
light matte material that fades out 3-6 m from the rider, so it ends in the
background, and it takes the contact shadow itself. "turn" looks down at 45
degrees with an orientation fixed in the world (the camera only translates
with the rider), at 640 x 1080, and draws a ring under the rider with an arc
from the starting heading to the current one and a needle at the current
heading. "nus" is a fixed high oblique view from the foot of one letter (pale
band) with the hopped path (pen down, #2a78d6) growing, at 640 x 1080. Each
"side" and "hands" (failure clips, 960 x 1080 panels for a pair side by side)
look from a fixed three-quarter direction and translate with the pelvis:
"side" frames 3.0 m of height for 1 m hops, "hands" 2.55 m from the front, so
the hands on the handle read. "turn" takes PANEL_W from the environment
(default 640). Each frame is written twice as transparent PNG: OUT/rider/NNNN.png (robot and
contact shadow, AgX) and OUT/marks/NNNN.png (ground marks, Standard view
transform, so their colours are exact). Optional trailing arguments: samples
(default 64) and the floor radiance seen by the subject (default 0.8). EEVEE
has no shadow catcher: light linking keeps the studio lights off the ground,
whose material turns the top light's shadow into alpha, and a bright lower
world hemisphere stands in for the stills' bouncing floor. Composite over the
near-white background and encode (25 fps), e.g. for a full-frame clip or a
panel (s=640x1080):

    ffmpeg -y -f lavfi -i color=c=0xf4f4f2:s=1920x1080:r=25 \
      -framerate 25 -i OUT/marks/%04d.png -framerate 25 -i OUT/rider/%04d.png \
      -filter_complex "[0][1]overlay=shortest=1[a];[a][2]overlay,format=yuv420p" \
      -c:v libx264 -crf 18 -preset slow -movflags +faststart -an clip.mp4

and for three 640 x 1080 panels side by side (A, B, C):

    ffmpeg -y -f lavfi -i color=c=0xf4f4f2:s=1920x1080:r=25 \
      -framerate 25 -i A/marks/%04d.png -framerate 25 -i A/rider/%04d.png \
      -framerate 25 -i B/marks/%04d.png -framerate 25 -i B/rider/%04d.png \
      -framerate 25 -i C/marks/%04d.png -framerate 25 -i C/rider/%04d.png \
      -filter_complex "[1][3][5]hstack=3[m];[2][4][6]hstack=3[r];\
    [0][m]overlay=shortest=1[a];[a][r]overlay,format=yuv420p" \
      -c:v libx264 -crf 18 -preset slow -movflags +faststart -an turn-compare.mp4

A poster is one frame through the same filter (K = frame number):

    ffmpeg -y -f lavfi -i color=c=0xf4f4f2:s=1920x1080 -i OUT/marks/K.png \
      -i OUT/rider/K.png -filter_complex "[0][1]overlay[a];[a][2]overlay" \
      -frames:v 1 poster.png
"""

import json
import math
import os
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1 :]
npz, mode, poses, out = argv[0], argv[1], argv[2].split(","), argv[3]
clip = mode in ("opening", "terrain", "turn", "nus", "side", "hands")
samples = int(argv[4]) if len(argv) > 4 else (64 if clip else 384)
percent = int(argv[5]) if len(argv) > 5 and not clip else 100
z = np.load(npz)
geoms = [json.loads(s) for s in z["geoms"]]
if clip:
    n_all = sum(k.startswith("pos/f") for k in z.files)
    f0, f1 = (0, n_all) if poses == ["all"] else map(int, poses[0].split(":"))
    poses = [f"f{i:04d}" for i in range(n_all)]  # all frames: cameras see the clip
bpy.ops.wm.read_factory_settings(use_empty=True)

# Materials. Neutral only: the report reserves colour for data.
DARK_LINKS = (
    "hip_pitch", "hip_roll", "ankle_pitch", "waist_yaw", "waist_roll",
    "shoulder_pitch", "shoulder_roll", "wrist", "logo", "pelvis_contour",
)  # fmt: skip


def material(name, color, rough, metal=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1.0)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Coat Roughness"].default_value = 0.25
    return m


MAT = {
    "shell": material("shell", (0.70, 0.695, 0.68), 0.42, coat=0.15),
    "joint": material("joint", (0.15, 0.15, 0.155), 0.38),
    "hand": material("hand", (0.27, 0.27, 0.275), 0.5),
    "graphite": material("graphite", (0.045, 0.047, 0.05), 0.3, metal=0.5, coat=0.4),
    "steel": material("steel", (0.62, 0.62, 0.63), 0.2, metal=1.0),
    "rubber": material("rubber", (0.025, 0.025, 0.025), 0.75),
    "visor": material("visor", (0.012, 0.012, 0.014), 0.12, coat=1.0),
}


def material_of(mesh):
    part = mesh.split("/")[-1]
    if mesh.startswith("pogo/"):
        if "Silver" in part:
            return MAT["steel"]
        return MAT["graphite"] if "Cylinder" in part else MAT["rubber"]
    if any(k in part for k in ("base", "thumb", "index", "middle", "ring", "pinky")):
        return MAT["hand"]
    return MAT["joint"] if any(k in part for k in DARK_LINKS) else MAT["shell"]


def visor(me):
    """Faces more than 4 mm inside the head's convex hull get the visor material."""
    bm = bmesh.new()
    bm.from_mesh(me)
    hull = bmesh.ops.convex_hull(bm, input=bm.verts, use_existing_faces=False)
    bmesh.ops.delete(
        bm, geom=hull["geom_interior"] + hull["geom_unused"], context="VERTS"
    )
    tree = BVHTree.FromBMesh(bm)
    me.materials.append(MAT["visor"])
    for f in me.polygons:
        f.material_index = int(tree.find_nearest(f.center)[3] > 0.004)
    bm.free()


scene = bpy.context.scene
data = {}
for key in {g[0] for g in geoms}:
    bm = bmesh.new()
    for v in z[f"v/{key}"]:
        bm.verts.new(v.tolist())
    bm.verts.ensure_lookup_table()
    for f in z[f"f/{key}"]:
        try:
            bm.faces.new([bm.verts[i] for i in f])
        except ValueError:  # duplicate face
            pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    me = bpy.data.meshes.new(key)
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(35))
    me.materials.append(material_of(key))
    if key == "robot/head_link":  # the face is a recess in the mesh: glaze it dark
        visor(me)
    data[key] = me
# Stills: two view layers composited: "subject" (all lights; the ground only
# bounces light) over "shadow" (subject invisible but casting shadows, only the
# top light and the world), so the ground shows a soft contact shadow and no
# long key-light shadow. Clips: one layer; light linking keeps the studio
# lights off the ground, whose material turns the top light's shadow into alpha.
coll = {}
for n in ("subject", "ground", "studio", "top"):
    coll[n] = bpy.data.collections.new(n)
    scene.collection.children.link(coll[n])
objs = []
for key, _ in geoms:
    ob = bpy.data.objects.new(key, data[key])
    coll["subject"].objects.link(ob)
    objs.append(ob)


def pose(name):
    for ob, p, r in zip(objs, z[f"pos/{name}"], z[f"mat/{name}"], strict=True):
        ob.matrix_world = Matrix.Translation(p) @ Matrix(r.tolist()).to_4x4()
    bpy.context.view_layer.update()


def points(name, every=7):
    return np.concatenate(
        [
            p + z[f"v/{g[0]}"][::every] @ r.T
            for g, p, r in zip(geoms, z[f"pos/{name}"], z[f"mat/{name}"], strict=True)
        ]
    )


# Rider heading (pelvis x axis) and the scene centre, from the first pose.
pelvis = [g[0] for g in geoms].index("robot/pelvis")
fwd = z[f"mat/{poses[0]}"][pelvis][:, 0]
heading = math.atan2(fwd[1], fwd[0])
pts = np.concatenate([points(n, 7 if not clip else 41) for n in poses])
lo, hi = pts.min(0), pts.max(0)
centre = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0))

# Ground: shadow catcher (stills) or shadow-to-alpha material (clips).
ground = bpy.data.objects.new("ground", bpy.data.meshes.new("ground"))
ground.data.from_pydata(
    [(-40, -40, 0), (40, -40, 0), (40, 40, 0), (-40, 40, 0)], [], [(0, 1, 2, 3)]
)
ground.location = (centre.x, centre.y, 0)
coll["ground"].objects.link(ground)
SHADOW = 0.65  # contact shadow opacity
SUN = 3.0  # top light in clips (W/m^2)
DIST = 4.0  # turn panels' camera distance
FLOOR = float(argv[5]) if len(argv) > 5 and clip else 0.8  # floor radiance, clips
if not clip:
    ground.data.materials.append(material("floor", (0.8, 0.8, 0.8), 0.6))
    ground.is_shadow_catcher = True
else:
    m = bpy.data.materials.new("shadow")
    m.use_nodes = True
    m.surface_render_method = "BLENDED"
    nt = m.node_tree
    nt.nodes.clear()
    lit = nt.nodes.new("ShaderNodeBsdfDiffuse")
    rgb = nt.nodes.new("ShaderNodeShaderToRGB")
    bw = nt.nodes.new("ShaderNodeRGBToBW")
    # unshadowed ground reads 0.263 * SUN in EEVEE (measured); 1% margin
    norm = nt.nodes.new("ShaderNodeMapRange")
    norm.inputs["From Min"].default_value = 0.0
    norm.inputs["From Max"].default_value = 0.26 * SUN
    norm.inputs["To Min"].default_value = SHADOW
    norm.inputs["To Max"].default_value = 0.0
    mix = nt.nodes.new("ShaderNodeMixShader")
    black = nt.nodes.new("ShaderNodeEmission")
    black.inputs["Color"].default_value = (0, 0, 0, 1)
    nt.links.new(lit.outputs[0], rgb.inputs[0])
    nt.links.new(rgb.outputs["Color"], bw.inputs[0])
    nt.links.new(bw.outputs[0], norm.inputs["Value"])
    nt.links.new(norm.outputs[0], mix.inputs[0])
    nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
    nt.links.new(black.outputs[0], mix.inputs[2])
    nt.links.new(
        mix.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"]
    )
    ground.data.materials.append(m)
    ground.visible_shadow = False
FADE = []  # x, y of the terrain's fade centre (value nodes), set per frame
if "terrain_v" in z.files:  # the terrain replaces the flat ground
    ground.hide_render = True
    me = bpy.data.meshes.new("terrain")
    me.from_pydata(z["terrain_v"].tolist(), [], z["terrain_f"].tolist())
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    m = material("terrain", (0.74, 0.74, 0.72), 0.9)
    m.surface_render_method = "DITHERED"
    nt = m.node_tree
    # a faint 0.5 m checker: the grid follows the surface, so slopes read
    check = nt.nodes.new("ShaderNodeTexChecker")
    check.inputs["Scale"].default_value = 2.0
    check.inputs["Color1"].default_value = (0.62, 0.62, 0.6, 1)
    check.inputs["Color2"].default_value = (0.55, 0.55, 0.535, 1)
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], check.inputs[0])
    nt.links.new(
        check.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"]
    )
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(
        nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"], sep.inputs[0]
    )
    d2 = None
    for axis in ("X", "Y"):
        FADE.append(nt.nodes.new("ShaderNodeValue"))
        diff = nt.nodes.new("ShaderNodeMath")
        diff.operation = "SUBTRACT"
        nt.links.new(sep.outputs[axis], diff.inputs[0])
        nt.links.new(FADE[-1].outputs[0], diff.inputs[1])
        sq = nt.nodes.new("ShaderNodeMath")
        sq.operation = "POWER"
        sq.inputs[1].default_value = 2.0
        nt.links.new(diff.outputs[0], sq.inputs[0])
        if d2 is None:
            d2 = sq
        else:
            add = nt.nodes.new("ShaderNodeMath")
            nt.links.new(d2.outputs[0], add.inputs[0])
            nt.links.new(sq.outputs[0], add.inputs[1])
            d2 = add
    fade = nt.nodes.new("ShaderNodeMapRange")
    fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = 9, 36
    nt.links.new(d2.outputs[0], fade.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMixShader")
    out_node = nt.nodes["Material Output"]
    nt.links.new(fade.outputs[0], mix.inputs[0])
    nt.links.new(nt.nodes["Principled BSDF"].outputs[0], mix.inputs[1])
    nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out_node.inputs["Surface"])
    me.materials.append(m)
    terrain = bpy.data.objects.new("terrain", me)
    coll["subject"].objects.link(terrain)


def aim(ob, target):
    ob.rotation_euler = (target - ob.location).to_track_quat("-Z", "Y").to_euler()


LIGHTS = []  # (object, azimuth from the camera, distance, height, target z offset)


def area(
    name, az, dist, height, size, power, dz, shape="DISK", size_y=None, shadow=True
):
    li = bpy.data.lights.new(name, "AREA")
    li.shape, li.size, li.energy = shape, size, power
    li.use_shadow = shadow
    if size_y:
        li.size_y = size_y
    ob = bpy.data.objects.new(name, li)
    coll["studio"].objects.link(ob)
    LIGHTS.append((ob, az, dist, height, dz))
    return ob


def place_lights(cam_az, c, mid):
    """Lights relative to the camera's world azimuth (radians) about c."""
    for ob, az, dist, height, dz in LIGHTS:
        a = cam_az + math.radians(az)
        ob.location = c + Vector((dist * math.cos(a), dist * math.sin(a), height))
        aim(ob, c + Vector((0, 0, mid + dz)))


# Camera: azimuth from the rider's front, lens, height; distance fits the poses'
# height (plus the ground) into FILL of the frame. Lights are placed relative
# to the camera, so all views get the same look.
hero = mode != "hop"
az, lens, cam_z, fill, key_az = (
    (62, 85, 1.25, 0.9, -112) if hero else (80, 135, 1.3, 0.9, -55)
)
area("key", key_az, 3.0, 4.6, 2.0, 240, 0)  # big soft key, high, from the side
strip = area("strip", -40, 3.0, 0, 0.5, 120, 0, "RECTANGLE", 2.5, shadow=False)
strip.visible_diffuse = False  # a softbox seen only in reflections: lines the metal
area("fill", 18, 5.0, 1.4, 5.0, 14, 0, shadow=False)  # broad low fill
area("rim", 93, 3.5, 3.0, 0.6, 220, 0.4, "RECTANGLE", 3.0, shadow=False)
bottom = min(lo[2], 0.0)
span = hi[2] - bottom + 0.06
mid = (hi[2] + bottom) / 2
LIGHTS[1] = (*LIGHTS[1][:3], mid, 0)  # the strip sits at the subject's middle
r = scene.render
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.lens = lens
cam.data.sensor_fit, cam.data.sensor_height = "VERTICAL", 36
if not clip:
    if hero:
        r.resolution_x, r.resolution_y = 2000, 3000
    else:  # width just fits the poses' lateral extent, so the frames sit in a row
        a = heading + math.radians(az)
        lateral = np.abs(
            (pts[:, :2] - np.array(centre[:2])) @ [-math.sin(a), math.cos(a)]
        )
        r.resolution_y = 1200
        r.resolution_x = 2 * round(600 * (2 * lateral.max() + 0.5) * fill / span)
    a = heading + math.radians(az)
    dist = span / 2 / fill / (18 / lens)
    cam.location = centre + Vector((dist * math.cos(a), dist * math.sin(a), cam_z))
    aim(cam, centre + Vector((0, 0, mid)))
    place_lights(a, centre, mid)
    top = bpy.data.lights.new("top", "AREA")
    top.shape, top.size, top.energy = "DISK", 2.5, 500
    ob = bpy.data.objects.new("top", top)
    coll["top"].objects.link(ob)
    ob.location = centre + Vector((0, 0, 12.0))  # also the contact shadow
    aim(ob, centre)
else:  # a sun straight down, as soft as the stills' 2.5 m disk at 12 m
    top = bpy.data.lights.new("top", "SUN")
    top.energy, top.angle = SUN, math.radians(12)
    top.use_shadow_jitter = True
    coll["top"].objects.link(bpy.data.objects.new("top", top))
    for ob, *_ in LIGHTS:  # studio lights light the subject only
        ob.light_linking.receiver_collection = coll["subject"]
world = bpy.data.worlds.new("studio")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.03
scene.world = world
if clip:  # the stills' lit floor, as the subject sees it: a bright lower hemisphere
    nt = world.node_tree
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    hemi = nt.nodes.new("ShaderNodeMapRange")
    hemi.inputs["From Min"].default_value, hemi.inputs["From Max"].default_value = (
        -0.15,
        0.05,
    )
    hemi.inputs["To Min"].default_value, hemi.inputs["To Max"].default_value = (
        FLOOR,
        0.03,
    )
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], hemi.inputs["Value"])
    nt.links.new(hemi.outputs[0], nt.nodes["Background"].inputs["Strength"])

r.resolution_percentage = percent
r.film_transparent = True
r.image_settings.file_format = "PNG"
r.image_settings.color_mode = "RGBA"
r.image_settings.color_depth = "16" if not clip else "8"
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"

if clip:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    e = scene.eevee
    e.taa_render_samples = samples
    e.use_raytracing = True
    e.ray_tracing_options.resolution_scale = "1"
    e.use_fast_gi = True
    e.fast_gi_method = "GLOBAL_ILLUMINATION"
    e.use_shadows = True
    e.shadow_ray_count, e.shadow_step_count = 2, 12
    for ob, *_ in LIGHTS:
        ob.data.use_shadow_jitter = ob.data.use_shadow
else:
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(d.type == backend for d in prefs.devices):
                break
        except TypeError:
            continue
    for d in prefs.devices:
        d.use = d.type != "CPU"
    c = scene.cycles
    c.device = "GPU"
    c.samples = samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.01
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.denoising_use_gpu = False  # keep GPU memory for the path tracer on a shared card
    c.max_bounces = 8
    c.tile_size = 1024
    print("device:", prefs.compute_device_type)

    subject = scene.view_layers[0]
    subject.name = "subject"
    subject.layer_collection.children["ground"].indirect_only = True
    shadow = scene.view_layers.new("shadow")
    shadow.layer_collection.children["subject"].indirect_only = True
    shadow.layer_collection.children["studio"].exclude = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = []
    for vl in (shadow, subject):
        n = tree.nodes.new("CompositorNodeRLayers")
        n.layer = vl.name
        layers.append(n)
    over = tree.nodes.new("CompositorNodeAlphaOver")
    # The shadow layer is black with alpha = shadow: rebuild it with alpha * SHADOW.
    split = tree.nodes.new("CompositorNodeSeparateColor")
    fade = tree.nodes.new("CompositorNodeMath")
    fade.operation, fade.inputs[1].default_value = "MULTIPLY", SHADOW
    join = tree.nodes.new("CompositorNodeCombineColor")
    tree.links.new(layers[0].outputs["Image"], split.inputs[0])
    tree.links.new(split.outputs["Alpha"], fade.inputs[0])
    tree.links.new(fade.outputs[0], join.inputs["Alpha"])
    tree.links.new(join.outputs[0], over.inputs[1])
    tree.links.new(layers[1].outputs["Image"], over.inputs[2])
    tree.links.new(
        over.outputs["Image"], tree.nodes.new("CompositorNodeComposite").inputs["Image"]
    )

    for k, name in enumerate(poses, 1):
        pose(name)
        r.filepath = out.replace("{}", str(k))
        bpy.ops.render.render(write_still=True)
        print("wrote", r.filepath)
    sys.exit(0)

# ---------------------------------------------------------------- clips
fps = float(z["fps"])
T = len(poses)


def smooth(x, sigma):
    """Zero-phase Gaussian smoothing along axis 0 (sigma in frames)."""
    h = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], h, 0), x, np.repeat(x[-1:], h, 0)])
    return np.stack(
        [np.convolve(c, k, "valid") for c in pad.reshape(len(pad), -1).T], 1
    ).reshape(x.shape)


P = np.array([z[f"pos/{n}"][pelvis] for n in poses])  # pelvis per frame
yaw = np.unwrap(
    [
        math.atan2(z[f"mat/{n}"][pelvis][1, 0], z[f"mat/{n}"][pelvis][0, 0])
        for n in poses
    ]
)
top_z = np.array([points(n, 41)[:, 2].max() for n in poses])


def ribbon(xy, width, z0=0.002):
    """Flat strip of WIDTH along polyline XY (N, 2) at height z0."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return [], []
    t = np.gradient(xy, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1) * width / 2
    v = np.concatenate([xy + nrm, xy - nrm])
    v = np.concatenate([v, np.full((len(v), 1), z0)], 1)
    n = len(xy)
    f = [(i, i + 1, n + i + 1, n + i) for i in range(n - 1)]
    return v.tolist(), f


marks = bpy.data.scenes.new("marks")
marks.camera = cam
marks.collection.objects.link(cam)
marks.render.engine = "BLENDER_EEVEE_NEXT"
marks.eevee.taa_render_samples = 16
mr = marks.render
mr.resolution_percentage = percent
mr.film_transparent = True
mr.image_settings.file_format = "PNG"
mr.image_settings.color_mode = "RGBA"
marks.view_settings.view_transform = "Standard"
marks.world = None


def flat(name, hex_rgb):
    """Unlit colour, exact under the Standard view transform."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    c = [int(hex_rgb[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*lin, 1)
    nt.links.new(em.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs[0])
    return m


MARKS = {}


def mark(name, mat, geometry):
    """Replace mark NAME's mesh with GEOMETRY = (verts, faces)."""
    ob = MARKS.get(name)
    if ob is None:
        ob = MARKS[name] = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        marks.collection.objects.link(ob)
    old = ob.data
    me = bpy.data.meshes.new(name)
    me.from_pydata(geometry[0], [], geometry[1])
    me.materials.append(mat)
    ob.data = me
    bpy.data.meshes.remove(old)


W, H = 1920, 1080
if mode in ("opening", "terrain"):  # three-quarter view, follows the heading
    c_xy = smooth(P[:, :2], 0.4 * fps)
    win = round(1.0 * fps)
    tip_g = [g[0] for g in geoms].index("pogo/pogo_Grip_Tip_Rubber_Mat_0")
    tip_z = np.array([z[f"pos/{n}"][tip_g][2] for n in poses])
    ground_z = (  # local ground: the lowest tip nearby (0 on flat ground)
        smooth(
            np.array([tip_z[max(i - win, 0) : i + win].min() for i in range(T)]), fps
        )
        if mode == "terrain"
        else np.zeros(T)
    )
    hi_z = smooth(
        np.array([top_z[max(i - win, 0) : i + win].max() for i in range(T)]), 0.6 * fps
    )
    sweep = math.radians(-30) if mode == "opening" else 0.0
    cam_az = smooth(yaw, 1.5 * fps) + math.radians(az) + np.linspace(0, sweep, T)
    cam.data.lens = 55 if mode == "opening" else 45
    elev = math.radians(5 if mode == "opening" else 14)

    def frame_cam(i):
        g = ground_z[i]
        bottom_i, top_i = -0.14, hi_z[i] - g + 0.1  # room for the shadow and the head
        mid_i = (top_i + bottom_i) / 2
        dist = (top_i - bottom_i) / 2 / (18 / cam.data.lens)
        c = Vector((*c_xy[i], g))
        for node, v in zip(FADE, c_xy[i], strict=False):
            node.outputs[0].default_value = v
        a = cam_az[i]
        cam.location = c + Vector(
            (dist * math.cos(a), dist * math.sin(a), mid_i + dist * math.tan(elev))
        )
        aim(cam, c + Vector((0, 0, mid_i)))
        place_lights(a, c, mid_i)

elif mode in ("side", "hands"):  # failure panels: fixed view, follows the pelvis
    W = 960
    # "side": 1 m hops from the side; "hands": 0.5 m hops, closer and from the
    # front, so the hands on the handle read
    z0, z1, a = (-0.15, 2.85, 62) if mode == "side" else (-0.15, 2.4, 30)
    c_xy = smooth(P[:, :2], 0.5 * fps)
    a = heading + math.radians(a)
    cam.data.lens = 50
    mid = (z0 + z1) / 2
    dist = (z1 - z0) / 2 / (18 / cam.data.lens)

    def frame_cam(i):
        c = Vector((*c_xy[i], 0.0))
        cam.location = c + Vector(
            (dist * math.cos(a), dist * math.sin(a), mid + 0.1 * dist)
        )
        aim(cam, c + Vector((0, 0, mid)))
        place_lights(a, c, mid)

elif mode == "turn":  # camera 45 degrees up, orientation fixed in the world
    W = int(os.environ.get("PANEL_W", 640))
    tip_g = [g[0] for g in geoms].index("pogo/pogo_Grip_Tip_Rubber_Mat_0")
    c_xy = smooth(np.array([z[f"pos/{n}"][tip_g][:2] for n in poses]), 0.5 * fps)
    pan = smooth(c_xy, 1.5 * fps)  # the camera translates with the rider, never turns
    h_s = smooth(yaw, 0.1 * fps)
    a, el, dist, aim_z, R = heading + math.radians(35), math.radians(45), DIST, 0.8, 0.5
    cam.data.lens = 50
    ring = flat("ring", "d6d6d3")
    arc_m = flat("arc", "3a3a3a")

    def frame_cam(i):
        c0 = Vector((*pan[i], 0.0))
        cam.location = c0 + Vector(
            (
                dist * math.cos(el) * math.cos(a),
                dist * math.cos(el) * math.sin(a),
                aim_z + dist * math.sin(el),
            )
        )
        aim(cam, c0 + Vector((0, 0, aim_z)))
        place_lights(a, c0, 1.0)
        c = c_xy[i]
        t = np.linspace(0, 2 * math.pi, 129)
        mark(
            "ring",
            ring,
            ribbon(c + R * np.stack([np.cos(t), np.sin(t)], 1), 0.012),
        )
        # the turn so far: an arc from the starting heading, and a needle now
        t = np.linspace(h_s[0], h_s[i], max(2, int(abs(h_s[i] - h_s[0]) / 0.03)))
        v, f = ribbon(c + R * np.stack([np.cos(t), np.sin(t)], 1), 0.03, 0.003)
        u = np.array([math.cos(h_s[i]), math.sin(h_s[i])])
        nrm = np.array([-u[1], u[0]])
        base = c + (R + 0.02) * u
        k = len(v)
        v += [
            [*(c + (R + 0.17) * u), 0.004],
            [*(base + 0.07 * nrm), 0.004],
            [*(base - 0.07 * nrm), 0.004],
        ]
        f += [(k, k + 1, k + 2)]
        stem = ribbon([c + 0.3 * R * u, base], 0.012, 0.003)
        v, f = v + stem[0], f + [tuple(len(v) + j for j in q) for q in stem[1]]
        mark("arc", arc_m, (v, f))

else:  # nus: one letter, fixed high oblique view from its foot
    W = 640
    path, ink = z["path"], z["ink"].astype(bool)
    trail, pen, step = smooth(z["trail"], 5), z["pen"].astype(bool), z["step"]

    def strokes(xy, down, width, z0):
        """Ribbons along the pen-down runs of XY."""
        v, f = [], []
        for run in np.split(
            np.arange(len(xy)), np.flatnonzero(np.diff(down.astype(int))) + 1
        ):
            if down[run[0]] and len(run) > 1:
                rv, rf = ribbon(xy[run], width, z0)
                v, f = v + rv, f + [tuple(len(v) + j for j in q) for q in rf]
        return v, f

    lo2, hi2 = path[ink].min(0), path[ink].max(0)
    c0 = Vector(((lo2[0] + hi2[0]) / 2, (lo2[1] + hi2[1]) / 2 + 0.1, 0.0))
    a, el = -math.pi / 2, math.radians(60)  # from -y: the letter reads upright
    cam.data.lens = 45
    # fit 2.4 m plus margins (N and U at NUS_SCALE=2), so all letters share a scale
    dist = float(max(hi2[0] - lo2[0], 2.4) + 1.4) / 2 / (18 / 45 * W / H)
    cam.location = c0 + dist * Vector((0, -math.cos(el), math.sin(el)))
    aim(cam, c0)
    mark("letters", flat("letters", "e0e0dd"), strokes(path, ink, 0.14, 0.001))
    blue = flat("path", "2a78d6")

    def frame_cam(i):
        place_lights(a, Vector((*P[i, :2], 0.0)), 1.0)
        n = step[i] + 1
        mark("path", blue, strokes(trail[:n], pen[:n], 0.045, 0.003))


r.resolution_x, r.resolution_y = W, H
mr.resolution_x, mr.resolution_y = W, H
for sub in ("rider", "marks"):
    Path(out, sub).mkdir(parents=True, exist_ok=True)
for i in range(f0, f1):
    pose(poses[i])
    frame_cam(i)
    r.filepath = f"{out}/rider/{i:04d}.png"
    bpy.ops.render.render(write_still=True)
    mr.filepath = f"{out}/marks/{i:04d}.png"
    bpy.ops.render.render(write_still=True, scene="marks")
    print("wrote frame", i, flush=True)
