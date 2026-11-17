"""Offscreen rendering helpers (CPU MuJoCo, EGL)."""

import mujoco
import numpy as np


def add_visual_defaults(spec: mujoco.MjSpec):
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.32, 0.34, 0.37],
        rgb2=[0.26, 0.28, 0.31],
        width=512,
        height=512,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=[0.5, 0.5, 0.5],
    )
    spec.add_material(
        name="grid", textures=["", "grid"], texrepeat=[10, 10], texuniform=True
    )
    for geom in spec.geoms:
        if geom.name == "floor" or geom.name.startswith("terrain"):
            geom.material = "grid"
    spec.worldbody.add_light(
        pos=[1, -2, 4], dir=[-0.2, 0.4, -1], diffuse=[0.8, 0.8, 0.8]
    )
    spec.worldbody.add_light(
        pos=[-2, 2, 3], dir=[0.5, -0.5, -1], diffuse=[0.4, 0.4, 0.4]
    )


def frames(model, data, views, width=640, height=480, lookat=None):
    out = []
    with mujoco.Renderer(model, height, width) as r:
        opt = mujoco.MjvOption()
        for az, el, dist in views:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = lookat if lookat is not None else data.subtree_com[0]
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            r.update_scene(data, cam, opt)
            out.append(r.render())
    return np.concatenate(out, axis=1)
