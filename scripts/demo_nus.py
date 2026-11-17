"""Hop 'NUS': a pure-pursuit path follower issues [h, vx, vy, yaw] through
HopCommand.set_command (the teleop/planner interface) and the policy rides.
Left: chase view. Right: fixed overhead view with the hopped path inked.

usage: [NORENDER=1] [NUS_BIAS=0|1] [NUS_SPEED=0.3] [NUS_LETTERS=NUS] [NUS_SCALE=1]
       [NUS_EXPORT=DIR] demo_nus.py RUN CKPT OUT.mp4 [hold|follow]
"hold" keeps the heading; "follow" turns to face along each stroke. The error
is the distance to the path near the follower's current point, over the
letters only (not the pen-up moves); the same follower driving an ideal point
gives the floor set by the outer loop alone. The last line is a JSON summary.
NUS_EXPORT=DIR saves the rendered world for scripts/render/ instead of the MP4.
"""

import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as iio
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from PIL import Image, ImageDraw, ImageFont

from pogo_g1 import mdp, render
from pogo_g1.env import make_env_cfg
from pogo_g1.evaluate import load_task
from pogo_g1.ppo import runner_cfg

run, ckpt, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
MODE = sys.argv[4] if len(sys.argv) > 4 else "hold"
H, LOOK, ACC = 0.15, 0.35, 1.0  # hop height, lookahead, command accel limit
V = float(os.environ.get("NUS_SPEED", "0.3"))  # path speed


def arc(c, r, a0, a1, n=24):
    t = np.radians(np.linspace(a0, a1, n))
    return [(c[0] + r * np.cos(a), c[1] + r * np.sin(a)) for a in t]


# Letters on the floor (m): x along the word, y up the letter. Strokes = pen down.
N = [(0.0, 0.0), (0.0, 2.0), (1.2, 0.0), (1.2, 2.0)]
U = [(2.0, 2.0), (2.0, 0.6)] + arc((2.6, 0.6), 0.6, 180, 360)[1:] + [(3.2, 2.0)]
S = arc((4.6, 1.5), 0.5, 20, 270) + arc((4.6, 0.5), 0.5, 90, -160)[1:]
# NUS_LETTERS picks letters (each starting where the rider stands); NUS_SCALE
# scales them (1: letters 2 m tall, an 18 m path).
SCALE = float(os.environ.get("NUS_SCALE", "1"))
strokes = [dict(N=N, U=U, S=S)[c] for c in os.environ.get("NUS_LETTERS", "NUS")]
strokes = [SCALE * (np.array(st) - strokes[0][0]) for st in strokes]
path, ink = [], []  # dense polyline and pen state per point
for k, st in enumerate(strokes):
    if k:  # pen up from the previous stroke's end
        a, b = np.array(path[-1]), np.array(st[0])
        for s in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.05)))[1:]:
            path.append(tuple(a + s * (b - a)))
            ink.append(False)
    for a, b in zip(st[:-1], st[1:], strict=False):
        a, b = np.array(a), np.array(b)
        for s in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.05)))[
            (0 if not path else 1) :
        ]:
            path.append(tuple(a + s * (b - a)))
            ink.append(True)
path = np.array(path)
ink = np.array(ink)
seglen = np.linalg.norm(np.diff(path, axis=0), axis=1)
arclen = np.concatenate([[0], np.cumsum(seglen)])
print(f"path length {arclen[-1]:.1f} m, {len(path)} points")

task = load_task(run)
task.worlds, task.airborne_prob, task.episode_s, task.apex_cap = 16, 0.0, 220.0, 0.0
env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
w = RslRlVecEnvWrapper(env)
r = MjlabOnPolicyRunner(w, dataclasses.asdict(runner_cfg()), None, device="cuda:0")
r.load(str(run / ckpt), load_cfg={"actor": True}, strict=False)
pol = r.get_inference_policy(device="cuda:0")
term = env.command_manager.get_term("hop")
robot = env.scene["robot"].data
obs = w.get_observations()
NW = env.num_envs
origins = env.scene.env_origins[:, :2].cpu().numpy()
dt = env.step_dt
BIAS = os.environ.get("NUS_BIAS", "1") == "1"


def steer(p, heading, j0):
    """Pure pursuit from position p (m, path frame) and heading: the wanted
    [vx, vy, yaw] in the body frame, the nearest path index, and the distance
    to the path near that index (not to a crossing stroke elsewhere)."""
    j = j0 + np.argmin(np.linalg.norm(path[j0 : j0 + 40] - p, axis=1))
    k = min(np.searchsorted(arclen, arclen[j] + LOOK), len(path) - 1)
    for kk in range(j + 1, k):  # do not look around a sharp corner
        a0, a1 = path[kk] - path[kk - 1], path[kk + 1] - path[kk]
        if (
            np.dot(a0, a1) < 0.5 * np.linalg.norm(a0) * np.linalg.norm(a1)
            and np.linalg.norm(path[kk] - p) > 0.12
        ):
            k = kk
            break
    d = path[k] - p
    vw = V * d / max(np.linalg.norm(d), 1e-6)
    c, sn = np.cos(heading), np.sin(heading)
    body = np.array([c * vw[0] + sn * vw[1], -sn * vw[0] + c * vw[1]])
    if MODE == "follow":
        err = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
        yaw = np.clip(1.5 * err, -0.5, 0.5)
        body *= 0.0 if abs(err) > 0.6 else max(0.0, np.cos(err)) ** 2
    else:
        yaw = np.clip(-1.0 * heading, -0.3, 0.3)
    near = np.linalg.norm(path[max(j - 20, 0) : j + 20] - p, axis=1).min()
    return np.array([body[0], body[1], yaw]), j, float(near)


# The same follower driving an ideal point that does exactly what it is
# commanded: the error floor set by the outer loop alone.
kp, kh, kj, kc, k_errs = np.zeros(2), 0.0, 0, np.zeros(3), []
for _ in range(int(200 / dt)):
    want, kj, near = steer(kp, kh, kj)
    if ink[kj]:
        k_errs.append(near)
    if kj >= len(path) - 3:
        break
    kc = kc + np.clip(want - kc, -ACC * dt, ACC * dt)
    c, sn = np.cos(kh), np.sin(kh)
    kp = kp + dt * np.array([c * kc[0] - sn * kc[1], sn * kc[0] + c * kc[1]])
    kh += dt * kc[2]
ideal = float(np.mean(k_errs))

pos0 = None
idx = np.zeros(NW, int)
cmd = np.zeros((NW, 3))
bias = np.zeros((NW, 2))
alive = np.ones(NW, bool)
t_end = np.full(NW, np.nan)
errs = [[] for _ in range(NW)]
qpos, trails, cmds = [], [], []
term._external_command = None
with torch.inference_mode():
    for i in range(int(200 / dt)):
        term.time_left[:] = 1e9  # commands come only from this follower, per world
        # The follower reads the true position (and, for the bias correction,
        # the true COM velocity): privileged state a real rider would estimate.
        P = robot.root_link_pos_w[:, :2].cpu().numpy() - origins
        if pos0 is None:
            pos0 = P.copy()
        P = P - pos0
        heading = robot.heading_w.cpu().numpy()
        want = np.zeros((NW, 3))
        pen = np.zeros(NW, bool)
        for n in range(NW):
            if i * dt < 2.0 or not np.isnan(t_end[n]):
                continue
            want[n], idx[n], near = steer(P[n], heading[n], idx[n])
            pen[n] = ink[idx[n]]
            if alive[n] and pen[n]:
                errs[n].append(near)
            if idx[n] >= len(path) - 3:
                t_end[n] = i * dt
        if np.all(~np.isnan(t_end) | ~alive) and i * dt > 5:
            break
        cmd = cmd + np.clip(want - cmd, -ACC * dt, ACC * dt)
        if i * dt >= 2.0 and BIAS:
            vb = mdp.com_velocity_yaw(env).cpu().numpy()
            bias += (dt / 3.0) * ((vb - cmd[:, :2]) - bias)
        sent = cmd.copy()
        sent[:, :2] -= bias
        term.cmd[:, 0] = H
        term.cmd[:, 1:] = torch.as_tensor(
            sent, device=term.cmd.device, dtype=term.cmd.dtype
        )
        obs, _, done, _ = w.step(pol(obs))
        alive &= ~done.cpu().numpy().astype(bool)
        qpos.append(env.sim.data.qpos.cpu().numpy().copy())
        trails.append((P.copy(), pen.copy()))
        cmds.append(sent.copy())
T = len(qpos) * dt
me = np.array([np.mean(e) if e else np.nan for e in errs])
print(
    f"rode {T:.1f} s; worlds finished {np.sum(~np.isnan(t_end) & alive)}/{NW}, failed {np.sum(~alive)}; "
    f"mean path distance per world {np.round(me, 2)}; median {np.nanmedian(me):.3f} m; finish times {np.round(t_end, 0)}"
)
summary = {
    "mode": MODE,
    "bias_correction": BIAS,
    "finished": int(np.sum(~np.isnan(t_end) & alive)),
    "failed": int(np.sum(~alive)),
    "path_err_median": round(float(np.nanmedian(me)), 3),
    "path_err_range": [round(float(np.nanmin(me)), 3), round(float(np.nanmax(me)), 3)],
    "ideal_point_err": round(ideal, 3),
    "finish_s_median": round(float(np.nanmedian(t_end)), 1),
}
print("JSON", json.dumps(summary))
if os.environ.get("NORENDER"):
    sys.exit(0)
order = np.argsort(np.where(alive & ~np.isnan(t_end), me, np.inf))
ok = [o for o in order if alive[o] and not np.isnan(t_end[o])]
W0 = int(ok[len(ok) // 2])  # median finishing world, not the best
print("rendering world", W0, "mean distance", round(float(me[W0]), 3))
if os.environ.get("NUS_EXPORT"):  # replay for scripts/render/export_poses.py
    ex = Path(os.environ["NUS_EXPORT"])
    ex.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(env.scene.spec.copy().compile(), str(ex / "model.mjb"))
    off = env.scene.env_origins[W0, :2].cpu().numpy() + pos0[W0]
    np.savez_compressed(
        ex / "replay.npz",
        qpos=np.array([q[W0] for q in qpos]),
        path=path + off,
        ink=ink,
        trail=np.array([tp[0][W0] for tp in trails]) + off,
        pen=np.array([tp[1][W0] for tp in trails]),
    )
    sys.exit(0)
qpos = [q[W0] for q in qpos]
trail = [(tp[0][W0], tp[1][W0]) for tp in trails]
cmds = [c[W0] for c in cmds]
errs_w = errs[W0]
t_end_w = t_end[W0]
spec = env.scene.spec.copy()
render.add_visual_defaults(spec)
m = spec.compile()
d = mujoco.MjData(m)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
W, Hh = 640, 720
world0 = env.scene.env_origins[W0].cpu().numpy()
pos0 = pos0[W0]
centre = np.array([world0[0] + pos0[0] + 2.6, world0[1] + pos0[1] + 1.0, 0.0])


def add_ink(scene, pts, rgba, width):
    for a, b in zip(pts[:-1], pts[1:], strict=False):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            width,
            np.array([*a, 0.005]),
            np.array([*b, 0.005]),
        )
        scene.ngeom += 1


with (
    iio.get_writer(out, fps=int(round(1 / dt)), quality=7) as wr,
    mujoco.Renderer(m, Hh, W, max_geom=20000) as chase,
    mujoco.Renderer(m, Hh, W, max_geom=20000) as top,
):
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135, -14, 4.0
    tc = mujoco.MjvCamera()
    tc.azimuth, tc.elevation, tc.distance = 90, -62, 8.5
    tc.lookat[:] = centre
    look = None
    ink_pts = []
    ghost = [world0[:2] + pos0 + q for q in path]
    for k, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        com = d.subtree_com[0].copy()
        look = com if look is None else 0.9 * look + 0.1 * com
        cam.lookat[:] = [look[0], look[1], 0.9]
        p, pen = trail[k]
        if k % 3 == 0:
            ink_pts.append((world0[:2] + pos0 + p, pen))
        frames = []
        for rr, c in ((chase, cam), (top, tc)):
            rr.update_scene(d, c)
            if rr is top:
                add_ink(rr.scene, ghost, (1, 1, 1, 0.12), 0.012)
                seg = []
                for xy, pen in ink_pts:
                    if pen:
                        seg.append(xy)
                    elif seg:
                        add_ink(rr.scene, seg, (0.17, 0.47, 0.84, 1), 0.035)
                        seg = []
                if seg:
                    add_ink(rr.scene, seg, (0.17, 0.47, 0.84, 1), 0.035)
            frames.append(rr.render())
        img = Image.fromarray(np.concatenate(frames, axis=1))
        g = ImageDraw.Draw(img, "RGBA")
        c = cmds[k]
        g.rectangle([12, 12, 470, 92], fill=(0, 0, 0, 150))
        g.text(
            (24, 18),
            "Hopping 'NUS' via the command interface"
            + (" (facing along the stroke)" if MODE == "follow" else " (heading held)"),
            font=small if MODE == "follow" else font,
            fill="white",
        )
        g.text(
            (24, 52),
            f"t {k * dt:5.1f} s   command vx {c[0]:+.2f}  vy {c[1]:+.2f}  yaw {c[2]:+.2f}   h {H:.2f} m",
            font=small,
            fill=(220, 220, 220),
        )
        g.text(
            (W + 16, 18),
            "overhead: blue = hopped path (pen down)",
            font=small,
            fill=(255, 255, 255),
        )
        wr.append_data(np.asarray(img.convert("RGB")))
json.dump(
    {
        "T": T,
        "world": W0,
        "t_end": float(t_end_w),
        "mean_err": float(np.mean(errs_w)),
        "p95_err": float(np.percentile(errs_w, 95)),
        "per_world_mean": [None if np.isnan(x) else float(x) for x in me],
        "alive": [bool(x) for x in alive],
        "trail": [[float(a), float(b), bool(pen)] for (a, b), pen in trail[::5]],
    },
    open(out.with_suffix(".json"), "w"),
)
print(out)
