"""Scripted command showcase: one world, a 42 s command schedule, rendered with a
world-fixed camera azimuth, command overlay and a top-down path inset."""

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
SCHEDULES = {
    "v1": (
        [  # (start s, label, [apex, vx, vy, yaw])
            (0, "Hop in place 0.15 m", [0.15, 0, 0, 0]),
            (4, "Forward 0.4 m/s", [0.15, 0.4, 0, 0]),
            (10, "Turn left 0.5 rad/s", [0.15, 0, 0, 0.5]),
            (16, "Sideways left 0.3 m/s", [0.15, 0, 0.3, 0]),
            (21, "Backward 0.3 m/s", [0.15, -0.3, 0, 0]),
            (26, "High hops 0.45 m", [0.45, 0, 0, 0]),
            (33, "Curve: forward 0.3 + turn right 0.4", [0.2, 0.3, 0, -0.4]),
            (39, "Stop: hop in place 0.12 m", [0.12, 0, 0, 0]),
        ],
        43,
    ),
    "v2": (
        [
            (0, "Hop in place 0.15 m", [0.15, 0, 0, 0]),
            (4, "Forward 0.4 m/s", [0.15, 0.4, 0, 0]),
            (10, "Turn left 0.5 rad/s", [0.15, 0, 0, 0.5]),
            (16, "Sideways right 0.3 m/s", [0.15, 0, -0.3, 0]),
            (21, "Backward 0.3 m/s", [0.15, -0.3, 0, 0]),
            (26, "High hops 0.75 m (climbs over ~5 hops)", [0.75, 0, 0, 0]),
            (40, "Down to 0.40 m, turning right 0.4", [0.40, 0, 0, -0.4]),
            (47, "Curve: forward 0.3 + turn left 0.4", [0.2, 0.3, 0, 0.4]),
            (53, "Stop: hop in place 0.12 m", [0.12, 0, 0, 0]),
        ],
        57,
    ),
}
SCHEDULES["v3"] = (
    [
        (0, "Hop in place 0.15 m", [0.15, 0, 0, 0]),
        (4, "Forward 0.4 m/s", [0.15, 0.4, 0, 0]),
        (10, "Stop", [0.15, 0, 0, 0]),
        (13, "Turn left 0.5 rad/s", [0.15, 0, 0, 0.5]),
        (19, "Sideways right 0.3 m/s", [0.15, 0, -0.3, 0]),
        (24, "Backward 0.3 m/s", [0.15, -0.3, 0, 0]),
        (29, "Climb: 0.50 m", [0.5, 0, 0, 0]),
        (36, "Climb: 0.75 m", [0.75, 0, 0, 0]),
        (44, "Climb: 1.00 m", [1.0, 0, 0, 0]),
        (54, "Climb: 1.20 m", [1.2, 0, 0, 0]),
        (65, "Down to 0.30 m, curve forward 0.3 + turn 0.4", [0.3, 0.3, 0, 0.4]),
        (73, "Stop: hop in place 0.12 m", [0.12, 0, 0, 0]),
    ],
    77,
)
# Experiment 2 comparison: the same turn command for per-hop and per-step policies.
# Terrain demo (run a stairs-trained policy; its config sets the terrain).
SCHEDULES["stairs"] = (
    [
        (0, "Hop in place 0.3 m", [0.3, 0, 0, 0]),
        (2, "Forward 0.4 m/s: climbing 10 cm stairs", [0.3, 0.4, 0, 0]),
    ],
    22,
)
SCHEDULES["turn"] = (
    [
        (0, "Hop in place 0.15 m", [0.15, 0, 0, 0]),
        (3, "Turn left 0.5 rad/s", [0.15, 0, 0, 0.5]),
    ],
    15,
)
SCHEDULE, SECONDS = SCHEDULES[sys.argv[4] if len(sys.argv) > 4 else "v1"]
task = load_task(run)
task.worlds, task.airborne_prob, task.episode_s, task.apex_cap = (
    8,
    0.0,
    SECONDS + 2,
    0.0,
)
env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
w = RslRlVecEnvWrapper(env)
r = MjlabOnPolicyRunner(w, dataclasses.asdict(runner_cfg()), None, device="cuda:0")
r.load(str(run / ckpt), load_cfg={"actor": True}, strict=False)
pol = r.get_inference_policy(device="cuda:0")
term = env.command_manager.get_term("hop")
obs = w.get_observations()
qpos, labels, cmds, apexes, alive = [], [], [], [], True
seg = -1
steps = int(SECONDS / env.step_dt)
with torch.inference_mode():
    for i in range(steps):
        t = i * env.step_dt
        while seg + 1 < len(SCHEDULE) and t >= SCHEDULE[seg + 1][0]:
            seg += 1
            term.set_command(*SCHEDULE[seg][2])
        obs, _, done, _ = w.step(pol(obs))
        if done[0]:
            alive = False
        hop = mdp.hops(env)[0].item()
        if hop:
            apexes.append((t, env._land_apex[0].item()))
        qpos.append(env.sim.data.qpos[0].cpu().numpy().copy())
        labels.append(SCHEDULE[seg][1])
        cmds.append(SCHEDULE[seg][2])
print("world 0 survived:", alive, "hops:", len(apexes))
spec = env.scene.spec.copy()
render.add_visual_defaults(spec)
m = spec.compile()
d = mujoco.MjData(m)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
W, H = 1280, 720
pelvis = m.body("robot/pelvis").id
path_xy, look = [], None
with (
    iio.get_writer(out, fps=int(round(1 / env.step_dt)), quality=7) as wr,
    mujoco.Renderer(m, H, W) as rr,
):
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135, -12, 4.2
    last_apex = None
    ai = 0
    for k, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        com = d.subtree_com[0].copy()
        look = com if look is None else 0.9 * look + 0.1 * com  # smoothed follow
        cam.lookat[:] = [
            look[0],
            look[1],
            look[2] + 0.1,
        ]  # follows the ground up stairs
        rr.update_scene(d, cam)
        img = Image.fromarray(rr.render())
        t = k * env.step_dt
        while ai < len(apexes) and apexes[ai][0] <= t:
            last_apex = apexes[ai][1]
            ai += 1
        path_xy.append(d.xpos[pelvis, :2].copy())
        g = ImageDraw.Draw(img, "RGBA")
        g.rectangle([16, 16, 560, 120], fill=(0, 0, 0, 150))
        g.text((30, 24), labels[k], font=font, fill="white")
        c = cmds[k]
        g.text(
            (30, 58),
            f"command: h {c[0]:.2f} m  vx {c[1]:+.1f}  vy {c[2]:+.1f}  yaw {c[3]:+.1f}",
            font=small,
            fill=(220, 220, 220),
        )
        g.text(
            (30, 84),
            f"t {t:4.1f} s   last apex {('%.2f m' % last_apex) if last_apex else '-'}",
            font=small,
            fill=(220, 220, 220),
        )
        # top-down path inset (world frame, metres)
        x0, y0, s, size = W - 236, 16, 34, 220
        g.rectangle([x0, y0, x0 + size, y0 + size], fill=(0, 0, 0, 150))
        p = np.array(path_xy)
        cxy = p[-1]
        pts = [
            (x0 + size / 2 + (pp[0] - cxy[0]) * s, y0 + size / 2 - (pp[1] - cxy[1]) * s)
            for pp in p[::5]
        ] + [(x0 + size / 2, y0 + size / 2)]
        pts = [(min(max(a, x0), x0 + size), min(max(b, y0), y0 + size)) for a, b in pts]
        if len(pts) > 1:
            g.line(pts, fill=(134, 182, 239, 255), width=3)
        R = d.xmat[pelvis].reshape(3, 3)
        hx, hy = R[0, 0], R[1, 0]
        n = np.hypot(hx, hy) or 1
        cx, cy = x0 + size / 2, y0 + size / 2
        g.line(
            [(cx, cy), (cx + 22 * hx / n, cy - 22 * hy / n)],
            fill=(235, 104, 52, 255),
            width=4,
        )
        g.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill="white")
        g.text(
            (x0 + 8, y0 + size - 24),
            "top view, 1 m ≈ 34 px",
            font=small,
            fill=(200, 200, 200),
        )
        wr.append_data(np.asarray(img.convert("RGB")))
json.dump(
    {"alive": alive, "apexes": apexes, "schedule": SCHEDULE},
    open(out.with_suffix(".json"), "w"),
)
print(out)
