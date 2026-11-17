"""What the steering reward pays for a turning gait, per step versus per hop.

usage: step_score.py RUN CKPT [YAW_CMD|vx=VX] [OUT.npz|-] [noise|noise=STD]

Runs the deterministic policy (or, with `noise`, the sampling one) in place at 0.15 m with a yaw-rate command
(default 0.5 rad/s; `vx=V` commands forward velocity V instead), 64 worlds, and scores its yaw motion after 5 s with the
steering kernel S(e) = 1/(1+(e/0.15)^2) two ways: on every step's yaw rate
(per-step reward) and on each hop cycle's average yaw rate (per-hop reward),
both as reward per second. "Not turning" earns S(command) under either. If the
turning gait earns less than not turning per step, the per-step reward makes
not turning the better local choice.
OUT.npz keeps the per-step yaw rate, planar velocity and tip contact (steps x
worlds) and every hop cycle's mean yaw rate, planar velocity and duration.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from pogo_g1 import mdp
from pogo_g1.env import make_env_cfg
from pogo_g1.evaluate import load_task
from pogo_g1.ppo import runner_cfg

SIGMA = 0.15
run, ck = Path(sys.argv[1]), sys.argv[2]
cmd = sys.argv[3] if len(sys.argv) > 3 else "0.5"
vx_cmd = float(cmd[3:]) if cmd.startswith("vx=") else 0.0
wz_cmd = 0.0 if cmd.startswith("vx=") else float(cmd)
out = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
# "noise" samples actions as in training; "noise=S" also sets the action std to S
arg = sys.argv[5] if len(sys.argv) > 5 else ""
noise = arg.startswith("noise")
task = load_task(run)
task.worlds, task.airborne_prob, task.episode_s, task.apex_cap = 64, 0.0, 30.0, 0.0
env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
w = RslRlVecEnvWrapper(env)
r = MjlabOnPolicyRunner(w, dataclasses.asdict(runner_cfg()), None, device="cuda:0")
r.load(str(run / ck), load_cfg={"actor": True}, strict=False)
pol = r.get_inference_policy(device="cuda:0")
if "=" in arg:
    pol.distribution.std_param.data.fill_(float(arg.split("=")[1]))
env.command_manager.get_term("hop").set_command(0.15, vx_cmd, 0.0, wz_cmd)
obs = w.get_observations()


def score(e):
    return 1.0 / (1.0 + (e / SIGMA) ** 2)


step_e, hop_e, hop_t, contact, vel, hop_vel = [], [], [], [], [], []
with torch.inference_mode():
    for i in range(int(20.0 / env.step_dt)):
        obs, _, _, _ = w.step(pol(obs, stochastic_output=noise))
        hop = mdp.hops(env).bool()
        if i * env.step_dt < 5.0:
            continue
        step_e.append((mdp.yaw_rate(env) - wz_cmd).cpu().numpy())
        contact.append(mdp.tip_contact(env).cpu().numpy())
        vel.append(mdp.com_velocity_yaw(env)[:, :2].cpu().numpy())
        if hop.any():
            hop_e.append((env._land_motion[hop, 2] - wz_cmd).cpu().numpy())
            hop_t.append(env._land_period[hop].cpu().numpy())
            hop_vel.append(env._land_motion[hop, :2].cpu().numpy())
if out:
    np.savez(
        out,
        command=wz_cmd,
        vx_command=vx_cmd,
        dt=env.step_dt,
        yaw=np.stack(step_e) + wz_cmd,
        contact=np.stack(contact),
        hop_yaw=np.concatenate(hop_e) + wz_cmd,
        hop_t=np.concatenate(hop_t),
        vel=np.stack(vel),
        hop_vel=np.concatenate(hop_vel),
    )
step_e = np.concatenate(step_e)
hop_e, hop_t = np.concatenate(hop_e), np.concatenate(hop_t)
yaw = step_e + wz_cmd
print(f"{run.name} {ck}: command {wz_cmd} rad/s")
print(f"  mean yaw rate {yaw.mean():.3f}, within-hop std {yaw.std():.3f} rad/s")
print(f"  per-step reward rate  {score(step_e).mean():.3f}")
print(f"  per-hop reward rate   {np.sum(score(hop_e) * hop_t) / np.sum(hop_t):.3f}")
print(f"  not turning (either)  {score(np.array(wz_cmd)):.3f}")
v, hv = np.concatenate(vel), np.concatenate(hop_vel)
print(
    f"  planar speed: per step {np.linalg.norm(v, axis=-1).mean():.3f}, hop mean {np.linalg.norm(hv, axis=-1).mean():.3f} m/s"
)
print(
    f"  planar score (command 0): per step {score(np.linalg.norm(v, axis=-1)).mean():.3f}, per hop {np.sum(score(np.linalg.norm(hv, axis=-1)) * hop_t) / np.sum(hop_t):.3f}"
)
