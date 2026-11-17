"""Candidate yaw-rate signals and how much a per-step score of each pays for
turning under training noise.

  record: yaw_signals.py record RUN CKPT YAW_CMD OUT.npz [noise]
  gain:   yaw_signals.py gain TURN.npz STILL.npz

`record` hops in place at 0.15 m (64 worlds, 20 s, the first 5 s skipped),
deterministic or with sampled actions as in training, and keeps every step's
pelvis, torso, stick and centroidal yaw rate (`mdp.centroidal_yaw_rate`,
L_z / I_zz of robot + stick) and the tip contact. `gain` prints, per signal,
the within-world swing (standard deviation) of the turning recording and the
reward per second that turning (command 0.5) earns over the in-place
recording when the signal is scored every step with S(e) = 1/(1+(e/0.15)^2).
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

SIGNALS = ("pelvis", "torso", "stick", "centroidal")


def record(run, ck, wz, out, noise):
    from pogo_g1 import mdp
    from pogo_g1.env import make_env_cfg
    from pogo_g1.evaluate import load_task
    from pogo_g1.ppo import runner_cfg

    task = load_task(run)
    task.worlds, task.airborne_prob, task.episode_s, task.apex_cap = 64, 0.0, 30.0, 0.0
    env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
    w = RslRlVecEnvWrapper(env)
    r = MjlabOnPolicyRunner(w, dataclasses.asdict(runner_cfg()), None, device="cuda:0")
    r.load(str(run / ck), load_cfg={"actor": True}, strict=False)
    pol = r.get_inference_policy(device="cuda:0")
    env.command_manager.get_term("hop").set_command(0.15, 0.0, 0.0, wz)
    robot, stick = env.scene["robot"], env.scene["pogo"]
    torso = robot.body_names.index("torso_link")
    rec = {k: [] for k in (*SIGNALS, "contact")}
    obs = w.get_observations()
    with torch.inference_mode():
        for i in range(int(20.0 / env.step_dt)):
            obs, _, _, _ = w.step(pol(obs, stochastic_output=noise))
            if i * env.step_dt < 5.0:
                continue
            rec["pelvis"].append(mdp.yaw_rate(env))
            rec["torso"].append(robot.data.body_link_ang_vel_w[:, torso, 2])
            rec["stick"].append(stick.data.root_link_ang_vel_w[:, 2])
            rec["centroidal"].append(mdp.centroidal_yaw_rate(env))
            rec["contact"].append(mdp.tip_contact(env))
    rec = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
    np.savez(out, dt=env.step_dt, command=wz, **rec)
    print({k: round(float(rec[k].mean()), 3) for k in SIGNALS})


def gain(turn, still, cmd=0.5, sigma=0.15):
    t, s = np.load(turn), np.load(still)
    for k in SIGNALS:
        swing = np.std(t[k] - t[k].mean(0), axis=0).mean()
        g = (1 / (1 + ((t[k] - cmd) / sigma) ** 2)).mean()
        g -= (1 / (1 + ((s[k] - cmd) / sigma) ** 2)).mean()
        print(f"{k:10s} swing {swing:.2f} rad/s  per-step turning gain {g:.3f}/s")


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(
            Path(sys.argv[2]),
            sys.argv[3],
            float(sys.argv[4]),
            sys.argv[5],
            len(sys.argv) > 6 and sys.argv[6] == "noise",
        )
    else:
        gain(sys.argv[2], sys.argv[3])
