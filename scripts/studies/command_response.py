"""Mid-episode command response: body-frame COM velocity per command segment.

usage: command_response.py RUN_DIR CKPT  (last line: JSON list of segments)
"""

import dataclasses
import json
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

run, ck = Path(sys.argv[1]), sys.argv[2]
task = load_task(run)
task.worlds, task.airborne_prob, task.episode_s, task.apex_cap = 32, 0.0, 100.0, 0.0
env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
w = RslRlVecEnvWrapper(env)
r = MjlabOnPolicyRunner(w, dataclasses.asdict(runner_cfg()), None, device="cuda:0")
r.load(str(run / ck), load_cfg={"actor": True}, strict=False)
pol = r.get_inference_policy(device="cuda:0")
term = env.command_manager.get_term("hop")
robot = env.scene["robot"].data
obs = w.get_observations()
schedule = [
    (3, (0.15, 0, 0, 0)),
    (8, (0.15, 0.3, 0, 0)),
    (8, (0.15, 0, 0.3, 0)),
    (8, (0.15, 0.3, 0, 0)),
    (8, (0.15, 0, 0, 0)),
]
segments = []
with torch.inference_mode():
    for dur, c in schedule:
        term.set_command(*c)
        vb, hs = [], []
        for i in range(int(dur / env.step_dt)):
            obs, _, done, _ = w.step(pol(obs))
            if i * env.step_dt > 2:  # after 2 s, body-frame COM velocity
                vb.append(mdp.com_velocity_yaw(env).cpu().numpy())
                hs.append(robot.root_link_ang_vel_w[:, 2].cpu().numpy())
        v = np.array(vb).mean(0).mean(0)
        yr = float(np.array(hs).mean())
        segments.append(
            {
                "cmd": list(c[1:]),
                "v": [round(float(x), 3) for x in v],
                "yaw": round(yr, 3),
            }
        )
        print(
            f"{ck} cmd {c[1:]}: body velocity mean {np.round(v, 3)}, yaw rate {yr:+.3f}"
        )
print("JSON", json.dumps(segments))
