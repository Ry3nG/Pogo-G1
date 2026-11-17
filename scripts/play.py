"""Roll out a checkpoint: metrics over many worlds + an MP4 of world 0.

  uv run --locked --extra train python scripts/play.py RUN_DIR [--ckpt model_2999.pt]
      [--worlds 256] [--seconds 20] [--set apex=(0.1,0.1)] [--video]

Starts from the static mounted pose (airborne_prob=0). The video is a CPU
MuJoCo replay of world 0's qpos (identical model, rendered offscreen).
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as iio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402

from pogo_g1 import mdp, render  # noqa: E402
from pogo_g1.env import make_env_cfg, parse_set  # noqa: E402
from pogo_g1.evaluate import (  # noqa: E402
    HoppingContinuity,
    latest_checkpoint,
    load_task,
)
from pogo_g1.ppo import runner_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ckpt")
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--video", action="store_true")
    ap.add_argument(
        "--teleop",
        action="store_true",
        help="Native viewer: WASD, QE turn, X hop in place, Space pause",
    )
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()

    task = load_task(args.run)
    task.worlds, task.airborne_prob, task.episode_s = args.worlds, 0.0, args.seconds + 1
    task.apex_cap = 0.0  # playback uses the requested commands, no curriculum
    if args.teleop:
        task.worlds, task.episode_s, task.robust = 1, 3600.0, False
    task = parse_set(task, args.set)
    ckpt = args.ckpt or latest_checkpoint(args.run)
    env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
    agent = runner_cfg()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(
        wrapped, dataclasses.asdict(agent), None, device="cuda:0"
    )
    runner.load(str(args.run / ckpt), load_cfg={"actor": True}, strict=False)
    policy = runner.get_inference_policy(device="cuda:0")

    if args.teleop:
        from mjlab.viewer import NativeMujocoViewer

        pending = SimpleQueue()
        command = env.command_manager.get_term("hop")
        height = sum(task.apex) / 2
        command.set_command(height)
        keys = {
            "w": (0.3, 0, 0),
            "s": (-0.3, 0, 0),
            "a": (0, 0.3, 0),
            "d": (0, -0.3, 0),
            "q": (0, 0, 0.4),
            "e": (0, 0, -0.4),
            "x": (0, 0, 0),
        }

        def controlled_policy(obs):
            # Viewer callbacks run on another thread. Drain only on the sim
            # thread; the next observation incorporates the new command without
            # inserting synthetic frames into the proprioceptive history.
            while True:
                try:
                    key = chr(pending.get_nowait()).lower()
                except Empty:
                    break
                if key in keys:
                    command.set_command(height, *keys[key])
                    print(f"command: h={height:g}, vx/vy/yaw={keys[key]}", flush=True)
            return policy(obs)

        print(
            "W/S forward/back, A/D left/right, Q/E turn, X hop in place; Space pause."
        )
        try:
            NativeMujocoViewer(
                wrapped, controlled_policy, key_callback=pending.put
            ).run()
        finally:
            env.close()
        return

    obs = wrapped.get_observations()
    steps = int(args.seconds / env.step_dt)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device="cuda:0")
    continuity = HoppingContinuity(env.num_envs, "cuda:0")
    fail_t = torch.full((env.num_envs,), float("nan"), device="cuda:0")
    hops = torch.zeros(env.num_envs, device="cuda:0")
    H, cmd, qpos0 = [], [], []
    with torch.inference_mode():
        for i in range(steps):
            obs, _, dones, _ = wrapped.step(policy(obs))
            hop = mdp.hops(env) * alive
            hops += hop
            newly = dones.bool() & alive & (env.episode_length_buf == 0)
            fail_t[newly] = i * env.step_dt
            alive &= ~newly
            continuity.update((i + 1) * env.step_dt, hop, alive)
            H.append(mdp.energy_height(env)[0].item())
            cmd.append(mdp.hop_command(env)[0].tolist())
            qpos0.append(env.sim.data.qpos[0].cpu().numpy().copy())
    res = {
        "run": str(args.run),
        "ckpt": ckpt,
        "set": args.set,
        "worlds": env.num_envs,
        "seconds": args.seconds,
        "success_rate": continuity.success.float().mean().item(),
        "survival_rate": alive.float().mean().item(),
        "mean_hops_survivors": hops[alive].mean().item() if alive.any() else 0.0,
        "median_fail_time_s": float(np.nanmedian(fail_t.cpu().numpy()))
        if (~alive).any()
        else None,
        "world0_energy_height_mean": float(np.mean(H)),
    }
    out = args.run / f"{args.tag}-{Path(ckpt).stem}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    if args.video:
        spec = env.scene.spec.copy()
        render.add_visual_defaults(spec)
        m = spec.compile()
        d = mujoco.MjData(m)
        path = args.run / f"{args.tag}-{Path(ckpt).stem}.mp4"
        with (
            iio.get_writer(path, fps=int(round(1 / env.step_dt))) as w,
            mujoco.Renderer(m, 480, 854) as r,
        ):
            cam = mujoco.MjvCamera()
            cam.azimuth, cam.elevation, cam.distance = 150, -12, 2.8
            for q in qpos0:
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                cam.lookat[:] = d.subtree_com[0]
                r.update_scene(d, cam)
                w.append_data(r.render())
        print(path)
    env.close()


if __name__ == "__main__":
    main()
