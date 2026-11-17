"""Train a pogo policy with PPO (rsl_rl via mjlab).

Example:
  uv run --locked --extra train python scripts/train.py --name m2-base --iters 3000
  ... --set robust=True --set 'apex=(0.05,0.15)' --set apex_cap=1.5
"""

import argparse
import dataclasses
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from pogo_g1.env import TaskCfg, make_env_cfg, parse_set
from pogo_g1.pogo import PHYSICS_VERSION
from pogo_g1.ppo import runner_cfg, to_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--iters", type=int, default=3000, help="Additional PPO updates")
    ap.add_argument(
        "--seed", type=int, help="Default: inherited on resume, otherwise 0"
    )
    ap.add_argument("--set", action="append", default=[], help="TaskCfg field=value")
    start = ap.add_mutually_exclusive_group()
    start.add_argument("--resume", type=Path)
    start.add_argument(
        "--warm-start",
        type=Path,
        help="Load actor and its normalizer only; reset critic, optimizer and iteration",
    )
    ap.add_argument("--root", type=Path, default=Path("artifacts/runs"))
    ap.add_argument(
        "--max-std",
        type=float,
        help="Upper bound on the action noise std (default none)",
    )
    args = ap.parse_args()

    checkpoint = args.resume or args.warm_start
    task = TaskCfg()
    if checkpoint:
        from pogo_g1.evaluate import load_task

        saved_task = load_task(
            checkpoint.parent, training=True
        )  # enforce the physics revision
        if args.resume:
            task = saved_task

    if args.seed is not None:
        task.seed = args.seed
    task = parse_set(task, args.set)
    out = args.root / f"{time.strftime('%m%d-%H%M')}-{args.name}-s{task.seed}"
    out.mkdir(parents=True, exist_ok=False)
    agent = runner_cfg(seed=task.seed, iters=args.iters)
    if args.warm_start:
        # A fresh critic changes the advantages; keep early actor updates small.
        agent.algorithm.learning_rate = 1e-4
        agent.algorithm.schedule = "fixed"
    agent_dict = to_dict(agent, symmetry=task.symmetry)
    if args.resume:
        agent_dict = json.loads((args.resume.parent / "config.json").read_text())[
            "agent"
        ]
        agent_dict.update(seed=task.seed, max_iterations=args.iters)
        if task.symmetry:
            agent_dict["algorithm"]["symmetry_cfg"] = to_dict(agent, True)["algorithm"][
                "symmetry_cfg"
            ]
        else:
            agent_dict["algorithm"].pop("symmetry_cfg", None)
    if args.max_std:
        agent_dict["actor"]["distribution_cfg"]["std_range"] = [1e-6, args.max_std]
    if task.pose_path:
        shutil.copyfile(task.pose_path, out / "mounted_pose.json")
        task.pose_path = str(out / "mounted_pose.json")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "src", "scripts"],
            capture_output=True,
            text=True,
        ).stdout
    )
    (out / "config.json").write_text(
        json.dumps(
            {
                "physics_version": PHYSICS_VERSION,
                "task": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in dataclasses.asdict(task).items()
                },
                "agent": agent_dict,
                "commit": commit,
                "dirty": dirty,
                "cmd": vars(args)
                | {"resume": str(args.resume), "root": str(args.root)},
            },
            indent=2,
            default=str,
        )
    )
    torch.manual_seed(task.seed)
    if not task.pose_path:
        from pogo_g1 import assembly

        assembly.mounted_pose(task.params, task.asset_dir).save(
            out / "mounted_pose.json"
        )
        task.pose_path = str(out / "mounted_pose.json")
    env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_dict.get("clip_actions"))
    runner = MjlabOnPolicyRunner(wrapped, agent_dict, str(out), device="cuda:0")
    if args.resume:
        runner.load(str(args.resume))
        # Checkpoints are saved after this iteration's optimizer update.
        runner.current_learning_iteration += 1
    elif args.warm_start:
        runner.load(str(args.warm_start), load_cfg={"actor": True})
    print(f"run dir: {out}", flush=True)
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
