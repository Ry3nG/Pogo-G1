"""Policy evaluation: one condition = one batch of worlds from the static mount.

Metrics (per condition):
  success        fraction surviving with continuous valid hops (see HoppingContinuity)
  survival       fraction with no physical failure within the horizon
  hops           valid hops per surviving world
  tip_apex_err   mean |measured peak tip clearance - command| over hops in the last
                 two thirds of the run (height climbs over several hops from
                 rest: ~8 s to 0.75 m) [m]
  vel_err        mean |COM planar velocity - command| after the first 3 s [m/s]
  yaw_err        mean |yaw rate - command| after 3 s [rad/s]
  hop_vel_err    mean |hop-cycle average COM planar velocity - command|, landing to
                 landing after 3 s [m/s]. Excludes the within-hop balance swing
                 that vel_err includes even when hopping in place.
  hop_yaw_err    the same for the hop-cycle average yaw rate [rad/s]
  hop_cyaw_err   the same for the centroidal yaw rate (mdp.centroidal_yaw_rate)
  grip_f / grip_t  per-hand net grip force [N] / moment [N m] percentiles (p50, p99)
  power          mean positive mechanical motor power [W]
  wrist_sat      fraction of samples with wrist pitch torque at its limit
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from pogo_g1 import contact, mdp, pogo
from pogo_g1.env import TaskCfg, make_env_cfg
from pogo_g1.ppo import runner_cfg


class HoppingContinuity:
    """Require a first valid hop within 3 s, then no gap longer than 2 s.

    Survival alone can be achieved by standing on a finite contact patch. A late
    hop or automatic reset cannot recover a previously failed first episode.
    """

    def __init__(self, worlds, device):
        self.deadline = torch.full((worlds,), 3.0, device=device)
        self.valid = torch.ones(worlds, dtype=torch.bool, device=device)
        self.has_hopped = torch.zeros_like(self.valid)

    def update(self, elapsed, hop, alive):
        self.valid &= alive & (elapsed <= self.deadline + 1e-6)
        landed = hop.bool() & alive
        self.has_hopped |= landed
        self.deadline = torch.where(landed, elapsed + 2.0, self.deadline)

    @property
    def success(self):
        return self.valid & self.has_hopped


def load_task(run: Path, training: bool = False) -> TaskCfg:
    """The run's task. Unless resuming training, the hand is at its rated
    strength: a derated training hand is not the hand being evaluated."""
    config = json.loads((Path(run) / "config.json").read_text())
    if config.get("physics_version") != pogo.PHYSICS_VERSION:
        raise ValueError(
            f"Checkpoint physics {config.get('physics_version', 'legacy-linear')} "
            f"does not match {pogo.PHYSICS_VERSION}. Replay v4 at 81ea3ed, v3 at 7e84e85, "
            "v2 at 33c4dde or legacy-linear at ce8526b; "
            "do not score them under a different physical model."
        )
    saved = config["task"]
    task = TaskCfg()
    for k, v in saved.items():
        if k in ("params", "contacts") or not hasattr(task, k):
            continue
        cur = getattr(task, k)
        if isinstance(cur, Path):
            v = Path(v)
        elif isinstance(cur, tuple):
            v = tuple(v)
        setattr(task, k, v)
    task.apex_cap = saved.get("apex_cap", 0.0)  # older runs had no curriculum
    if not training:
        task.grip_strength = 1.0
    task.base_velocity = saved.get("base_velocity", False)
    task.params = pogo.PogoParams(**saved["params"])
    task.contacts = contact.ContactParams(**saved["contacts"])
    if (Path(run) / "mounted_pose.json").exists():
        task.pose_path = str(Path(run) / "mounted_pose.json")
    return task


def latest_checkpoint(run: Path) -> str:
    return sorted(
        Path(run).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])
    )[-1].name


class Evaluator:
    """Keeps one policy; builds a fresh env per condition."""

    def __init__(self, run: Path, ckpt: str | None = None):
        self.run = Path(run)
        self.ckpt = ckpt or latest_checkpoint(self.run)

    def __call__(
        self,
        overrides: dict,
        worlds=256,
        seconds=30.0,
        record_world=None,
        command_steps=None,
    ):
        """Optional (time_s, [apex, vx, vy, yaw]) steps share one uninterrupted episode."""
        task = load_task(self.run)
        task.worlds, task.airborne_prob, task.episode_s = worlds, 0.0, seconds + 5
        task.in_place_frac = 0.0
        task.apex_cap = 0.0  # fixed commands: no training curriculum
        for k, v in overrides.items():
            if "." in k:
                field, subfield = k.split(".", 1)
                setattr(
                    task,
                    field,
                    dataclasses.replace(getattr(task, field), **{subfield: v}),
                )
            else:
                setattr(task, k, v)
        env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
        try:
            return self._rollout(env, task, seconds, record_world, command_steps)
        finally:
            env.close()

    def _rollout(self, env, task, seconds, record_world, command_steps=None):
        agent = runner_cfg()
        w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(
            w, dataclasses.asdict(agent), None, device="cuda:0"
        )
        runner.load(str(self.run / self.ckpt), load_cfg={"actor": True}, strict=False)
        policy = runner.get_inference_policy(device="cuda:0")
        obs = w.get_observations()
        n, dev = env.num_envs, env.device
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        continuity = HoppingContinuity(n, dev)
        hops = torch.zeros(n, device=dev)
        tip_apex_err, tip_apex, vel_err, yaw_err = [], [], [], []
        hop_vel_err, hop_yaw_err, hop_cyaw_err = [], [], []
        cycle_v = torch.zeros(n, 2, device=dev)
        cycle_w = torch.zeros(n, device=dev)
        cycle_c = torch.zeros(n, device=dev)
        cycle_t = torch.zeros(n, device=dev)
        landed_before = torch.zeros(n, dtype=torch.bool, device=dev)
        grip_f, grip_t, power, wrist = [], [], [], []
        # What ends each world's run, and how much of the stroke it uses.
        tm = env.termination_manager
        causes = [t for t in tm.active_terms if t != "timeout"]
        fall_cause = {t: 0 for t in causes}
        stroke_max = torch.zeros(n, device=dev)
        robot = env.scene["robot"]
        wrist_ids = [
            i for i, a in enumerate(robot.actuator_names) if "wrist_pitch" in a
        ]
        limits = torch.as_tensor(
            [
                env.sim.mj_model.actuator(f"robot/{a}").forcerange[1]
                for a in robot.actuator_names
            ],
            device=dev,
        )
        act_joint = torch.as_tensor(
            [list(robot.joint_names).index(a) for a in robot.actuator_names], device=dev
        )
        traj = []
        segments = []
        segment_vel, segment_yaw = [], []
        if command_steps:
            command = env.command_manager.get_term("hop")
            for start, values in command_steps:
                segments.append(
                    {
                        "start_s": start,
                        "command": list(values),
                        "observation_effective_s": start + env.step_dt,
                    }
                )
                segment_vel.append([])
                segment_yaw.append([])
        segment = -1
        steps = int(seconds / env.step_dt)
        # Height climbs over several hops from rest (~8 s to 0.75 m), so the
        # apex metrics use the last two thirds of the run (after 10 s of 30 s).
        settle_s = max(3.0, seconds / 3)
        with torch.inference_mode():
            for i in range(steps):
                if command_steps and segment + 1 < len(command_steps):
                    start, values = command_steps[segment + 1]
                    if i * env.step_dt >= start - 1e-9:
                        segment += 1
                        command.set_command(*values)
                        # The next normal env.step adds this command to history.
                        # Recomputing history here would insert a synthetic frame.
                obs, _, dones, _ = w.step(policy(obs))
                died = dones.bool() & alive & ~tm.get_term("timeout").bool()
                if died.any():
                    for t in causes:
                        fall_cause[t] += int((tm.get_term(t).bool() & died).sum())
                alive &= ~dones.bool()
                stroke_max = torch.where(
                    alive, torch.maximum(stroke_max, mdp.compression(env)), stroke_max
                )
                hop = mdp.hops(env).bool() & alive
                hops += hop
                continuity.update((i + 1) * env.step_dt, hop, alive)
                cmd = mdp.hop_command(env)
                settled = hop & (i * env.step_dt > settle_s)
                if settled.any():
                    tip_apex.append(env._land_apex[settled].cpu())
                    tip_apex_err.append(
                        (env._land_apex[settled] - cmd[settled, 0]).abs().cpu()
                    )
                wz = robot.data.root_link_ang_vel_w[:, 2]
                cycle_v += mdp.com_velocity_yaw(env) * env.step_dt
                cycle_w += wz * env.step_dt
                cycle_c += mdp.centroidal_yaw_rate(env) * env.step_dt
                cycle_t += env.step_dt
                done = hop & landed_before & (i * env.step_dt > 3.0)
                if done.any():
                    t = cycle_t[done, None]
                    hop_vel_err.append(
                        torch.linalg.vector_norm(
                            cycle_v[done] / t - cmd[done, 1:3], dim=-1
                        ).cpu()
                    )
                    hop_yaw_err.append(
                        (cycle_w[done] / t[:, 0] - cmd[done, 3]).abs().cpu()
                    )
                    hop_cyaw_err.append(
                        (cycle_c[done] / t[:, 0] - cmd[done, 3]).abs().cpu()
                    )
                landed_before |= hop
                cycle_v[hop], cycle_w[hop], cycle_c[hop], cycle_t[hop] = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                if i * env.step_dt > 3.0 and alive.any():
                    vel_err.append(
                        torch.linalg.vector_norm(
                            mdp.com_velocity_yaw(env) - cmd[:, 1:3], dim=-1
                        )[alive].cpu()
                    )
                    yaw_err.append((wz - cmd[:, 3]).abs()[alive].cpu())
                if command_steps:
                    segment_vel[segment].append(
                        torch.linalg.vector_norm(
                            mdp.com_velocity_yaw(env) - cmd[:, 1:3], dim=-1
                        )[alive].cpu()
                    )
                    segment_yaw[segment].append(
                        (robot.data.root_link_ang_vel_w[:, 2] - cmd[:, 3])
                        .abs()[alive]
                        .cpu()
                    )
                    segments[segment]["survival_end"] = alive.float().mean().item()
                f, t = mdp.grip_wrench(env)
                grip_f.append(f[alive].flatten().cpu())
                grip_t.append(t[alive].flatten().cpu())
                tau = robot.data.actuator_force
                qd = robot.data.joint_vel[:, act_joint]
                power.append((tau * qd).clamp(min=0).sum(-1)[alive].cpu())
                wrist.append(
                    (tau[:, wrist_ids].abs() >= 0.99 * limits[wrist_ids])
                    .float()[alive]
                    .flatten()
                    .cpu()
                )
                if record_world is not None:
                    traj.append(env.sim.data.qpos[record_world].cpu().numpy().copy())

        def cat(v):
            return torch.cat(v).numpy() if v else np.empty(0)

        def mean(v):
            values = cat(v)
            return float(np.mean(values)) if values.size else None

        def percentile(values, q):
            return float(np.percentile(values, q)) if values.size else None

        gf, gt = cat(grip_f), cat(grip_t)
        res = {
            "success": continuity.success.float().mean().item(),
            "survival": alive.float().mean().item(),
            "hops": hops[alive].mean().item() if alive.any() else None,
            "tip_apex_err": mean(tip_apex_err),
            "tip_apex_mean": mean(tip_apex),
            "vel_err": mean(vel_err),
            "yaw_err": mean(yaw_err),
            "hop_vel_err": mean(hop_vel_err),
            "hop_yaw_err": mean(hop_yaw_err),
            "hop_cyaw_err": mean(hop_cyaw_err),
            "grip_f_p50": percentile(gf, 50),
            "grip_f_p99": percentile(gf, 99),
            "grip_t_p50": percentile(gt, 50),
            "grip_t_p99": percentile(gt, 99),
            "power_w": mean(power),
            "wrist_sat": mean(wrist),
            "stroke_frac_max": (stroke_max.max() / task.params.stroke).item(),
            "stroke_frac_p50": (stroke_max.median() / task.params.stroke).item(),
            "fall_cause": {t: c for t, c in fall_cause.items() if c},
        }
        if command_steps:
            for j, result in enumerate(segments):
                result.update(
                    end_s=segments[j + 1]["start_s"]
                    if j + 1 < len(segments)
                    else seconds,
                    vel_err=mean(segment_vel[j]),
                    yaw_err=mean(segment_yaw[j]),
                    alive_samples=int(cat(segment_vel[j]).size),
                )
            res["segments"] = segments
            res["segment_metrics"] = (
                "All samples from command issue, including transition; only samples "
                "before each world's first failure. Survival denominator is all worlds. "
                "Commands enter observations after one policy step."
            )
        if record_world is not None:
            res["_traj"] = traj
            res["_origin"] = env.scene.env_origins[record_world].cpu().numpy()
            res["_model"] = env.sim.mj_model
        return res
