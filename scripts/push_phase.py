"""Push timing within the hop cycle.

  uv run --locked --extra train python scripts/push_phase.py RUN [--ckpt ...]
      [--apex 0.3] [--worlds 256] [--dv 0.3,0.6,0.9]

Deterministic policy hopping in place at a fixed apex command. After 10 s of
settling, each world receives one horizontal push, a constant force on the
pelvis for 0.1 s whose impulse equals the robot + pogo mass times dv (so it
would change the whole system's velocity by dv if nothing pushed back). The
push starts at a pre-defined event of that world's hop cycle:

- stance: the step the tip touches down;
- ascent: the step the tip leaves the ground;
- descent: the step the system centre of mass starts to fall (apex).

The no-push control uses the first touchdown after 10 s as its reference time.
Each world is then followed for 10 s. Measures: falls (all worlds count),
horizontal drift of the pelvis from where the push started (at +3 s and +10 s),
the hop-cycle speed of the first full cycle after the push, and recovery time
(time to the first of two consecutive landings whose hop-cycle speed is within
0.25 m/s of the zero command, the no-push control's own range). Results: RUN/push-phase-<ckpt>.json.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from pogo_g1 import mdp
from pogo_g1.env import make_env_cfg
from pogo_g1.evaluate import latest_checkpoint, load_task
from pogo_g1.ppo import runner_cfg

# TOL: the unpushed policy's own hop-to-hop speed reaches ~0.25 m/s (p90 of
# the no-push control in a 32-world pilot), so recovery means back to that.
SETTLE_S, FOLLOW_S, PUSH_S, TOL = 10.0, 10.0, 0.1, 0.25
PHASES = ("none", "stance", "ascent", "descent")
DIRECTIONS = {"forward": (1.0, 0.0), "left": (0.0, 1.0)}


def rollout(run, ckpt, phase, direction, dv, apex, worlds, freeze=0.0):
    task = load_task(run)
    task.worlds, task.airborne_prob, task.apex_cap = worlds, 0.0, 0.0
    task.episode_s = SETTLE_S + FOLLOW_S + 10.0
    env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
    try:
        w = RslRlVecEnvWrapper(env)
        runner = MjlabOnPolicyRunner(
            w, dataclasses.asdict(runner_cfg()), None, device="cuda:0"
        )
        runner.load(str(run / ckpt), load_cfg={"actor": True}, strict=False)
        policy = runner.get_inference_policy(device="cuda:0")
        env.command_manager.get_term("hop").set_command(apex, 0.0, 0.0, 0.0)
        robot = env.scene["robot"]
        pelvis = [robot.body_names.index("pelvis")]
        n, dev, dt = env.num_envs, env.device, env.step_dt
        mass = mdp.system_mass(env)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        t_push = torch.full((n,), float("nan"), device=dev)
        p0 = torch.zeros(n, 2, device=dev)
        push_left = torch.zeros(n, dtype=torch.long, device=dev)
        force = torch.zeros(n, 1, 3, device=dev)
        drift3 = torch.full((n,), float("nan"), device=dev)
        drift10 = torch.full((n,), float("nan"), device=dev)
        first_cycle = torch.full((n,), float("nan"), device=dev)
        recovered = torch.full((n,), float("nan"), device=dev)
        landings_after = torch.zeros(n, dtype=torch.long, device=dev)
        ok_streak = torch.zeros(n, dtype=torch.long, device=dev)
        prev_contact = mdp.tip_contact(env)
        prev_vz = mdp.system_com_velocity(env)[:, 2]
        steps = int((SETTLE_S + FOLLOW_S + 2.0) / dt)
        obs = w.get_observations()
        ux, uy = DIRECTIONS[direction]
        n_push = round(PUSH_S / dt)
        # Control: hold the action from the push start for `freeze` seconds,
        # so what remains is the passive response of body, stick and contact.
        held, freeze_left = None, torch.zeros(n, dtype=torch.long, device=dev)
        speed_end = torch.full((n,), float("nan"), device=dev)
        with torch.inference_mode():
            for i in range(steps):
                t = i * dt
                act = policy(obs)
                if held is None:
                    held = act.clone()
                act = torch.where((freeze_left > 0)[:, None], held, act)
                held = torch.where((freeze_left > 0)[:, None], held, act)
                freeze_left = (freeze_left - 1).clamp(min=0)
                obs, _, dones, _ = w.step(act)
                alive &= ~dones.bool()
                contact = mdp.tip_contact(env)
                vz = mdp.system_com_velocity(env)[:, 2]
                event = {
                    "none": contact & ~prev_contact,
                    "stance": contact & ~prev_contact,
                    "ascent": ~contact & prev_contact,
                    "descent": ~contact & (prev_vz > 0) & (vz <= 0),
                }[phase]
                start = event & alive & t_push.isnan() & (t >= SETTLE_S)
                if start.any():
                    t_push[start] = t
                    p0[start] = robot.data.root_link_pos_w[start, :2]
                    if phase != "none":
                        h = robot.data.heading_w[start]
                        f = (mass[start] * dv / PUSH_S)[:, None]
                        fx = f[:, 0] * (torch.cos(h) * ux - torch.sin(h) * uy)
                        fy = f[:, 0] * (torch.sin(h) * ux + torch.cos(h) * uy)
                        force[start, 0, 0], force[start, 0, 1] = fx, fy
                        push_left[start] = n_push
                        freeze_left[start] = round(freeze / dt)
                active = push_left > 0
                robot.write_external_wrench_to_sim(
                    torch.where(active[:, None, None], force, 0.0),
                    torch.zeros_like(force),
                    body_ids=pelvis,
                )
                push_left = (push_left - 1).clamp(min=0)
                since = t - t_push
                ended = alive & (since >= PUSH_S) & speed_end.isnan()
                v = torch.linalg.vector_norm(
                    mdp.system_com_velocity(env)[:, :2], dim=-1
                )
                speed_end[ended] = v[ended]
                pos = robot.data.root_link_pos_w[:, :2]
                d = torch.linalg.vector_norm(pos - p0, dim=-1)
                live = alive & ~t_push.isnan()
                at3 = live & (since >= 3.0) & drift3.isnan()
                drift3[at3] = d[at3]
                at10 = live & (since >= FOLLOW_S) & drift10.isnan()
                drift10[at10] = d[at10]
                # A full hop cycle that started after the push ended.
                land = mdp.hops(env).bool() & live & (since > PUSH_S)
                landings_after += land
                speed = torch.linalg.vector_norm(env._land_motion[:, :2], dim=-1)
                full = land & (landings_after >= 2) & first_cycle.isnan()
                first_cycle[full] = speed[full]
                ok = land & (speed <= TOL)
                ok_streak = torch.where(
                    land, torch.where(ok, ok_streak + 1, 0), ok_streak
                )
                rec = land & (ok_streak == 2) & recovered.isnan()
                # Time of the first of the two good landings: one cycle back.
                recovered[rec] = since[rec] - env._land_period[rec]
                prev_contact, prev_vz = contact, vz
                if (
                    not (t_push.isnan() & alive).any()
                    and (since[~t_push.isnan()] > FOLLOW_S).all()
                ):
                    break
        pushed = ~t_push.isnan()
        fell_before = ~pushed & ~alive

        def stats(x):
            x = x[pushed & alive].cpu().numpy()
            x = x[~np.isnan(x)]
            if not x.size:
                return None
            return {
                "mean": float(x.mean()),
                "p50": float(np.median(x)),
                "p90": float(np.percentile(x, 90)),
            }

        return {
            "phase": phase,
            "direction": direction,
            "dv": dv,
            "worlds": n,
            "pushed": int(pushed.sum()),
            "fell_before_push": int(fell_before.sum()),
            "falls": float((pushed & ~alive).float().sum() / max(int(pushed.sum()), 1)),
            "not_recovered": float(
                ((pushed & alive & recovered.isnan()).float().sum())
                / max(int((pushed & alive).sum()), 1)
            ),
            "recovery_s": stats(recovered),
            "drift_3s_m": stats(drift3),
            "drift_10s_m": stats(drift10),
            "first_cycle_speed": stats(first_cycle),
            "speed_at_push_end": stats(speed_end),
            "freeze_s": freeze,
        }
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ckpt")
    ap.add_argument("--apex", type=float, default=0.3)
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--dv", default="0.3,0.6,0.9")
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--directions", default=",".join(DIRECTIONS))
    ap.add_argument("--tag", default="", help="suffix for the results file")
    ap.add_argument(
        "--freeze",
        type=float,
        default=0.0,
        help="hold the action [s] from the push start",
    )
    args = ap.parse_args()
    ckpt = args.ckpt or latest_checkpoint(args.run)
    tag = (f"-frozen{args.freeze:g}" if args.freeze else "") + (
        f"-{args.tag}" if args.tag else ""
    )
    out = args.run / f"push-phase-{Path(ckpt).stem}-apex{args.apex:g}{tag}.json"
    results = json.loads(out.read_text())["results"] if out.exists() else {}
    jobs = [("none", "forward", 0.0)] + [
        (p, d, float(v))
        for v in args.dv.split(",")
        for d in args.directions.split(",")
        for p in args.phases.split(",")
        if p != "none"
    ]
    for phase, direction, dv in jobs:
        key = f"{phase}-{direction}-{dv:g}"
        if key in results or phase not in args.phases.split(","):
            continue
        r = rollout(
            args.run, ckpt, phase, direction, dv, args.apex, args.worlds, args.freeze
        )
        results[key] = r
        print(key, json.dumps(r), flush=True)
        out.write_text(
            json.dumps(
                {
                    "checkpoint": ckpt,
                    "apex": args.apex,
                    "settle_s": SETTLE_S,
                    "follow_s": FOLLOW_S,
                    "push_s": PUSH_S,
                    "recovery_tolerance": TOL,
                    "results": results,
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
