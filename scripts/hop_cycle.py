"""Hop-cycle analysis: where does the robot inject energy?

Rolls out a checkpoint in place (apex command --apex) on --worlds worlds, records
the pogo state and joint mechanical power, cuts the record into touchdown-to-
touchdown cycles and plots phase-averaged curves. Saves PNG + NPZ in the run dir.
"""

import argparse
import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402

from pogo_g1 import mdp  # noqa: E402
from pogo_g1.env import make_env_cfg  # noqa: E402
from pogo_g1.evaluate import latest_checkpoint, load_task  # noqa: E402
from pogo_g1.ppo import runner_cfg  # noqa: E402

GROUPS = {
    "hip": ("hip_pitch", "hip_roll", "hip_yaw"),
    "knee": ("knee",),
    "ankle": ("ankle",),
    "waist": ("waist",),
    "arms": ("shoulder", "elbow", "wrist"),
    "fingers": ("_thumb_", "_index_", "_middle_", "_ring_", "_pinky_"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ckpt")
    ap.add_argument("--apex", type=float, default=0.12)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--set", action="append", default=[], help="params.<field>=value")
    args = ap.parse_args()
    ckpt = args.ckpt or latest_checkpoint(args.run)
    task = load_task(args.run)
    task.apex_cap = 0.0
    task.worlds, task.airborne_prob, task.episode_s, task.robust = (
        args.worlds,
        0.0,
        args.seconds + 5,
        False,
    )
    task.apex, task.vx, task.vy, task.yaw_rate = (
        (args.apex,) * 2,
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    for item in args.set:
        k, v = item.split("=")
        task.params = dataclasses.replace(
            task.params, **{k.removeprefix("params."): float(v)}
        )
    env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
    agent = runner_cfg()
    w = RslRlVecEnvWrapper(env)
    r = MjlabOnPolicyRunner(w, dataclasses.asdict(agent), None, device="cuda:0")
    r.load(str(args.run / ckpt), load_cfg={"actor": True}, strict=False)
    pol = r.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    names = list(robot.actuator_names)
    aj = torch.as_tensor(
        [list(robot.joint_names).index(a) for a in names], device="cuda:0"
    )
    gid = {
        g: [i for i, n in enumerate(names) if any(k in n for k in keys)]
        for g, keys in GROUPS.items()
    }
    rec = {
        k_: [] for k_ in ["contact", "x", "xd", "H", "z", "alive", *GROUPS, "spring"]
    }
    obs = w.get_observations()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device="cuda:0")
    with torch.inference_mode():
        for _ in range(int(args.seconds / env.step_dt)):
            obs, _, d, _ = w.step(pol(obs))
            alive &= ~d.bool()
            p = robot.data.actuator_force * robot.data.joint_vel[:, aj]
            x, xd = mdp.compression(env), mdp.compression_vel(env)
            rec["contact"].append(mdp.tip_contact(env).float())
            rec["x"].append(x)
            rec["xd"].append(xd)
            rec["H"].append(mdp.energy_height(env))
            rec["z"].append(env.scene["pogo"].data.root_link_pos_w[:, 2])
            rec["alive"].append(alive.float())
            rec["spring"].append(
                -mdp.spring_force(env) * xd
            )  # power delivered BY the spring to the bodies
            for g, ids in gid.items():
                rec[g].append(p[:, ids].sum(-1))
    R = {k_: torch.stack(v).cpu().numpy() for k_, v in rec.items()}  # (T, N)
    env.close()
    dt = env.step_dt
    # Touchdown indices per world; resample each cycle to 100 phase bins.
    bins, cycles, periods, duty, work = (
        100,
        {k_: [] for k_ in ["x", "H", "spring", *GROUPS]},
        [],
        [],
        {g: [] for g in GROUPS},
    )
    t0 = int(3.0 / dt)
    for j in range(R["contact"].shape[1]):
        if R["alive"][-1, j] < 1:
            continue
        c = R["contact"][t0:, j]
        td = np.where((c[1:] > 0) & (c[:-1] == 0))[0] + 1 + t0
        for a, b in zip(td[:-1], td[1:], strict=True):
            n = b - a
            periods.append(n * dt)
            duty.append(R["contact"][a:b, j].mean())
            ph = np.linspace(0, n - 1, bins)
            for k_ in cycles:
                cycles[k_].append(np.interp(ph, np.arange(n), R[k_][a:b, j]))
            for g in GROUPS:
                work[g].append(R[g][a:b, j].sum() * dt)
    if not periods:
        raise SystemExit(
            "No complete cycles from surviving worlds; no cycle statistics to report."
        )
    C = {k_: np.array(v) for k_, v in cycles.items()}
    phase = np.linspace(0, 100, bins)
    fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax[0].plot(phase, C["x"].mean(0) * 100, label="spring compression [cm]")
    ax[0].plot(phase, C["H"].mean(0) * 100, label="energy height H [cm]")
    ax[0].axvline(np.mean(duty) * 100, color="k", ls=":", lw=1)
    ax[0].legend(fontsize=8)
    ax[1].plot(phase, C["spring"].mean(0), label="spring → bodies [W]", color="k")
    ax[1].legend(fontsize=8)
    for g in GROUPS:
        ax[2].plot(phase, C[g].mean(0), label=f"{g} ({np.mean(work[g]):+.1f} J/hop)")
    ax[2].axvline(np.mean(duty) * 100, color="k", ls=":", lw=1)
    ax[2].set_xlabel("% hop cycle (touchdown → touchdown); dotted: liftoff")
    ax[2].set_ylabel("joint mech. power [W]")
    ax[2].legend(fontsize=8)
    fig.suptitle(
        f"{args.run.name} {ckpt}: apex cmd {args.apex} m, f={1 / np.mean(periods):.2f} Hz, duty {np.mean(duty):.2f}"
    )
    fig.tight_layout()
    tag = f"cycle-{Path(ckpt).stem}-apex{args.apex}" + "".join(
        f"-{s}" for s in args.set
    )
    fig.savefig(args.run / f"{tag}.png", dpi=130)
    np.savez(
        args.run / f"{tag}.npz",
        phase=phase,
        periods=periods,
        duty=duty,
        checkpoint=ckpt,
        **{f"cyc_{k_}": v for k_, v in C.items()},
        **{f"work_{g}": v for g, v in work.items()},
    )
    print(
        f"cycles={len(periods)} freq={1 / np.mean(periods):.3f}Hz duty={np.mean(duty):.3f}",
        {g: round(float(np.mean(v)), 2) for g, v in work.items()},
    )


if __name__ == "__main__":
    main()
