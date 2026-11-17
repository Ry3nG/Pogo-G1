"""Each learner's own spread of the quantity its yaw term scores, under its
own sampled actions, per checkpoint.

  record: own_fluctuation.py record OUTDIR RUN CKPT [CKPT ...] [--unclamped]
  table:  own_fluctuation.py table OUTDIR

`record` repeats the `step_score.py ... noise` protocol: hop in place at
0.15 m, 64 worlds, 20 s with the first 5 s skipped, actions sampled as in
training, yaw command 0.5 ("turn") and 0 ("still"). The action distribution
is built from the run's own config, so a `--max-std` cap applies as in
training; `step_score.py` builds the default distribution, which drops the
cap, and `--unclamped` reproduces that. Each recording keeps the pelvis and
centroidal yaw rates (steps x worlds), every hop cycle's mean pelvis yaw rate
and period, tip contact, terminations and the action std:
OUTDIR/<run>/<ckpt>-<turn|still>[-unclamped].npz.

`table` writes OUTDIR/own-fluctuation.json and .md. Spreads come from the
turn recording, with the definitions behind artifacts/steering-conditions.csv:
pelvis yaw rate, std over all steps and worlds (step_score.py); hop-cycle
mean, std over all hops (the "hop-average spread"); moving average with tau,
std over time per world, world mean, after max(1.5 s, 3 tau)
(steering_table.py); L_z/I_zz, std over time per world, world mean
(yaw_signals.py). The scored spread is the one the run's yaw term scores.
Gains are reward per second at unit weight, S(e) = 1/(1+(e/0.15)^2) against
0.5: "own" is the turn recording minus the still one (yaw_signals.py gain);
"at own spread" shifts the turn recording's scored quantity to mean 0.5 and
to mean 0 and takes the difference, so it does not depend on how much the
learner turns yet.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np

SIGMA, CMD = 0.15, 0.5
CONDITION = {
    "hop": "per hop",
    "step": "per step",
    "window": "window",
    "centroidal": "momentum",
}


def record(outdir, run, cks, unclamped):
    import dataclasses

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from pogo_g1 import mdp
    from pogo_g1.env import make_env_cfg
    from pogo_g1.evaluate import load_task
    from pogo_g1.ppo import runner_cfg

    agent = dataclasses.asdict(runner_cfg())
    own = json.loads((run / "config.json").read_text())["agent"]
    own_dist = own["actor"]["distribution_cfg"]
    if not unclamped:
        agent["actor"]["distribution_cfg"] = own_dist
    tag = "-unclamped" if unclamped else ""
    (outdir / run.name).mkdir(parents=True, exist_ok=True)
    for ck in cks:
        if not (run / ck).exists():
            print(f"MISSING {run.name} {ck}", flush=True)
            continue
        for name, wz in (("turn", CMD), ("still", 0.0)):
            out = outdir / run.name / f"{Path(ck).stem}-{name}{tag}.npz"
            if out.exists():
                continue
            task = load_task(run)
            task.worlds, task.airborne_prob = 64, 0.0
            task.episode_s, task.apex_cap = 30.0, 0.0
            env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
            w = RslRlVecEnvWrapper(env)
            r = MjlabOnPolicyRunner(w, copy.deepcopy(agent), None, device="cuda:0")
            r.load(str(run / ck), load_cfg={"actor": True}, strict=False)
            pol = r.get_inference_policy(device="cuda:0")
            d = pol.distribution
            std = d.std_param.detach().clamp(d.std_range[0], d.std_range[1])
            env.command_manager.get_term("hop").set_command(0.15, 0.0, 0.0, wz)
            rec = {k: [] for k in ("pelvis", "centroidal", "contact", "vel", "done")}
            hop_yaw, hop_t, hop_w = [], [], []
            obs = w.get_observations()
            with torch.inference_mode():
                for i in range(int(20.0 / env.step_dt)):
                    obs, _, done, _ = w.step(pol(obs, stochastic_output=True))
                    hop = mdp.hops(env).bool()
                    if i * env.step_dt < 5.0:
                        continue
                    # clone: yaw_rate is a view of a buffer updated in place
                    rec["pelvis"].append(mdp.yaw_rate(env).clone())
                    rec["centroidal"].append(mdp.centroidal_yaw_rate(env).clone())
                    rec["contact"].append(mdp.tip_contact(env).clone())
                    rec["vel"].append(mdp.com_velocity_yaw(env)[:, :2].clone())
                    rec["done"].append(done.bool().clone())
                    if hop.any():
                        hop_yaw.append(env._land_motion[hop, 2].cpu().numpy())
                        hop_t.append(env._land_period[hop].cpu().numpy())
                        hop_w.append(torch.nonzero(hop)[:, 0].cpu().numpy())
            rec = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
            np.savez(
                out,
                command=wz,
                dt=env.step_dt,
                # a policy that does not hop yet records no hop cycles
                hop_yaw=np.concatenate(hop_yaw) if hop_yaw else np.zeros(0),
                hop_t=np.concatenate(hop_t) if hop_t else np.zeros(0),
                hop_world=np.concatenate(hop_w) if hop_w else np.zeros(0, int),
                action_std=std.cpu().numpy(),
                std_param=d.std_param.detach().cpu().numpy(),
                std_range=np.array(d.std_range, dtype=float),
                train_std_range=np.array(own_dist.get("std_range", [1e-6, 1e6])),
                steer_timescale=task.steer_timescale,
                steer_tau=task.steer_tau,
                yaw_weight=task.yaw_weight,
                **rec,
            )
            print(
                f"{run.name} {ck} {name}{tag}: pelvis mean {rec['pelvis'].mean():.3f}"
                f" std {rec['pelvis'].std():.3f}, centroidal std"
                f" {np.std(rec['centroidal'], 0).mean():.3f}, action std"
                f" {std.mean():.3f}, terminations {int(rec['done'].sum())}",
                flush=True,
            )
            env.close()
            del r, w, env, pol
            torch.cuda.empty_cache()


def score(e):
    return 1.0 / (1.0 + (e / SIGMA) ** 2)


def ema(y, dt, tau):
    """As steering_table.py: from the first recorded step, start-up skipped."""
    out, m = np.empty_like(y), y[0].copy()
    for i in range(len(y)):
        m = m + dt / tau * (y[i] - m)
        out[i] = m
    return out[int(max(1.5, 3 * tau) / dt) :]


def signals(d):
    """Each candidate scored quantity: (values, weights or None, spread)."""
    dt = float(d["dt"])
    out = {
        "pelvis": (d["pelvis"], None, float(d["pelvis"].std())),
        "hop": (
            d["hop_yaw"],
            d["hop_t"],
            float(d["hop_yaw"].std()) if len(d["hop_yaw"]) else float("nan"),
        ),
        "centroidal": (
            d["centroidal"],
            None,
            float(np.std(d["centroidal"], axis=0).mean()),
        ),
    }
    for tau in (0.1, 0.5, 2.0):
        y = ema(d["pelvis"], dt, tau)
        out[f"ema{tau:g}"] = (y, None, float(np.std(y, axis=0).mean()))
    return out


def rate(y, w, shift=0.0):
    if len(y) == 0:
        return float("nan")
    s = score(y + shift - CMD)
    return float(np.sum(s * w) / np.sum(w)) if w is not None else float(s.mean())


def scored_key(ts, tau):
    if ts == "window":
        return f"ema{tau:g}"
    return {"hop": "hop", "step": "pelvis", "centroidal": "centroidal"}[ts]


def condition(run, ts, tau, weight, capped):
    if ts == "window":
        return f"window {tau:g} s"
    if ts == "step" and weight > 2.0:
        return "raised weight"
    if ts == "step" and capped:
        return "capped noise"
    return CONDITION[ts]


def table(outdir):
    rows = []
    for f in sorted(outdir.glob("*/model_*-turn*.npz")):
        unclamped = f.stem.endswith("-unclamped")
        still = f.with_name(f.name.replace("-turn", "-still"))
        t = np.load(f)
        s = np.load(still) if still.exists() else None
        ts, tau = str(t["steer_timescale"]), float(t["steer_tau"])
        weight = float(t["yaw_weight"])
        capped = float(t["train_std_range"][1]) < 1e5
        key = scored_key(ts, tau)
        sig_t = signals(t)
        sig_s = signals(s) if s is not None else None
        row = {
            "run": f.parent.name,
            "checkpoint": int(f.stem.split("_")[1].split("-")[0]),
            "condition": condition(f.parent.name, ts, tau, weight, capped),
            "sampling": "std from stored param, cap dropped (as step_score.py)"
            if unclamped
            else "as in training (run's own distribution config)",
            "scored_quantity": key,
            "scored_spread": sig_t[key][2],
            "raw_pelvis_spread": sig_t["pelvis"][2],
            "raw_pelvis_within_world": float(np.std(t["pelvis"], 0).mean()),
            "mean_yaw_turn": float(t["pelvis"].mean()),
            "mean_yaw_still": float(s["pelvis"].mean()) if s is not None else None,
            "action_std_mean": float(t["action_std"].mean()),
            "std_param_mean": float(t["std_param"].mean()),
            "terminations_turn": int(t["done"].sum()),
            "terminations_still": int(s["done"].sum()) if s is not None else None,
            "hops_per_world_s": len(t["hop_yaw"])
            / (t["pelvis"].shape[0] * float(t["dt"]) * t["pelvis"].shape[1]),
            "worlds": int(t["pelvis"].shape[1]),
            "seconds": round(t["pelvis"].shape[0] * float(t["dt"]), 3),
            "spreads": {k: v[2] for k, v in sig_t.items()},
        }
        y, w, _ = sig_t[key]
        if len(y) == 0:
            mu = float("nan")
        elif w is not None:
            mu = float(np.sum(y * w) / np.sum(w))
        else:
            mu = float(y.mean())
        row["gain_at_own_spread"] = rate(y, w, CMD - mu) - rate(y, w, -mu)
        if s is not None:
            ys, ws, _ = sig_s[key]
            row["gain_own"] = rate(y, w) - rate(ys, ws)
            row["gains_own_all"] = {
                k: rate(*sig_t[k][:2]) - rate(*sig_s[k][:2]) for k in sig_t
            }
        rows.append(row)
    order = [
        "per hop",
        "per step",
        "window 0.1 s",
        "window 0.5 s",
        "window 2 s",
        "momentum",
        "raised weight",
        "capped noise",
    ]
    rows.sort(
        key=lambda r: (
            order.index(r["condition"]),
            r["sampling"],
            r["checkpoint"],
            r["run"][-2:],
        )
    )
    (outdir / "own-fluctuation.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = [
        "| condition | run | updates | scored quantity | scored spread | raw pelvis spread"
        " | gain own (turn-still) | gain at own spread | mean yaw at 0.5 / 0"
        " | action std | falls (turn/still) | worlds x s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    names = {
        "pelvis": "pelvis yaw rate",
        "hop": "hop-cycle mean",
        "centroidal": "L_z/I_zz",
    }
    for r in rows:
        q = names.get(r["scored_quantity"], f"EMA tau {r['scored_quantity'][3:]} s")
        cond = r["condition"] + (" (cap dropped)" if "dropped" in r["sampling"] else "")
        ms = r["mean_yaw_still"]
        g = r.get("gain_own")
        lines.append(
            f"| {cond} | {r['run']} | {r['checkpoint']} | {q} | {r['scored_spread']:.3f}"
            f" | {r['raw_pelvis_spread']:.3f} | {'-' if g is None else f'{g:.3f}'}"
            f" | {r['gain_at_own_spread']:.3f}"
            f" | {r['mean_yaw_turn']:.2f} / {'-' if ms is None else f'{ms:.2f}'}"
            f" | {r['action_std_mean']:.3f} | {r['terminations_turn']}/{r['terminations_still']}"
            f" | {r['worlds']} x {r['seconds']:g} |"
        )
    lines += ["", *summary(rows)]
    (outdir / "own-fluctuation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


CSV_NAME = {
    "per hop": "per hop",
    "per step": "per step",
    "window 0.1 s": "window 0.1 s",
    "window 0.5 s": "window 0.5 s",
    "window 2 s": "window 2 s",
    "momentum": "centroidal per step",
    "raised weight": "gain-matched per step (yaw weight x3.4)",
    "capped noise": "capped noise (max std 0.25)",
}


def summary(rows):
    """Seed ranges per condition and checkpoint, next to the CSV's values."""
    import csv

    ref = {
        r["condition"]: r
        for r in csv.DictReader(Path("artifacts/steering-conditions.csv").open())
    }
    groups = {}
    for r in rows:
        k = (r["condition"], "dropped" in r["sampling"], r["checkpoint"])
        groups.setdefault(k, []).append(r)

    def rng(rs, key, fmt="{:.2f}"):
        v = [x[key] for x in rs if x.get(key) is not None]
        if not v:
            return "-"
        lo, hi = fmt.format(min(v)), fmt.format(max(v))
        return lo if lo == hi else f"{lo}-{hi}"

    out = [
        "| condition | updates | seeds | scored spread (CSV) | raw pelvis spread (CSV)"
        " | gain own | gain at own spread (CSV gain) | action std |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (cond, dropped, ck), rs in groups.items():
        c = ref[CSV_NAME[cond]]
        seeds = ",".join(x["run"][-1] for x in rs)
        out.append(
            f"| {cond}{' (cap dropped)' if dropped else ''} | {ck} | {seeds}"
            f" | {rng(rs, 'scored_spread')} ({c['scored_swing_rad_s']})"
            f" | {rng(rs, 'raw_pelvis_spread')} ({c['raw_yaw_swing_rad_s']})"
            f" | {rng(rs, 'gain_own', '{:.3f}')}"
            f" | {rng(rs, 'gain_at_own_spread', '{:.3f}')} ({c['turning_gain_per_s']})"
            f" | {rng(rs, 'action_std_mean')} |"
        )
    return out


if __name__ == "__main__":
    if sys.argv[1] == "record":
        args = [a for a in sys.argv[2:] if a != "--unclamped"]
        record(Path(args[0]), Path(args[1]), args[2:], "--unclamped" in sys.argv)
    else:
        table(Path(sys.argv[2]))
