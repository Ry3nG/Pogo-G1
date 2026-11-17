"""Reward timescale on a second periodic task: mjlab's G1 flat walking.

mjlab's recipe scores yaw rate every step with a wide kernel (std 0.71 rad/s).
Here only the yaw-rate term changes: the pogo's kernel 1/(1+(e/std)^2) (or
mjlab's exp(-(e/std)^2), `--kernel exp`) on the pelvis yaw rate, scored on the
instantaneous rate ("step") or on its exponential moving average ("window",
time constant tau); xy angular velocity is dropped from the term.
Heading control is off, so the yaw command is a constant rate between
resamples. Everything else is mjlab's Mjlab-Velocity-Flat-Unitree-G1.

  train:  walk_timescale.py train NAME --std 0.15 --score step [--tau 0.5] [--seed 0] [--iters 3000]
  record: walk_timescale.py record RUN_DIR CKPT OUT.npz [noise]
  gain:   walk_timescale.py gain OUT.npz --std 0.15     (offline turning gain)
  eval:   walk_timescale.py eval RUN_DIR CKPT          (cycle-averaged yaw error)
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-Unitree-G1"
ROOT = Path("artifacts/walk")


def yaw_error(env):
    rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return rate - env.command_manager.get_command("twist")[:, 2]


def kernel(e, std, cauchy):
    """mjlab's exp(-(e/std)^2), or the pogo's bounded 1/(1+(e/std)^2), which
    still pays 0.08 at a 0.5 rad/s error with std 0.15 (exp: 1.5e-5)."""
    return 1.0 / (1.0 + (e / std) ** 2) if cauchy else torch.exp(-((e / std) ** 2))


def step_yaw(env, std: float, cauchy: bool):
    return kernel(yaw_error(env), std, cauchy)


def window_yaw(env, std: float, tau: float, cauchy: bool):
    """Moving average of the yaw rate, restarted at each episode's first step."""
    rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    if getattr(env, "_ema_step", None) != env.common_step_counter:
        if not hasattr(env, "_ema"):
            env._ema = rate.clone()
        first = env.episode_length_buf <= 1
        a = env.step_dt / tau
        env._ema = torch.where(first, rate, env._ema + a * (rate - env._ema))
        env._ema_step = env.common_step_counter
    err = env._ema - env.command_manager.get_command("twist")[:, 2]
    return kernel(err, std, cauchy)


def make_env_cfg(std, score, tau, seed, worlds=4096, cauchy=True):
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = worlds
    cfg.seed = seed
    cfg.commands["twist"].heading_command = False
    cfg.commands["twist"].rel_heading_envs = 0.0
    cfg.commands["twist"].ranges.heading = None
    term = cfg.rewards["track_angular_velocity"]
    if score == "step":
        cfg.rewards["track_angular_velocity"] = RewardTermCfg(
            func=step_yaw, weight=term.weight, params={"std": std, "cauchy": cauchy}
        )
    else:
        cfg.rewards["track_angular_velocity"] = RewardTermCfg(
            func=window_yaw,
            weight=term.weight,
            params={"std": std, "tau": tau, "cauchy": cauchy},
        )
    return cfg


def make_runner(env, seed, log_dir):
    agent = load_rl_cfg(TASK)
    agent.seed = seed
    agent.logger = "tensorboard"
    runner_cls = load_runner_cls(TASK)
    w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    return w, runner_cls(w, dataclasses.asdict(agent), log_dir, "cuda:0")


def train(a):
    run = ROOT / a.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(a) | {"task": TASK}, indent=1))
    env = ManagerBasedRlEnv(
        make_env_cfg(a.std, a.score, a.tau, a.seed, a.worlds, a.kernel == "cauchy"),
        device="cuda:0",
    )
    _, runner = make_runner(env, a.seed, str(run))
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)


def load_policy(run, ckpt, worlds, cmd=None):
    c = json.loads((Path(run) / "config.json").read_text())
    cauchy = c.get("kernel", "exp") == "cauchy"  # the first control ran mjlab's
    cfg = make_env_cfg(c["std"], c["score"], c["tau"], 0, worlds, cauchy)
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.events.pop("push_robot", None)  # no pushes in evaluation
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    w, runner = make_runner(env, 0, None)
    runner.load(str(Path(run) / ckpt), load_cfg={"actor": True}, strict=False)
    return env, w, runner.get_inference_policy(device="cuda:0")


def hold(env, vx, wz):
    term = env.command_manager.get_term("twist")
    term.vel_command_b[:, 0], term.vel_command_b[:, 1] = vx, 0.0
    term.vel_command_b[:, 2] = wz
    term.time_left[:] = 1e9
    term.is_standing_env[:] = False


def rollout(env, w, pol, vx, wz, seconds, noise=False, skip=3.0):
    with torch.inference_mode():
        env.reset()
    obs = w.get_observations()
    rates, alive = [], torch.ones(env.num_envs, dtype=torch.bool, device="cuda:0")
    with torch.inference_mode():
        for i in range(int(seconds / env.step_dt)):
            hold(env, vx, wz)
            obs, _, done, extras = w.step(pol(obs, stochastic_output=noise))
            alive &= ~(done.bool() & ~extras["time_outs"].bool())  # falls only
            if i * env.step_dt >= skip:
                rates.append(
                    env.scene["robot"].data.root_link_ang_vel_b[:, 2].cpu().numpy()
                )
    return np.stack(rates), alive.cpu().numpy()


def record(a):
    env, w, pol = load_policy(a.run, a.ckpt, 64)
    out = {}
    for tag, wz in (("turn", 0.5), ("still", 0.0)):
        y, alive = rollout(env, w, pol, 0.5, wz, 15.0, noise=a.noise)
        out[tag] = y
        print(tag, "mean", y.mean().round(3), "alive", alive.mean())
    np.savez(a.out, dt=env.step_dt, **out)


def gain(a):
    """Swing, gait period and the reward per second gained by turning (0.5
    rad/s command, turning vs walking straight) per step and on moving
    averages, from `record` output: the prediction before training."""
    d = np.load(a.out)
    dt, turn, still = float(d["dt"]), d["turn"], d["still"]

    def score(e):
        return 1.0 / (1.0 + (e / a.std) ** 2)

    def ema(y, tau):
        out, m = np.empty_like(y), y[0].copy()
        for i in range(len(y)):
            m = m + dt / tau * (y[i] - m)
            out[i] = m
        return out[int(1.5 / dt) :]

    swing = np.std(turn - turn.mean(0), axis=0).mean()
    spec = np.abs(np.fft.rfft(still - still.mean(0), axis=0)).mean(1)
    freq = np.fft.rfftfreq(len(still), dt)
    period = 1.0 / freq[1 + np.argmax(spec[1:])]
    res = {
        "swing": round(float(swing), 3),
        "period_s": round(float(period), 3),
        "step": round(float(score(turn - 0.5).mean() - score(still - 0.5).mean()), 4),
    }
    for tau in (0.1, 0.25, 0.5, 1.0):
        g = score(ema(turn, tau) - 0.5).mean() - score(ema(still, tau) - 0.5).mean()
        res[f"window_{tau}"] = round(float(g), 4)
    print("JSON", json.dumps(res))


def evaluate(a):
    """Yaw error averaged over 1 s (about two gait cycles) at +-0.5 rad/s while
    walking at 0.5 m/s, and the success fraction, 256 worlds x 20 s."""
    env, w, pol = load_policy(a.run, a.ckpt, 256)
    res = {}
    for wz in (0.5, -0.5):
        y, alive = rollout(env, w, pol, 0.5, wz, 20.0)
        n = int(round(1.0 / env.step_dt))
        blocks = y[: len(y) // n * n].reshape(-1, n, y.shape[1]).mean(1)
        blocks = blocks[:, alive] if alive.any() else blocks  # fallen worlds reset
        res[f"turn_{wz}"] = {
            "cycle_yaw_err": round(float(np.abs(blocks - wz).mean()), 4),
            "mean_rate": round(float(y.mean()), 4),
            "alive": float(alive.mean()),
        }
    print("JSON", json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("name")
    t.add_argument("--std", type=float, required=True)
    t.add_argument("--score", choices=("step", "window"), required=True)
    t.add_argument("--tau", type=float, default=0.5)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--iters", type=int, default=3000)
    t.add_argument("--worlds", type=int, default=4096)
    t.add_argument("--kernel", choices=("cauchy", "exp"), default="cauchy")
    r = sub.add_parser("record")
    r.add_argument("run")
    r.add_argument("ckpt")
    r.add_argument("out")
    r.add_argument("noise", nargs="?", default="")
    g = sub.add_parser("gain")
    g.add_argument("out")
    g.add_argument("--std", type=float, default=0.15)
    e = sub.add_parser("eval")
    e.add_argument("run")
    e.add_argument("ckpt")
    a = ap.parse_args()
    if a.cmd == "record":
        a.noise = a.noise == "noise"
    {"train": train, "record": record, "gain": gain, "eval": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
