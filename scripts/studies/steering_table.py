"""One table of every steering condition, for the report's
gain-versus-learning figure: training-time turning gain (reward per second,
sampled actions), within-hop swing of the raw yaw rate and of the quantity
the reward scores, the update at which the turn is first held, and the runs.

usage: steering_table.py [OUT.csv]   (default artifacts/steering-conditions.csv)

Gains and swings are quoted from the experiment named in each row. Computed
here, from existing files only:
- first held: pogo, the first watcher checkpoint (64 x 10 s, every 500) from
  which the mean hop-cycle yaw error at +-0.5 rad/s stays <= 0.15 to the
  run's end (artifacts/curves; reproduces every reported value); walking, the
  same with both 1 s cycle errors <= 0.15 (artifacts/walk/eval.txt);
- the sampled swing of the moving averages (std over time within a world,
  world mean) from the recordings in artifacts/steer-signal (hopn: per hop
  seed 0, win05n: window 0.5 s seed 0, both at 4000 with sampled actions),
  after a start-up of 3 tau (at least 1.5 s);
- walking: the window's sampled swing and the mjlab control's gain under its
  own exp kernel (0.71) from the control's sampled recording
  (artifacts/walk/record/ctrl-noise.npz, model_3000).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

CURVES = Path("artifacts/curves")
STEER = Path("artifacts/steer-signal")
WALK = Path("artifacts/walk")
TOL = 0.15


def ema(y, dt, tau):
    out, m = np.empty_like(y), y[0].copy()
    for i in range(len(y)):
        m = m + dt / tau * (y[i] - m)
        out[i] = m
    return out[int(max(1.5, 3 * tau) / dt) :]


def swing(y):
    return float(np.std(y, axis=0).mean())


def pogo_held(run, last):
    rows = sorted(
        (json.loads(x) for x in (CURVES / f"{run}.jsonl").read_text().splitlines()),
        key=lambda r: r["iter"],
    )
    rows = [r for r in rows if r["iter"] <= last]

    def ok(r):
        v = [r[c]["hop_yaw_err"] for c in ("turn_0.5", "turn_-0.5")]
        return None not in v and sum(v) / 2 <= TOL

    for i, r in enumerate(rows):
        if all(ok(x) for x in rows[i:]):
            return str(r["iter"])
    return f"none by {last}"


def walk_held(name):
    rows = {}
    for line in (WALK / "eval.txt").read_text().splitlines():
        parts = line.split(" ", 3)
        if parts[0].endswith(f"/{name}") and len(parts) == 4 and parts[2] == "JSON":
            r = json.loads(parts[3])
            rows[int(parts[1])] = all(r[c]["cycle_yaw_err"] <= TOL for c in r)
    its = sorted(rows)
    for i, it in enumerate(its):
        if all(rows[j] for j in its[i:]):
            return str(it)
    return f"none by {its[-1]}"


def pogo_window_swings(tau):
    vals = []
    for f in ("hopn-turn.npz", "win05n-turn.npz"):
        d = np.load(STEER / f)
        vals.append(swing(ema(d["yaw"], float(d["dt"]), tau)))
    return f"{min(vals):.2f}-{max(vals):.2f}"


def main():
    out = Path(
        sys.argv[1] if len(sys.argv) > 1 else "artifacts/steering-conditions.csv"
    )
    w = np.load(WALK / "record/ctrl-noise.npz")
    wdt, turn, still = float(w["dt"]), w["turn"], w["still"]
    ctrl_gain = (
        np.exp(-(((turn - 0.5) / 0.71) ** 2)).mean()
        - np.exp(-(((still - 0.5) / 0.71) ** 2)).mean()
    )
    walk_window_swing = swing(ema(turn, wdt, 0.5))

    def runs(*names):
        return ";".join(names)

    hop = runs("1012-0002-vel-obs-s0", "1011-1913-vel-obs-s1", "1015-0300-vel-obs-s2")
    rows = [
        # task, condition, scored signal, runs, last update, gain, gain source,
        # raw swing, scored swing, swing source
        (
            "pogo",
            "per hop",
            "pelvis yaw, hop-cycle average at landing",
            hop,
            4000,
            "0.38-0.64",
            "experiment of 2026-10-22 (per-hop and window s0 policies)",
            "1.29-1.44",
            "0.12-0.22",
            "experiment of 2026-10-22 (hop-average spread)",
        ),
        (
            "pogo",
            "per step",
            "pelvis yaw, every 20 ms",
            runs(
                "1015-1133-step-steer-s0",
                "1015-1455-step-steer-s1",
                "1015-1819-step-steer-s2",
            ),
            4000,
            "0.003-0.012",
            "experiment of 2026-10-22",
            "1.29-1.44",
            "1.29-1.44",
            "experiment of 2026-10-22",
        ),
        (
            "pogo",
            "per step, 12000 updates",
            "pelvis yaw, every 20 ms",
            runs("1016-1140-step-steer-long-s0", "1016-2155-step-steer-long-s1"),
            12000,
            "0.003-0.012",
            "experiment of 2026-10-22",
            "1.29-1.44",
            "1.29-1.44",
            "experiment of 2026-10-22",
        ),
        (
            "pogo",
            "window 0.1 s",
            "pelvis yaw, moving average tau 0.1 s, every step",
            runs(
                "1020-1719-window-tau0.1-s0",
                "1021-0306-window-tau0.1-s1",
                "1020-1419-window-tau0.1-s2",
            ),
            4000,
            "0.11-0.27",
            "experiment of 2026-10-22",
            "1.29-1.44",
            pogo_window_swings(0.1),
            "computed (hopn, win05n)",
        ),
        (
            "pogo",
            "window 0.5 s",
            "pelvis yaw, moving average tau 0.5 s, every step",
            runs(
                "1019-1115-window-steer-s0",
                "1019-1443-window-steer-s1",
                "1019-1805-window-steer-s2",
            ),
            4000,
            "0.46-0.70",
            "experiment of 2026-10-22",
            "1.29-1.44",
            pogo_window_swings(0.5),
            "computed (hopn, win05n); experiment of 2026-10-22: 0.17",
        ),
        (
            "pogo",
            "window 2 s",
            "pelvis yaw, moving average tau 2 s, every step",
            runs(
                "1020-2033-window-tau2-s0",
                "1021-0623-window-tau2-s1",
                "1020-2247-window-tau2-s2",
            ),
            4000,
            "0.50-0.69",
            "experiment of 2026-10-22",
            "1.29-1.44",
            pogo_window_swings(2.0),
            "computed (hopn, win05n)",
        ),
        (
            "pogo",
            "gain-matched per step (yaw weight x3.4)",
            "pelvis yaw, every 20 ms",
            runs(
                "1020-1354-step-gain-s0",
                "1020-2352-step-gain-s1",
                "1020-0553-step-gain-s2",
            ),
            4000,
            "0.04-0.15",
            "experiment of 2026-10-22 (own sampled gait, s0, 1000-4000 updates)",
            "0.41-0.67",
            "0.41-0.67",
            "experiment of 2026-10-22 (own sampled gait, s0)",
        ),
        (
            "pogo",
            "capped noise (max std 0.25)",
            "pelvis yaw, every 20 ms",
            runs(
                "1022-1251-step-maxstd-s0",
                "1022-1601-step-maxstd-s1",
                "1022-0850-step-maxstd-s2",
            ),
            4000,
            "0.008",
            "experiment of 2026-10-22 (own sampled gait, s0-s1)",
            "1.13-1.17",
            "1.13-1.17",
            "experiment of 2026-10-22 (own sampled gait, s0-s1)",
        ),
        (
            "pogo",
            "centroidal per step",
            "L_z / I_zz of robot + stick, every 20 ms",
            runs(
                "1025-1621-centroidal-s0",
                "1025-2023-centroidal-s1",
                "1025-2341-centroidal-s2",
            ),
            4000,
            "0.38-0.47",
            "experiment of 2026-10-25 (per-hop and window s0 policies)",
            "1.29-1.44",
            "0.19-0.21",
            "experiment of 2026-10-25",
        ),
        (
            "walking",
            "per step, width 0.15",
            "pelvis yaw, every 20 ms",
            runs("walk/step0.15-s0", "walk/step0.15-s1", "walk/step0.15-s2"),
            3000,
            "0.10",
            "experiment of 2026-10-26 (control's sampled gait, model_3000)",
            "0.49",
            "0.49",
            "experiment of 2026-10-26",
        ),
        (
            "walking",
            "window 0.5 s, width 0.15",
            "pelvis yaw, moving average tau 0.5 s",
            runs("walk/window0.15-s0", "walk/window0.15-s1", "walk/window0.15-s2"),
            3000,
            "0.77",
            "experiment of 2026-10-26 (control's sampled gait, model_3000)",
            "0.49",
            f"{walk_window_swing:.2f}",
            "computed (ctrl-noise)",
        ),
        (
            "walking",
            "mjlab control, exp width 0.71",
            "pelvis yaw, every 20 ms",
            runs("walk/step0.71-s0"),
            3000,
            f"{ctrl_gain:.2f}",
            "computed (ctrl-noise, own exp kernel)",
            "0.49",
            "0.49",
            "experiment of 2026-10-26",
        ),
    ]
    header = (
        "task",
        "condition",
        "scored_signal",
        "runs",
        "first_held_updates",
        "turning_gain_per_s",
        "gain_source",
        "raw_yaw_swing_rad_s",
        "scored_swing_rad_s",
        "swing_source",
        "seeds",
    )
    with out.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for task, cond, sig, rs, last, g, gs, raw, sc, ss in rows:
            names = rs.split(";")
            if task == "pogo":
                held = [pogo_held(r, last) for r in names]
            else:
                held = [walk_held(r.split("/")[1]) for r in names]
            wr.writerow(
                (task, cond, sig, rs, ";".join(held), g, gs, raw, sc, ss, len(names))
            )
            print(cond, held, g, sc)
    print("wrote", out)


if __name__ == "__main__":
    main()
