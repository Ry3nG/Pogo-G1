"""Data for the report figures, written as JSON for the HTML/Observable Plot
templates in figures/src (figures/src/build.py renders them to PDF and PNG).

  uv run --locked --extra train python scripts/fig_data.py figures/data [signal|learning ...]

signal   `studies/yaw_signals.py` recordings of the per-hop policy (seed 0,
         4000 updates) at a 0.5 rad/s command with sampled actions: pelvis and
         momentum-based (L_z/I_zz) yaw rate, stance phases, hop-cycle means;
         and the expected per-step kernel reward against the mean error for
         the recorded spreads (Gaussian fluctuation, as in Section 4).
learning `watch_curves.py` curves (64 worlds x 10 s every 500 updates) for per
         hop, per step, window 0.5 s and the momentum-based yaw rate, and the
         steering table (`studies/steering_table.py`) for the gain figure.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

YAWQ = Path("artifacts/steer-signal/yawq")
CURVES = Path("artifacts/curves")
TABLE = Path("artifacts/steering-conditions.csv")
CMD, TOL = 0.5, 0.15


def kernel(e, s=TOL):
    return 1.0 / (1.0 + (e / s) ** 2)


def signal():
    d = np.load(YAWQ / "hop0-noise-turn.npz")
    dt, w = float(d["dt"]), 0
    t0, t1 = 4.0, 7.0  # three seconds after start-up
    i0, i1 = int(t0 / dt), int(t1 / dt)
    t = np.arange(i0, i1) * dt - t0
    contact = d["contact"][i0:i1, w].astype(bool)
    stance, start = [], None
    for k, c in enumerate(contact):
        if c and start is None:
            start = k
        if not c and start is not None:
            stance.append([float(t[start]), float(t[k])])
            start = None
    if start is not None:
        stance.append([float(t[start]), float(t[-1])])
    # hop-cycle means between consecutive touchdowns (whole recording, then cut)
    c = d["contact"][:, w].astype(bool)
    td = [k for k in range(1, len(c)) if c[k] and not c[k - 1]]
    pelvis = d["pelvis"][:, w]
    hops = [
        [a * dt - t0, b * dt - t0, float(pelvis[a:b].mean())]
        for a, b in zip(td, td[1:], strict=False)
        if a * dt >= t0 - 0.5 and b * dt <= t1 + 0.5
    ]
    det = np.load(YAWQ / "hop0-det-turn.npz")
    spread = {
        "pelvis_sampled": float(d["pelvis"][150:].std(0).mean()),
        "pelvis_deterministic": float(det["pelvis"][150:].std(0).mean()),
        "momentum_sampled": float(d["centroidal"][150:].std(0).mean()),
    }
    # expected per-step kernel reward E[k(m + xi)], xi ~ N(0, sigma^2)
    xi = np.linspace(-6, 6, 2401)
    pdf = np.exp(-0.5 * xi**2) / np.sqrt(2 * np.pi)
    m = np.linspace(-1.0, 1.0, 201)
    curves = {}
    for name, sigma in (
        ("pelvis_sampled", spread["pelvis_sampled"]),
        ("momentum_sampled", spread["momentum_sampled"]),
    ):
        ek = [float(np.trapezoid(kernel(mm + sigma * xi) * pdf, xi)) for mm in m]
        curves[name] = {"sigma": sigma, "m": m.round(4).tolist(), "reward": ek}
    still = np.load(YAWQ / "hop0-noise-still.npz")
    inside = lambda x: float((np.abs(x[150:] - CMD) <= TOL).mean())  # noqa: E731
    hits = {
        sig: {"turning": inside(d[key]), "in_place": inside(still[key])}
        for sig, key in (("pelvis", "pelvis"), ("momentum", "centroidal"))
    }
    # per-step yaw-rate histograms and the measured kernel reward (Fig 4b)
    edges = np.round(np.arange(-3.0, 4.0 + 0.025, 0.05), 4)
    hist = {"bin": 0.05, "lo": float(edges[0]), "hi": float(edges[-1])}
    for sig, key in (("pelvis", "pelvis"), ("momentum", "centroidal")):
        hist[sig] = {}
        for case, rec in (("turning", d), ("in_place", still)):
            x = rec[key][150:].ravel().astype(float)
            hist[sig][case] = {
                "n": int(x.size),
                "counts": np.histogram(x, edges)[0].tolist(),
                "mean": round(float(x.mean()), 4),
                "inside": float((np.abs(x - CMD) <= TOL).mean()),
                "reward": float(kernel(x - CMD).mean()),
            }
    return {
        "dt": dt,
        "command": CMD,
        "hits": hits,
        "hist": hist,
        "tolerance": TOL,
        "t": t.round(3).tolist(),
        "pelvis": pelvis[i0:i1].round(4).tolist(),
        "momentum": d["centroidal"][i0:i1, w].round(4).tolist(),
        "stance": stance,
        "hops": [h for h in hops if h[1] > 0 and h[0] < t1 - t0],
        "spread": spread,
        "expected": curves,
    }


RUNS = {
    "per hop": ["1012-0002-vel-obs-s0", "1011-1913-vel-obs-s1", "1015-0300-vel-obs-s2"],
    "per step": [
        "1015-1133-step-steer-s0",
        "1015-1455-step-steer-s1",
        "1015-1819-step-steer-s2",
    ],
    "window 0.5 s": [
        "1019-1115-window-steer-s0",
        "1019-1443-window-steer-s1",
        "1019-1805-window-steer-s2",
    ],
    "momentum": [
        "1025-1621-centroidal-s0",
        "1025-2023-centroidal-s1",
        "1025-2341-centroidal-s2",
    ],
}


def yaw(row):
    v = [row.get(k, {}).get("hop_yaw_err") for k in ("turn_0.5", "turn_-0.5")]
    return None if None in v else float(np.mean(v))


def learning():
    curves = []
    for cond, runs in RUNS.items():
        for seed, run in enumerate(runs):
            for line in (CURVES / f"{run}.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r["iter"] <= 4000 and (y := yaw(r)) is not None:
                    curves.append(
                        {"condition": cond, "seed": seed, "iter": r["iter"], "yaw": y}
                    )
    conditions = []
    with TABLE.open() as f:
        for r in csv.DictReader(f):
            lo, _, hi = r["turning_gain_per_s"].partition("-")
            held = []
            for v in r["first_held_updates"].split(";"):
                held.append(
                    {"held": None, "censored": int(v.split()[-1])}
                    if v.startswith("none")
                    else {"held": int(v), "censored": None}
                )
            conditions.append(
                {
                    "task": r["task"],
                    "condition": r["condition"],
                    "gain_lo": float(lo),
                    "gain_hi": float(hi or lo),
                    "seeds": held,
                }
            )
    return {"tolerance": TOL, "curves": curves, "conditions": conditions}


CAPABILITY = ["1012-0002-vel-obs-s0", "1011-1913-vel-obs-s1", "1015-0300-vel-obs-s2"]


def capability():
    """Capability map of the per-hop recipe at 4000 updates, three seeds:
    worst success, worst apex error, mean planar and worst yaw error per cell."""
    seeds = [
        json.loads(
            (Path("artifacts/runs") / r / "capability-model_4000.json").read_text()
        )["cells"]
        for r in CAPABILITY
    ]
    cells = []
    for key in seeds[0]:
        if key == "sequence":
            continue
        height, command = key.split("-", 1)
        vals = [s[key] for s in seeds]
        cells.append(
            {
                "height": float(height),
                "command": command,
                "success": min(v["success"] for v in vals),
                "apex_err": max(v["tip_apex_err"] for v in vals),
                "planar_err": float(np.mean([v["hop_vel_err"] for v in vals])),
                "yaw_err": max(v["hop_yaw_err"] for v in vals),
            }
        )
    return {"runs": CAPABILITY, "cells": cells}


RUNS_DIR = Path("artifacts/runs")
GRIP_OLD = {  # fixed-grid scores of the baselines, before the per-height search
    "1012-0002-vel-obs-s0": "artifacts/grip-strength/hop-s0-4000.json",
    "1011-1913-vel-obs-s1": "artifacts/grip-strength/1011-1913-vel-obs-s1-4000.json",
    "1015-0300-vel-obs-s2": "artifacts/grip-strength/1015-0300-vel-obs-s2-4000.json",
}
TRAINED = {  # training hand strength -> resumed runs (2000 updates from 4000)
    1.0: "resume-full",
    0.7: "grip-0.7",
    0.5: "grip-half",
    0.35: "grip-0.35",
}


def margins(path):
    """Safety factor per height: 1 / the lowest hand strength whose success is
    >= 0.95 with every stronger tested hand passing (as grip_strength.py)."""
    rows = json.loads(Path(path).read_text())
    out = {}
    for h in sorted({r["apex"] for r in rows}):
        lowest = None
        for r in sorted(
            (r for r in rows if r["apex"] == h), key=lambda r: -r["strength"]
        ):
            if r["success"] < 0.95:
                break
            lowest = r["strength"]
        out[str(h)] = None if lowest is None else round(1 / lowest, 3)
    return out


def grip():
    """Grip margin of the trained policies by hop height, and the margin and
    full-strength cost bought by resuming at a weaker hand (three seeds each).
    Missing runs (still training) are simply absent."""
    trained = []
    for seed, run in enumerate(CAPABILITY):
        new = RUNS_DIR / run / "grip-strength-model_4000.json"
        src = new if new.exists() else Path(GRIP_OLD[run])
        trained.append(
            {"seed": seed, "updates": 4000, "source": str(src), "factor": margins(src)}
        )
        long = RUNS_DIR / run / "grip-strength-model_19999.json"
        if long.exists():
            trained.append(
                {
                    "seed": seed,
                    "updates": 19999,
                    "source": str(long),
                    "factor": margins(long),
                }
            )
    sweep = []
    for strength, name in TRAINED.items():
        for run in sorted(RUNS_DIR.glob(f"*-{name}-s*")):
            g, c = (
                run / "grip-strength-model_6000.json",
                run / "capability-model_6000.json",
            )
            if not (g.exists() and c.exists()):
                continue
            cells = {
                k: v
                for k, v in json.loads(c.read_text())["cells"].items()
                if k != "sequence"
            }
            fwd = [v["hop_vel_err"] for k, v in cells.items() if "forward" in k]
            turn = [v["hop_yaw_err"] for k, v in cells.items() if "turn" in k]
            factor = margins(g)
            sweep.append(
                {
                    "strength": strength,
                    "run": run.name,
                    "seed": int(run.name[-1]),
                    "factor": factor,
                    "worst_factor": min(
                        (f for f in factor.values() if f), default=None
                    ),
                    "planar_fwd": float(np.mean(fwd)),
                    "yaw_turn": float(np.mean(turn)),
                    "min_success": min(v["success"] for v in cells.values()),
                }
            )
    return {"trained": trained, "sweep": sweep}


NUS_MODE = "hold"  # heading held: the cleaner of the two modes


def nus():
    """The 18 m 'NUS' path and the median world's hopped path for each seed
    (`demo_nus.py ... hold` with NUS_BIAS=1 and NUS_EXPORT, 4000 updates;
    replays in artifacts/render/nus-hold-s*, `follow` in nus-full-s*)."""
    out = {"seeds": [], "mode": NUS_MODE}
    for seed in range(3):
        d = np.load(
            f"artifacts/render/nus-{'hold' if NUS_MODE == 'hold' else 'full'}-s{seed}/replay.npz"
        )
        o = d["path"][0]
        if "path" not in out:
            ink = d["ink"].astype(bool)
            strokes, cur = [], []
            for p, on in zip(d["path"] - o, ink, strict=True):
                if on:
                    cur.append(p[:2].round(3).tolist())
                elif cur:
                    strokes.append(cur)
                    cur = []
            out["path"] = strokes + ([cur] if cur else [])
        # only the pen-down stretches: the moves between letters are not scored
        trail, pen = d["trail"][:, :2] - o[:2], d["pen"].astype(bool)
        segs, cur = [], []
        for k, (p, on) in enumerate(zip(trail, pen, strict=True)):
            if on and k % 3 == 0:
                cur.append(p.round(3).tolist())
            elif not on and cur:
                segs.append(cur)
                cur = []
        out["seeds"].append(segs + ([cur] if cur else []))
    return out


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[2:] or ["signal", "learning", "capability", "grip", "nus"]:
        (out / f"{name}.json").write_text(json.dumps(globals()[name]()))
        print("wrote", out / f"{name}.json")


if __name__ == "__main__":
    main()
