"""Pneumatic-v5 evaluation suite (DESIGN.md section 4).

  uv run --locked --extra train python scripts/eval_suite.py RUN_DIR [--ckpt ...] [--only a,b]

Writes a checkpoint/world-count/horizon-specific report. Default: 256 worlds, 30 s, static
mounted start, nominal physics unless stated; training-time DR, pushes and
latency are OFF unless the condition sets them.
"""

import argparse
import json
from pathlib import Path

from pogo_g1.evaluate import Evaluator
from pogo_g1.pogo import PHYSICS_VERSION

H = 0.12  # default apex command for robustness conditions
IN_PLACE = {
    "apex": (H, H),
    "vx": (0.0, 0.0),
    "vy": (0.0, 0.0),
    "yaw_rate": (0.0, 0.0),
    "robust": False,
}


def cmd(apex=H, vx=0.0, vy=0.0, wz=0.0):
    return {
        "apex": (apex, apex),
        "vx": (vx, vx),
        "vy": (vy, vy),
        "yaw_rate": (wz, wz),
        "robust": False,
    }


CONDITIONS = {
    # Command tracking
    "apex_0.05": cmd(0.05),
    "apex_0.10": cmd(0.10),
    "apex_0.15": cmd(0.15),
    "apex_0.20": cmd(0.20),
    "apex_0.25": cmd(0.25),
    "apex_0.30": cmd(0.30),
    "apex_0.35": cmd(0.35),
    "apex_0.40": cmd(0.40),
    "apex_0.50": cmd(0.50),
    "apex_0.60": cmd(0.60),
    "apex_0.75": cmd(0.75),
    "apex_1.00": cmd(1.00),
    "fwd_0.4": cmd(vx=0.4),
    "back_0.3": cmd(vx=-0.3),
    "left_0.3": cmd(vy=0.3),
    "right_0.3": cmd(vy=-0.3),
    "turn_0.5": cmd(wz=0.5),
    # Integration
    "dt_1ms": IN_PLACE | {"timestep": 0.001, "decimation": 20},
    # Disturbances
    "push_0.3": IN_PLACE | {"eval_push": 0.3},
    "push_0.6": IN_PLACE | {"eval_push": 0.6},
    "push_0.9": IN_PLACE | {"eval_push": 0.9},
    "latency_10ms": IN_PLACE | {"latency_s": 0.01},
    "latency_20ms": IN_PLACE | {"latency_s": 0.02},
    "latency_40ms": IN_PLACE | {"latency_s": 0.04},
    # Physical uncertainty, including conditions outside the training range.
    "pressure_80kpa": IN_PLACE | {"params.pressure": 80000.0},
    "pressure_180kpa": IN_PLACE | {"params.pressure": 180000.0},
    "isothermal": IN_PLACE | {"params.exponent": 1.0},
    "adiabatic": IN_PLACE | {"params.exponent": 1.4},
    "pogo_3kg": IN_PLACE | {"params.frame_mass": 2.1},
    "pogo_9kg": IN_PLACE | {"params.frame_mass": 8.1},
    "ground_friction_0.3": IN_PLACE | {"contacts.ground_friction": 0.3},
    "seal_friction_5n": IN_PLACE | {"params.seal_friction": 5.0},
    "seal_friction_25n": IN_PLACE | {"params.seal_friction": 25.0},
    "damping_5": IN_PLACE | {"params.damping": 5.0},
    "damping_25": IN_PLACE | {"params.damping": 25.0},
    "waves_3cm": IN_PLACE | {"terrain": "waves", "terrain_height": 0.03},
    "slope_0.08": IN_PLACE | {"terrain": "slope", "terrain_slope": 0.08},
}


CONDITIONS.update(
    {
        "nominal_0.12": cmd(),
        "fwd_0.2": cmd(vx=0.2),
        "back_0.15": cmd(vx=-0.15),
        "turn_-0.5": cmd(wz=-0.5),
        "command_sequence": cmd(),
    }
)
SEQUENCE_STEPS = (
    (0.0, (H, 0.0, 0.0, 0.0)),
    (5.0, (H, 0.4, 0.0, 0.0)),
    (15.0, (H, 0.0, 0.0, 0.0)),
    (20.0, (H, 0.0, 0.0, 0.5)),
    (25.0, (H, 0.0, 0.0, 0.0)),
)


def fmt(v):
    if v is None:
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ckpt")
    ap.add_argument("--only", default="")
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()
    unknown = set(args.only.split(",")) - CONDITIONS.keys() - {""}
    if unknown:
        ap.error(f"Unknown evaluation conditions: {sorted(unknown)}")
    names = (
        [n for n in CONDITIONS if n in args.only.split(",")]
        if args.only
        else list(CONDITIONS)
    )
    if "command_sequence" in names and args.seconds != 30.0:
        ap.error("command_sequence uses the fixed 30 s command protocol")
    ev = Evaluator(args.run, args.ckpt)
    out = args.run / f"suite-{Path(ev.ckpt).stem}-{args.worlds}x{args.seconds:g}s.json"
    results = json.loads(out.read_text())["conditions"] if out.exists() else {}
    report = {
        "physics_version": PHYSICS_VERSION,
        "checkpoint": ev.ckpt,
        "seed": json.loads((args.run / "config.json").read_text())["task"]["seed"],
        "continuous_hopping": {"first_hop_deadline_s": 3.0, "max_gap_s": 2.0},
        "worlds": args.worlds,
        "seconds": args.seconds,
        "conditions": results,
        "overrides": CONDITIONS,
        "command_steps": {"command_sequence": SEQUENCE_STEPS},
    }
    for n in names:
        options = {"command_steps": SEQUENCE_STEPS} if n == "command_sequence" else {}
        results[n] = ev(
            dict(CONDITIONS[n]), worlds=args.worlds, seconds=args.seconds, **options
        )
        results[n]["status"] = "evaluated"
        print(n, {k: fmt(v) for k, v in results[n].items()}, flush=True)
        out.write_text(json.dumps(report, indent=1))
    cols = [
        "status",
        "success",
        "survival",
        "hops",
        "tip_apex_err",
        "vel_err",
        "yaw_err",
        "hop_vel_err",
        "hop_yaw_err",
        "grip_f_p99",
        "grip_t_p99",
        "power_w",
    ]
    lines = ["| condition | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for n, r in results.items():
        lines.append(f"| {n} | " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    if "command_sequence" in results:
        lines.extend(
            [
                "",
                "Command sequence: all transition samples included; errors use "
                "pre-failure samples and survival uses all worlds. One policy-step "
                "command observation delay. No additional pass/fail gate.",
                "",
                "| time (s) | command [h, vx, vy, yaw] | vel_err (m/s) | yaw_err (rad/s) | survival_end | alive_samples |",
                "|---|---|---|---|---|---|",
            ]
        )
        for r in results["command_sequence"]["segments"]:
            lines.append(
                f"| {r['start_s']:g}-{r['end_s']:g} | {r['command']} | "
                f"{fmt(r['vel_err'])} | {fmt(r['yaw_err'])} | "
                f"{fmt(r['survival_end'])} | {r['alive_samples']} |"
            )
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
