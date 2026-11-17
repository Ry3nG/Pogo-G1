"""Joint command capability of a policy.

  uv run --locked --extra train python scripts/capability.py RUN [--ckpt ...]
      [--heights 0.15,0.3,0.5,0.75,1.0] [--worlds 256] [--seconds 30]

Deterministic policy, static mounted start. Every height is combined with four
planar commands: in place, forward 0.3 m/s, turn 0.5 rad/s, and forward 0.3 m/s
while turning 0.5 rad/s. Each cell reports continuous-hopping success,
settled apex error, hop-cycle velocity and yaw errors, within-hop yaw swing
(per-sample minus hop-cycle yaw error), wrist time at its limit, spring stroke
used and first-failure causes. A 30 s stop / forward / stop / turn / stop
sequence measures the response to command changes. This measures the learned
policy's range, not the robot's or the stick's physical limits. Always on
flat ground, whatever terrain the policy was trained on.
Results: RUN/capability-<ckpt>.json, written after every cell.
"""

import argparse
import json
from pathlib import Path

from pogo_g1.evaluate import Evaluator

PLANAR = {
    "in_place": (0.0, 0.0),
    "forward": (0.3, 0.0),
    "turn": (0.0, 0.5),
    "forward_turn": (0.3, 0.5),
}
SEQUENCE_H = 0.3
SEQUENCE = (
    (0.0, (SEQUENCE_H, 0.0, 0.0, 0.0)),
    (5.0, (SEQUENCE_H, 0.3, 0.0, 0.0)),
    (12.0, (SEQUENCE_H, 0.0, 0.0, 0.0)),
    (18.0, (SEQUENCE_H, 0.0, 0.0, 0.5)),
    (24.0, (SEQUENCE_H, 0.0, 0.0, 0.0)),
)


def cell(h, vx, wz):
    return {
        "apex": (h, h),
        "vx": (vx, vx),
        "vy": (0.0, 0.0),
        "yaw_rate": (wz, wz),
        "robust": False,
        "terrain": "flat",  # flat ground even for a policy trained on terrain
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ckpt")
    ap.add_argument("--heights", default="0.15,0.3,0.5,0.75,1.0")
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()
    ev = Evaluator(args.run, args.ckpt)
    out = args.run / f"capability-{Path(ev.ckpt).stem}.json"
    results = json.loads(out.read_text())["cells"] if out.exists() else {}

    def save():
        out.write_text(
            json.dumps(
                {
                    "checkpoint": ev.ckpt,
                    "worlds": args.worlds,
                    "seconds": args.seconds,
                    "sequence": SEQUENCE,
                    "cells": results,
                },
                indent=1,
            )
        )

    for h in (float(x) for x in args.heights.split(",")):
        for name, (vx, wz) in PLANAR.items():
            key = f"{h:.2f}-{name}"
            if key in results:
                continue
            r = ev(cell(h, vx, wz), worlds=args.worlds, seconds=args.seconds)
            if r["yaw_err"] is not None and r["hop_yaw_err"] is not None:
                r["yaw_swing"] = r["yaw_err"] - r["hop_yaw_err"]
            results[key] = r
            print(
                key,
                {
                    k: r.get(k)
                    for k in (
                        "success",
                        "tip_apex_err",
                        "hop_vel_err",
                        "hop_yaw_err",
                        "wrist_sat",
                        "stroke_frac_max",
                        "fall_cause",
                    )
                },
                flush=True,
            )
            save()
    if "sequence" not in results:
        r = ev(
            cell(SEQUENCE_H, 0.0, 0.0),
            worlds=args.worlds,
            seconds=30.0,
            command_steps=SEQUENCE,
        )
        results["sequence"] = r
        print("sequence", r.get("success"), r.get("segments"), flush=True)
        save()


if __name__ == "__main__":
    main()
