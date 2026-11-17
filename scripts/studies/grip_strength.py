"""Grip safety margin: how weak can the hands get before the policy loses the
handle? Finger servo torque limits are scaled (TaskCfg.grip_strength) at each
commanded height, in place; success and first failure cause per cell.

usage: grip_strength.py RUN CKPT OUT.json [WORLDS] [SECONDS] [--heights ...]
           [--search] [--command VX,YAW]

Default: the fixed strength grid below. --search finds, per height, the lowest
strength whose success is >= PASS with every tested strength above it passing:
strengths fall from 1.0 in 0.1 steps until one fails, then the midpoint 0.05
above the failure is tried. The safety factor is 1 / that strength. Cells
already in OUT.json are reused. --command holds a forward velocity (m/s) and
yaw rate (rad/s) instead of hopping in place (default 0,0); give it its own
OUT.json.
"""

import argparse
import json
from pathlib import Path

from pogo_g1.evaluate import Evaluator

STRENGTHS = (1.0, 0.7, 0.6, 0.5, 0.4)
HEIGHTS = "0.15,0.5,1.0"
PASS = 0.95


def edge(rows, h):
    """Lowest passing strength at height h, contiguous from 1.0 (None if 1.0 fails)."""
    cells = sorted(
        ((r["strength"], r["success"]) for r in rows if r["apex"] == h), reverse=True
    )
    lowest = None
    for s, ok in cells:
        if ok < PASS:
            break
        lowest = s
    return lowest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("ckpt")
    ap.add_argument("out", type=Path)
    ap.add_argument("worlds", type=int, nargs="?", default=128)
    ap.add_argument("seconds", type=float, nargs="?", default=20.0)
    ap.add_argument("--heights", default=HEIGHTS)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--command", default="0,0")
    args = ap.parse_args()
    vx, yaw = (float(x) for x in args.command.split(","))
    ev = Evaluator(args.run, args.ckpt)
    rows = json.loads(args.out.read_text()) if args.out.exists() else []
    found = {(r["strength"], r["apex"]): r["success"] for r in rows}

    def success(s, h):
        if (s, h) in found:
            return found[(s, h)]
        res = ev(
            {
                "apex": (h, h),
                "vx": (vx, vx),
                "vy": (0.0, 0.0),
                "yaw_rate": (yaw, yaw),
                "robust": False,
                "grip_strength": s,
            },
            worlds=args.worlds,
            seconds=args.seconds,
        )
        keep = ("success", "survival", "tip_apex_err", "hop_vel_err", "fall_cause")
        row = {"strength": s, "apex": h} | {k: res[k] for k in keep if k in res}
        if vx or yaw:
            row |= {"vx": vx, "yaw_rate": yaw, "hop_yaw_err": res["hop_yaw_err"]}
        print(json.dumps(row), flush=True)
        rows.append(row)
        found[(s, h)] = row["success"]
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
        return row["success"]

    for h in (float(x) for x in args.heights.split(",")):
        if not args.search:
            for s in STRENGTHS:
                success(s, h)
            continue
        s = 1.0
        while s > 0.05 and success(s, h) >= PASS:
            s = round(s - 0.1, 2)
        if 0.05 < s < 1.0:
            success(round(s + 0.05, 2), h)
        low = edge(rows, h)
        print(
            f"edge apex {h}: lowest {low}, factor {1 / low if low else None}",
            flush=True,
        )


if __name__ == "__main__":
    main()
