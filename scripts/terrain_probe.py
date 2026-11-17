"""Terrain beyond flat ground for a flat-trained policy.

  uv run --locked --extra train python scripts/terrain_probe.py RUN [--ckpt ...]
      [--worlds 128] [--seconds 30]

Deterministic policy, no terrain in training. Rough ground (10 cm cells of
random height) in place and moving forward; raised blocks and inverted-pyramid
stairs reached by hopping forward at 0.3 m/s with 0.3 m hops. Results:
RUN/terrain-<ckpt>.json, written after every condition.
"""

import argparse
import json
from pathlib import Path

from pogo_g1.evaluate import Evaluator


def cmd(apex, vx):
    return {
        "apex": (apex, apex),
        "vx": (vx, vx),
        "vy": (0.0, 0.0),
        "yaw_rate": (0.0, 0.0),
        "robust": False,
    }


CONDITIONS = {}
for h in (0.02, 0.03, 0.05):
    CONDITIONS[f"rough_{h:g}_in_place"] = cmd(0.15, 0.0) | {
        "terrain": "rough",
        "terrain_height": h,
    }
    CONDITIONS[f"rough_{h:g}_forward"] = cmd(0.3, 0.3) | {
        "terrain": "rough",
        "terrain_height": h,
    }
for h in (0.05, 0.10, 0.15):
    CONDITIONS[f"blocks_{h:g}_forward"] = cmd(0.3, 0.3) | {
        "terrain": "steps",
        "terrain_height": h,
    }
    CONDITIONS[f"stairs_{h:g}_forward"] = cmd(0.3, 0.3) | {
        "terrain": "stairs",
        "terrain_height": h,
    }


# Generalisation of a stairs-trained policy.
GENERAL = {
    "stairs_0.075_forward": cmd(0.3, 0.3)
    | {"terrain": "stairs", "terrain_height": 0.075},
    "stairs_0.125_forward": cmd(0.3, 0.3)
    | {"terrain": "stairs", "terrain_height": 0.125},
    "stairs_down_0.1_forward": cmd(0.3, 0.3)
    | {"terrain": "stairs_down", "terrain_height": 0.1},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ckpt")
    ap.add_argument("--worlds", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--general", action="store_true", help="unseen stairs only")
    args = ap.parse_args()
    ev = Evaluator(args.run, args.ckpt)
    out = args.run / f"terrain-{Path(ev.ckpt).stem}.json"
    results = json.loads(out.read_text())["conditions"] if out.exists() else {}
    for name, overrides in (GENERAL if args.general else CONDITIONS).items():
        if name in results:
            continue
        r = ev(dict(overrides), worlds=args.worlds, seconds=args.seconds)
        results[name] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(
            name,
            {k: r.get(k) for k in ("success", "survival", "hop_vel_err", "fall_cause")},
            flush=True,
        )
        out.write_text(
            json.dumps(
                {
                    "checkpoint": ev.ckpt,
                    "worlds": args.worlds,
                    "seconds": args.seconds,
                    "conditions": results,
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
