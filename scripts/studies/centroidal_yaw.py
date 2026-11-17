"""Hop-cycle yaw error of the capability map's turn cells, from the pelvis
(as `capability.py` reports it) and from the centroidal yaw rate
(`mdp.centroidal_yaw_rate`).

usage: centroidal_yaw.py RUN CKPT [WORLDS] [SECONDS]
Results: RUN/centroidal-yaw-<ckpt>.json, written after every cell.
"""

import json
import sys
from pathlib import Path

from pogo_g1.evaluate import Evaluator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capability import PLANAR, cell  # noqa: E402


def main():
    run, ck = Path(sys.argv[1]), sys.argv[2]
    worlds = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    ev = Evaluator(run, ck)
    out = run / f"centroidal-yaw-{Path(ck).stem}.json"
    rows = json.loads(out.read_text()) if out.exists() else {}
    for h in (0.15, 0.3, 0.5, 0.75, 1.0):
        for name in ("turn", "forward_turn"):
            key = f"{h:.2f}-{name}"
            if key in rows:
                continue
            vx, wz = PLANAR[name]
            r = ev(cell(h, vx, wz), worlds=worlds, seconds=seconds)
            rows[key] = {k: r[k] for k in ("success", "hop_yaw_err", "hop_cyaw_err")}
            print(key, rows[key], flush=True)
            out.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
