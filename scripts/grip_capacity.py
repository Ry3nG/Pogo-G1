"""Pull-out capacity of one Revo2 hand closed around the pogo handle.

Uses the grasp fixture (robot.grasp_fixture): hand fixed in space, zero gravity, the
handle a free 0.2 kg bar of the real radius. After the settled grasp, a force (or
moment) is ramped on the bar until it slips (> 1 cm, > 0.15 rad tilt, or > 0.3 rad spin for the axial moment).
Directions are in the grasp frame: z along the bar, x/y across it.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from pogo_g1 import pogo, robot

TESTS = {
    "+x": ("force", 0),
    "-x": ("force", 0, -1),
    "+y": ("force", 1),
    "-y": ("force", 1, -1),
    "+z": ("force", 2),
    "-z": ("force", 2, -1),
    "moment_z": ("moment", 2),
    "-moment_z": ("moment", 2, -1),
    "moment_x": ("moment", 0),
    "-moment_x": ("moment", 0, -1),
    "moment_y": ("moment", 1),
    "-moment_y": ("moment", 1, -1),
}


def capacity(side: str, radius: float) -> dict:
    m = robot.grasp_fixture(side, radius).compile()
    d = mujoco.MjData(m)
    robot.close_on_bar(m, d, side)
    settled = mujoco.MjData(m)
    mujoco.mj_copyData(settled, m, d)
    b = m.body("bar").id
    frame = np.empty(9)
    mujoco.mju_quat2Mat(frame, np.asarray(robot.settled_grasp(side)["quat"]))
    frame = frame.reshape(3, 3)
    u = np.asarray(robot._flexion_axis(side))  # bar axis in the bar body frame
    n = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
    n /= np.linalg.norm(n)
    out = {}
    for name, (kind, k, *sign) in TESTS.items():
        mujoco.mj_copyData(d, m, settled)
        p0, r0 = d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()
        vec = frame[:, k] * (sign[0] if sign else 1)
        step = 0.004 if kind == "moment" else 0.2
        out[name] = None
        for i in range(1, 2001):
            d.xfrc_applied[b] = 0
            j = 3 if kind == "moment" else 0
            d.xfrc_applied[b, j : j + 3] = vec * i * step
            mujoco.mj_step(m, d)
            r = d.xmat[b].reshape(3, 3)
            slip = np.linalg.norm(d.xpos[b] - p0) > 0.01
            slip |= np.arccos(np.clip((r @ u) @ (r0 @ u), -1, 1)) > 0.15
            if name.endswith("moment_z"):
                slip |= np.arccos(np.clip((r @ n) @ (r0 @ n), -1, 1)) > 0.3
            if slip or not np.isfinite(d.qpos).all():
                out[name] = round(i * step, 2)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/grip_capacity.json"))
    args = ap.parse_args()
    radius = pogo.handle_radius()
    res = {
        "physics_version": pogo.PHYSICS_VERSION,
        "handle_radius_m": radius,
        "units": "N (force), N m (moment); null: held",
    }
    for side in robot.SIDES:
        res[side] = capacity(side, radius)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
