"""Explicit material pairs: coefficients belong to an interaction, not one geom.

MuJoCo otherwise takes the maximum friction of equally prioritised geoms. That
would turn a slippery floor back into a grippy floor. Native elliptic cones model
isotropic slip; MuJoCo's pyramid is up to 29% weaker along its diagonals.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ContactParams:
    ground_friction: float = 0.8
    pedal_friction: float = 0.8
    grip_friction: float = 1.0
    tip_patch_radius: float = 0.008  # effective loaded contact patch [m]
    grip_torsion: float = 0.005  # moment/normal-force length at mu=1 (baseline)
    rolling_resistance: float = 0.0003  # moment / normal force [m]
    tip_timeconst: float = 0.004  # 2 x timestep, the stiffest MuJoCo allows

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("contact parameters must be finite and nonnegative")
        if self.tip_timeconst <= 0:
            raise ValueError("tip contact time constant must be positive")


DEFAULT = ContactParams()


def configure_solver(spec):
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 3.0


def tip_pair(spec, tip, ground, params=DEFAULT):
    mu = params.ground_friction
    # Uniform pressure disk: torsional capacity = 2/3 * mu * radius * normal load.
    # Spin/rolling coefficients have dimensions of length, unlike sliding mu.
    spec.add_pair(
        name=f"contact/tip/{ground}",
        geomname1=tip,
        geomname2=ground,
        condim=6,
        friction=[
            mu,
            mu,
            2 / 3 * mu * params.tip_patch_radius,
            params.rolling_resistance,
            params.rolling_resistance,
        ],
        solref=[params.tip_timeconst, 1.0],
        solreffriction=[0.004, 1.0],
        solimp=[0.9, 0.95, 0.002, 0.5, 2.0],
    )


def hand_pairs(spec, handle, hand_prefix="robot/", params=DEFAULT):
    mu = params.grip_friction
    for geom in spec.geoms:
        if geom.name.startswith(hand_prefix) and geom.name.endswith("_contact"):
            spec.add_pair(
                name=f"contact/grip/{geom.name}",
                geomname1=geom.name,
                geomname2=handle,
                condim=4,
                friction=[mu, mu, mu * params.grip_torsion, 0, 0],
                solref=[0.004, 1.0],
            )


def assembly_pairs(spec, params=DEFAULT, ground_names=("floor",)):
    configure_solver(spec)
    for ground in ground_names:
        tip_pair(spec, "pogo/tip", ground, params)
    for side in ("left", "right"):
        for i in range(1, 8):
            foot = f"robot/{side}_foot{i}_collision"
            mu = params.pedal_friction
            spec.add_pair(
                name=f"contact/pedal/{side}{i}",
                geomname1=foot,
                geomname2=f"pogo/{side}_pedal",
                condim=3,
                friction=[mu, mu, 0, 0, 0],
                solref=[0.008, 1.0],
            )
    hand_pairs(spec, "pogo/handlebar", params=params)
