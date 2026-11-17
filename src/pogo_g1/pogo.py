"""Parametric passive pogo stick.

Physics is set by PogoParams; geometry comes from the dimensionally adapted
Vurtego art asset (see docs/ASSETS.md). Meshes are visual only (no mass, no
collision). Collision/inertia use primitive proxies of the same dimensions.

The gas spring follows P V**n = constant, with absolute pressure inside and
atmospheric pressure outside. A positive sum of native polynomial springs runs
on CPU and Warp, including the high-compression tail. Its exact potential is used
by rewards and diagnostics. These are documented design parameters, not a V4
identification. See DESIGN.md for provenance and the model revision boundary.
"""

import json
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import mujoco
import numpy as np
from numpy.polynomial.legendre import leggauss

from pogo_g1 import contact

DEFAULT_ASSET = Path("data/assets/vurtego/small-v4-pro-v3")
PHYSICS_VERSION = "pneumatic-v5"
PHYSICS_TIMESTEP = 0.002  # 10 steps per 50 Hz action; checked against 1 ms
ATMOSPHERE = 101325.0  # Pa
BOTTOM_OUT_MARGIN = 0.04  # m: begin penalising the final 40 mm of travel


@dataclass(frozen=True)
class PogoParams:
    pressure: float = 128000.0  # gauge Pa; G1 weight in lb / 4 psi (~18.6 psi)
    bore: float = 0.0762  # patent example: 3 in piston diameter
    volume: float = 0.00309  # m^3, including disclosed dead volume (ratio 2.76)
    exponent: float = 1.3  # polytropic exponent, between isothermal and adiabatic
    damping: float = 10.0  # N s/m, viscous losses
    seal_friction: float = 12.0  # N, native Coulomb frictionloss
    stroke: float = 0.432  # m, Small V4 Pro inferred 17 in travel
    frame_mass: float = 3.64  # kg (tube, pedals, handles); estimated split
    plunger_mass: float = 0.9  # kg (unsprung); total 4.54 kg from Small FAQ

    def __post_init__(self):
        for k, v in asdict(self).items():
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{k} must be finite and nonnegative")
        if (
            min(
                self.pressure,
                self.bore,
                self.volume,
                self.stroke,
                self.frame_mass,
                self.plunger_mass,
            )
            <= 0
        ):
            raise ValueError(
                "pressure, bore, volume, stroke and masses must be positive"
            )
        if not 1.0 <= self.exponent <= 1.4:
            raise ValueError("gas exponent must be between 1.0 and 1.4")
        if self.area * self.stroke >= self.volume:
            raise ValueError("stroke must leave positive gas volume")

    @property
    def area(self) -> float:
        return np.pi * self.bore**2 / 4

    @property
    def preload(self) -> float:
        return self.pressure * self.area

    @property
    def mass(self) -> float:
        return self.frame_mass + self.plunger_mass

    def gas_force(self, x):
        """Reference polytropic force; x is compression in metres."""
        ratio = self.volume / (self.volume - self.area * x)
        return self.area * (
            (ATMOSPHERE + self.pressure) * ratio**self.exponent - ATMOSPHERE
        )

    def gas_energy(self, x):
        """Integral of the reference force from 0 to x [J]."""
        ratio = self.volume / (self.volume - self.area * x)
        work = (
            np.log(ratio)
            if self.exponent == 1.0
            else (ratio ** (self.exponent - 1) - 1) / (self.exponent - 1)
        )
        return (
            ATMOSPHERE + self.pressure
        ) * self.volume * work - ATMOSPHERE * self.area * x

    @cached_property
    def spring(self) -> tuple[float, tuple[float, float, float]]:
        """Preloaded Taylor cubic; positive tendon terms supply the remainder.

        The unshifted cubic has positive derivative everywhere. Shifting to its
        unique negative root gives a globally restorative native joint spring.
        """
        n, rate = self.exponent, self.area / self.volume
        scale = (ATMOSPHERE + self.pressure) * self.area
        c1 = scale * n * rate
        c2 = scale * n * (n + 1) * rate**2 / 2
        c3 = scale * n * (n + 1) * (n + 2) * rate**3 / 6
        roots = np.polynomial.polynomial.polyroots([self.preload, c1, c2, c3])
        ref = float(next(r.real for r in roots if abs(r.imag) < 1e-8))
        return ref, (c1 + 2 * c2 * ref + 3 * c3 * ref**2, c2 + 3 * c3 * ref, c3)

    @cached_property
    def gas_terms(self):
        """Positive quadrature of the Taylor remainder, as native cubic springs.

        F(x) = Taylor3(x) + integral_0^x F''''(t) (x-t)^3 / 6 dt.
        Twelve Gauss nodes give <0.03% error through 10 mm overtravel for the
        tested 80-180 kPa, n=1-1.4 and compression ratios up to 4.77. Each term
        k*max(0,x-t)^3 has nonnegative stiffness and an exact quartic potential.
        Tendons wrap only the pogo slider: no new DOF, actuator or attachment.
        """
        end = self.stroke + 0.01
        if self.area * end >= self.volume:
            raise ValueError("gas chamber must leave volume at 10 mm stop overtravel")
        nodes, weights = leggauss(12)
        knots, weights = (nodes + 1) * end / 2, weights * end / 2
        n, rate = self.exponent, self.area / self.volume
        coeff = (
            weights
            * (ATMOSPHERE + self.pressure)
            * self.area
            * rate**4
            * n
            * (n + 1)
            * (n + 2)
            * (n + 3)
            * (1 - rate * knots) ** (-n - 4)
            / 6
        )
        return knots, coeff

    def add_spring(self, spec, joint_name="compression"):
        """Install the complete conservative gas approximation on one joint."""
        joint = spec.joint(joint_name)
        joint.springref, joint.stiffness = self.spring
        for i, (knot, coeff) in enumerate(zip(*self.gas_terms, strict=True)):
            tendon = spec.add_tendon(
                name=f"gas_{i}",
                springlength=[-10 * self.stroke, knot],
                stiffness=[0, 0, coeff],
                limited=False,
                group=5,
            )
            tendon.wrap_joint(joint_name, 1.0)
        grid = np.linspace(-0.005, self.stroke + 0.01, 4097)
        error = np.max(np.abs(self.spring_force(grid) / self.gas_force(grid) - 1))
        if error > 0.02:
            raise ValueError(f"gas spring needs more resolution: error={error:.3%}")

    def spring_force(self, x):
        ref, (a, b, c) = self.spring
        x = np.asarray(x)
        y = x - ref
        knots, coeff = self.gas_terms
        return y * (a + y * (b + y * c)) + (
            np.maximum(0, x[..., None] - knots) ** 3 * coeff
        ).sum(-1)

    def spring_energy(self, x):
        ref, (a, b, c) = self.spring
        x = np.asarray(x)
        y, y0 = x - ref, -ref
        knots, coeff = self.gas_terms
        return (
            a / 2 * (y * y - y0 * y0)
            + b / 3 * (y**3 - y0**3)
            + c / 4 * (y**4 - y0**4)
            + (np.maximum(0, x[..., None] - knots) ** 4 * coeff).sum(-1) / 4
        )


NOMINAL = PogoParams()


def load_geometry(asset_dir: Path = DEFAULT_ASSET) -> dict:
    return json.loads((Path(asset_dir) / "manifest.json").read_text())


def handle_radius(asset_dir: Path = DEFAULT_ASSET) -> float:
    return float(
        np.ptp(load_geometry(asset_dir)["bounds"]["Grip_003_Rubber_Mat_0"], axis=0)[0]
        / 2
    )


def build_spec(
    params: PogoParams = NOMINAL,
    asset_dir: Path | None = DEFAULT_ASSET,
    floor: bool = False,
    contacts: contact.ContactParams = contact.DEFAULT,
) -> mujoco.MjSpec:
    """Frame origin = pedal top plane on the tube axis; x forward, z up."""
    geo = load_geometry(asset_dir or DEFAULT_ASSET)
    b = {k: np.asarray(v) for k, v in geo["bounds"].items()}
    spec = mujoco.MjSpec()
    spec.modelname = "pogo"
    spec.compiler.degree = False
    if floor:
        contact.configure_solver(spec)
        spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1]
        )

    tip_drop = geo["pedal_top_above_tip_bottom_m"]
    frame = spec.worldbody.add_body(name="pogo_frame", pos=[0, 0, tip_drop])
    frame.add_freejoint(name="pogo_root")

    cyl = b["Cylinder_Base_blinn1_0"]
    tube_r = float(cyl[1, 0])
    tube_h = float(np.ptp(cyl[:, 2]))
    # Compound rigid-body inertia from tube (70%), pedals (20%) and bar (10%).
    # Distribution remains a design estimate; MuJoCo applies parallel-axis terms.
    frame.add_geom(
        name="outer_tube",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=cyl.mean(0),
        size=[tube_r, tube_h / 2],
        group=3,
        mass=0.7 * params.frame_mass,
    )

    half = np.asarray(geo["pedal_half_size"])
    for side, c in geo["pedal_centres"].items():
        frame.add_geom(
            name=f"{side}_pedal",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=c,
            size=half,
            group=3,
            mass=0.1 * params.frame_mass,
        )
    # Handlebar = the measured span of the two rubber grips. Hand contacts use the
    # same stiff contact time constant as the finger geoms (robot.SOLREF).
    radius = handle_radius(asset_dir or DEFAULT_ASSET)
    g = geo["grips"]
    lo = geo["bounds"]["Grip_005_Rubber_Mat_0"][0][1]
    hi = geo["bounds"]["Grip_003_Rubber_Mat_0"][1][1]
    x, _, z = g["left"]
    frame.add_geom(
        name="handlebar",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[x, lo + radius, z, x, hi - radius, z],
        size=[radius, 0, 0],
        condim=4,
        friction=[1.0, 0.005, 0.0001],
        solref=[0.004, 1.0],
        group=3,
        mass=0.1 * params.frame_mass,
    )
    for side in ["left", "right"]:
        frame.add_site(name=f"{side}_grip", pos=g[side], size=[0.01] * 3, group=5)
    frame.add_site(name="pedal_centre", pos=[0, 0, 0], group=5)

    plunger = frame.add_body(name="pogo_plunger")
    plunger.add_joint(
        name="compression",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        range=[0, params.stroke],
        limited=True,
        damping=[params.damping, 0, 0],
        frictionloss=params.seal_friction,
        solref_limit=[2 * PHYSICS_TIMESTEP, 1],  # stiffest MuJoCo allows
        solimp_limit=[0.99, 0.999, 0.001, 0.5, 2],
    )
    params.add_spring(spec)
    rod = b["Slide_Shaft_Silver_MAT_0"]
    rod_r, rod_h = float(rod[1, 0]), float(np.ptp(rod[:, 2]))
    plunger.add_geom(
        name="shaft",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=rod.mean(0),
        size=[rod_r, rod_h / 2],
        group=3,
        mass=0.8 * params.plunger_mass,
    )
    tip = b["Grip_Tip_Rubber_Mat_0"]
    plunger.add_geom(
        name="tip",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0, 0, float(tip[0, 2]) + float(np.ptp(tip[:, 0])) / 2],
        size=[float(np.ptp(tip[:, 0])) / 2, 0, 0],
        group=3,
        mass=0.2 * params.plunger_mass,
    )
    plunger.add_site(name="tip_bottom", pos=[0, 0, float(tip[0, 2])], group=5)
    spec.add_exclude(bodyname1="pogo_frame", bodyname2="pogo_plunger")
    if floor:
        contact.tip_pair(spec, "tip", "floor", contacts)

    if asset_dir is not None:
        moving = set(geo["moving_parts"])
        spec.add_texture(
            name="vurtego_label",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str((Path(asset_dir) / "basecolor.png").resolve()),
        )
        mats = [".35 .35 .35 1", ".04 .04 .04 1", "1 1 1 1", ".025 .025 .025 1"]
        for i, rgba in enumerate(mats):
            m = spec.add_material(
                name=f"pogo_mat{i}", rgba=[float(v) for v in rgba.split()]
            )
            if i == 2:
                m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "vurtego_label"
        material_of = geo.get("materials", {})
        for name in sorted(geo["bounds"]):
            spec.add_mesh(
                name=f"pogo_{name}",
                file=str((Path(asset_dir) / f"{name}.obj").resolve()),
            )
            (plunger if name in moving else frame).add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"pogo_{name}",
                material=f"pogo_mat{material_of.get(name, 0)}",
                contype=0,
                conaffinity=0,
                density=0,
                group=2,
            )
    return spec


def rest_height(asset_dir: Path = DEFAULT_ASSET) -> float:
    """Pedal-top height above ground with the plunger at full extension."""
    return load_geometry(asset_dir)["pedal_top_above_tip_bottom_m"]
