"""MDP terms for the pogo task: command, observations, rewards, events, terminations.

Conventions: `env.scene["robot"]`, `env.scene["pogo"]`; the pogo entity has one
joint, `compression` (x >= 0), and a free root at the pedal-top plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import matrix_from_quat

from pogo_g1 import pogo
from pogo_g1.robot import ACTIVE, finger_target, is_finger

ACTIVE_JOINTS = {f"{k}_joint" for k in ACTIVE}

G = 9.81


# --------------------------------------------------------------------------- #
# Cached model constants
# --------------------------------------------------------------------------- #


def _consts(env):
    c = getattr(env, "_pogo_consts", None)
    if c is not None:
        return c
    m = env.sim.mj_model
    gas_ids = [i for i in range(m.ntendon) if m.tendon(i).name.startswith("pogo/gas_")]
    c = {
        "gas_ids": gas_ids,
        "gas_knots": torch.as_tensor(
            m.tendon_lengthspring[gas_ids, 1], device=env.device, dtype=torch.float32
        ),
        "springref": float(m.qpos_spring[m.joint("pogo/compression").qposadr[0]]),
        "feet": [m.site(f"robot/{s}_foot").id for s in ["left", "right"]],
        "pedals": [m.geom(f"pogo/{s}_pedal").id for s in ["left", "right"]],
        "pedal_half_z": float(m.geom_size[m.geom("pogo/left_pedal").id, 2]),
        "tip": m.geom("pogo/tip").id,
        "tip_radius": float(m.geom_size[m.geom("pogo/tip").id, 0]),
        "robot_body": m.body("robot/pelvis").id,
        "pogo_body": m.body("pogo/pogo_frame").id,
    }
    env._pogo_consts = c
    return c


# --------------------------------------------------------------------------- #
# Physical quantities
# --------------------------------------------------------------------------- #


def compression(env):
    return env.scene["pogo"].data.joint_pos[:, 0]


def compression_vel(env):
    return env.scene["pogo"].data.joint_vel[:, 0]


def _spring(env):
    """Read the actual native coefficients, including per-world randomisation."""
    jid = env.sim.mj_model.joint("pogo/compression").id
    a = env.sim.model.jnt_stiffness[:, jid]
    bc = env.sim.model.jnt_stiffnesspoly[:, jid]
    return a, bc[:, 0], bc[:, 1]


@requires_model_fields("jnt_stiffness", "jnt_stiffnesspoly", "tendon_stiffnesspoly")
def randomize_spring(env, env_ids, ranges=(0.8, 1.2)):
    """Scale the whole force curve and its potential, never just its linear term."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    jid = env.sim.mj_model.joint("pogo/compression").id
    scale = torch.empty(len(env_ids), device=env.device).uniform_(*ranges)
    m = env.sim.mj_model
    env.sim.model.jnt_stiffness[env_ids, jid] = float(m.jnt_stiffness[jid]) * scale
    coeff = torch.as_tensor(
        m.jnt_stiffnesspoly[jid], device=env.device, dtype=scale.dtype
    )
    env.sim.model.jnt_stiffnesspoly[env_ids, jid] = scale[:, None] * coeff
    ids = _consts(env)["gas_ids"]
    coeff = torch.as_tensor(
        m.tendon_stiffnesspoly[ids], device=env.device, dtype=scale.dtype
    )
    env.sim.model.tendon_stiffnesspoly[env_ids[:, None], ids] = (
        scale[:, None, None] * coeff
    )


@requires_model_fields("pair_friction")
def randomize_contacts(env, env_ids, tip_patch_radius=0.008, grip_torsion=0.005):
    """Material-pair uncertainty, with one isotropic coefficient per interface."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    model = env.sim.mj_model
    for kind, bounds in (
        ("tip", (0.4, 1.0)),
        ("pedal", (0.5, 1.0)),
        ("grip", (0.7, 1.1)),
    ):
        ids = [
            i
            for i in range(model.npair)
            if model.pair(i).name.startswith(f"contact/{kind}/")
        ]
        mu = torch.empty(len(env_ids), device=env.device).uniform_(*bounds)
        for i in ids:
            base = torch.as_tensor(
                model.pair_friction[i], device=env.device, dtype=mu.dtype
            )
            value = base.repeat(len(env_ids), 1)
            value[:, :2] = mu[:, None]
            spin = (
                2 / 3 * tip_patch_radius
                if kind == "tip"
                else grip_torsion
                if kind == "grip"
                else 0.0
            )
            value[:, 2] = spin * mu
            env.sim.model.pair_friction[env_ids, i] = value


def spring_force(env):
    """Restoring-force magnitude along the passive slider [N]."""
    a, b, c = _spring(env)
    y = compression(env) - _consts(env)["springref"]
    extra, coeff = _gas_terms(env)
    return y * (a + y * (b + y * c)) + (coeff * extra**3).sum(-1)


def _gas_terms(env):
    const = _consts(env)
    extra = (compression(env)[:, None] - const["gas_knots"]).clamp_min(0)
    coeff = env.sim.model.tendon_stiffnesspoly[:, const["gas_ids"], 1]
    return extra, coeff


def spring_energy(env):
    """Native polynomial potential relative to full extension [J]."""
    a, b, c = _spring(env)
    ref = _consts(env)["springref"]
    y, y0 = compression(env) - ref, -ref
    extra, coeff = _gas_terms(env)
    return (
        a / 2 * (y * y - y0 * y0)
        + b / 3 * (y**3 - y0**3)
        + c / 4 * (y**4 - y0**4)
        + (coeff * extra**4).sum(-1) / 4
    )


def energy_height(env):
    """Terrain-relative height-tracking surrogate [m].

    H = (z_com - z_rest) + v_com,z^2 / 2g + E_spring(x) / (M g), with z_rest the
    system COM height in the mounted pose. This is a height-tracking surrogate,
    not total mechanical energy: internal joint/relative kinetic energy is not
    included. The spring contribution uses the simulator's exact potential.
    """
    rest = env.command_manager.get_term("hop").cfg.com_rest
    z = env.sim.data.subtree_com[:, 0, 2] - ground_height(env) - rest
    vz = system_com_velocity(env)[:, 2]
    return z + vz * vz / (2 * G) + spring_energy(env) / (system_mass(env) * G)


def tip_clearance(env):
    c = _consts(env)
    return _terrain_data(env).heights[:, 0] - c["tip_radius"]


def _terrain_data(env):
    # mjlab normally senses after rewards/terminations. Refresh once at their
    # current control step, otherwise they would read the previous step's rays.
    if getattr(env, "_terrain_sense_step", -1) != env.common_step_counter:
        env.sim.sense()
        env._terrain_sense_step = env.common_step_counter
    return env.scene["tip_ground_height"].data


def ground_height(env):
    return _terrain_data(env).hit_pos_w[:, 0, 2]


def terrain_missing(env):
    ray = _terrain_data(env)
    return (ray.distances[:, 0] < 0) | (ray.normals_w[:, 0, 2] <= 0)


def _masses(env):
    c = _consts(env)
    masses = env.sim.model.body_subtreemass
    return masses[:, c["robot_body"]], masses[:, c["pogo_body"]]


def system_mass(env):
    robot_mass, pogo_mass = _masses(env)
    return robot_mass + pogo_mass


def tip_contact(env):
    return env.scene["tip_contact"].data.found.reshape(env.num_envs, -1).any(-1)


def system_com_velocity(env):
    """Mass-weighted robot + pogo COM velocity (world frame), (N, 3)."""
    robot_mass, pogo_mass = _masses(env)
    v = (
        robot_mass[:, None] * env.scene["robot_com_vel"].data
        + pogo_mass[:, None] * env.scene["pogo_com_vel"].data
    )
    return v / (robot_mass + pogo_mass)[:, None]


def _yaw_frame(env, v_w):
    h = env.scene["robot"].data.heading_w
    cos, sin = torch.cos(h), torch.sin(h)
    return torch.stack(
        [cos * v_w[:, 0] + sin * v_w[:, 1], -sin * v_w[:, 0] + cos * v_w[:, 1]], -1
    )


def com_velocity_yaw(env):
    return _yaw_frame(env, system_com_velocity(env))


def yaw_rate(env):
    return env.scene["robot"].data.root_link_ang_vel_w[:, 2]


def centroidal_yaw_rate(env):
    """Vertical angular momentum of robot + pogo about their joint COM over the
    matching moment of inertia, L_z / I_zz. Its long-run mean is the turning
    rate, but only external torque changes L_z, so the joints' motion within
    a hop (and their exploration noise) barely moves it."""
    parts = []
    for name in ("robot", "pogo"):
        e = env.scene[name]
        ids = e.indexing.body_ids
        m = env.sim.model.body_mass[..., ids].expand(env.num_envs, -1)
        inertia = env.sim.model.body_inertia[..., ids, :].expand(env.num_envs, -1, -1)
        d = e.data
        parts.append(
            (m, inertia, d.body_com_pos_w, d.body_com_quat_w, d.body_com_vel_w)
        )
    mass = sum(p[0].sum(-1) for p in parts)
    com = sum((p[0][..., None] * p[2]).sum(1) for p in parts) / mass[:, None]
    lz, izz = 0.0, 0.0
    for m, inertia, x, q, v in parts:
        rot = matrix_from_quat(q)
        i_w = rot @ torch.diag_embed(inertia) @ rot.transpose(-1, -2)
        r = x - com[:, None]
        lz = lz + (i_w[..., 2, :] * v[..., 3:]).sum((-1, -2))
        lz = lz + (m * (r[..., 0] * v[..., 1] - r[..., 1] * v[..., 0])).sum(1)
        izz = izz + i_w[..., 2, 2].sum(1) + (m * r[..., :2].square().sum(-1)).sum(1)
    return lz / izz


def foot_error(env):
    """Per-foot distance [m] from the foot site to its pedal-top centre, (N, 2)."""
    c = _consts(env)
    feet = env.sim.data.site_xpos[:, c["feet"]]
    centre = env.sim.data.geom_xpos[:, c["pedals"]]
    normal = env.sim.data.geom_xmat[:, c["pedals"]].reshape(-1, 2, 3, 3)[..., 2]
    top = centre + normal * (c["pedal_half_z"] - 0.002)
    return torch.linalg.vector_norm(feet - top, dim=-1)


def grip_wrench(env):
    """Per-hand net contact force [N] and moment about the grasp centre [N m], (N, 2)."""
    forces, torques = [], []
    for side in ("left", "right"):
        s = env.scene[f"{side}_grip_contact"].data
        f = s.force[:, 0]
        centre = env.sim.data.site_xpos[
            :, env.sim.mj_model.site(f"robot/{side}_grasp").id
        ]
        t = s.torque[:, 0] + torch.cross(s.pos[:, 0] - centre, f, dim=-1)
        forces.append(torch.linalg.vector_norm(f, dim=-1))
        torques.append(torch.linalg.vector_norm(t, dim=-1))
    return torch.stack(forces, -1), torch.stack(torques, -1)


def grip_force_max(env):
    return grip_wrench(env)[0].max(-1).values


def grip_torque_max(env):
    return grip_wrench(env)[1].max(-1).values


def grip_distance_max(env):
    m = env.sim.mj_model
    hands = [m.site(f"robot/{s}_grasp").id for s in ("left", "right")]
    handles = [m.site(f"pogo/{s}_grip").id for s in ("left", "right")]
    return (
        torch.linalg.vector_norm(
            env.sim.data.site_xpos[:, hands] - env.sim.data.site_xpos[:, handles],
            dim=-1,
        )
        .max(-1)
        .values
    )


def grip_contact_fraction(env):
    return (
        torch.stack(
            [
                env.scene[f"{s}_grip_contact"]
                .data.found.reshape(env.num_envs, -1)
                .any(-1)
                for s in ("left", "right")
            ],
            -1,
        )
        .float()
        .mean(-1)
    )


def finger_torque_max(env):
    e = env.scene["robot"]
    ids = [i for i, n in enumerate(e.actuator_names) if is_finger(n)]
    return e.data.actuator_force[:, ids].abs().max(-1).values


def grip_lost(env):
    # One hand leaving its initial handle centre by > 5 cm loses the grasp.
    return grip_distance_max(env) > 0.05


def tilt_cos(env, name):
    return -env.scene[name].data.projected_gravity_b[:, 2]


# --------------------------------------------------------------------------- #
# Command: [apex height h*, vx, vy (yaw frame), yaw rate]
# --------------------------------------------------------------------------- #


class HopCommand(CommandTerm):
    cfg: HopCommandCfg

    def __init__(self, cfg: HopCommandCfg, env):
        super().__init__(cfg, env)
        self.cmd = torch.zeros(self.num_envs, 4, device=self.device)
        self._external_command = None
        # Apex ceiling [m] and its window: [landings, apex sum, command sum, falls].
        self.ceiling = torch.tensor(cfg.ranges.apex[1], device=self.device)
        self.window = torch.zeros(4, device=self.device)
        # Seconds since each world's command last changed: height takes a few
        # hops to settle after a change (pumping adds bounded energy per stance).
        self.held = torch.zeros(self.num_envs, device=self.device)
        for k in ["apex_error", "vel_error"]:
            self.metrics[k] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self.cmd

    def _resample_command(self, env_ids):
        self.held[env_ids] = 0.0
        if self._external_command is not None:
            self.cmd[env_ids] = self._external_command
            return
        r = self.cfg.ranges
        n = len(env_ids)
        u = lambda lo_hi: torch.empty(n, device=self.device).uniform_(*lo_hi)  # noqa: E731
        lo = r.apex[0]
        self.cmd[env_ids, 0] = lo + (self.ceiling - lo) * torch.rand(
            n, device=self.device
        )
        self.cmd[env_ids, 1] = u(r.vx)
        self.cmd[env_ids, 2] = u(r.vy)
        self.cmd[env_ids, 3] = u(r.yaw_rate)
        still = torch.rand(n, device=self.device) < self.cfg.rel_in_place
        self.cmd[env_ids[still], 1:] = 0.0

    def _update_command(self, env_ids=None):
        if env_ids is None:
            self.held += self._env.step_dt

    def set_command(self, apex, vx=0.0, vy=0.0, yaw_rate=0.0):
        """Hold an external command across resampling and resets.

        Keyboard and higher-level controllers use the same [m, m/s, m/s, rad/s]
        interface. Linear velocity is in the robot's yaw frame. No physical state
        or joint target is changed here; only the policy's command observation.
        """
        values = torch.as_tensor(
            [apex, vx, vy, yaw_rate], device=self.device, dtype=self.cmd.dtype
        )
        if not torch.isfinite(values).all() or apex < 0:
            raise ValueError("commands must be finite with nonnegative apex")
        self._external_command = values
        self.cmd[:] = values
        self.held[:] = 0.0

    def _update_metrics(self):
        steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["apex_error"] += (
            energy_height(self._env) - self.cmd[:, 0]
        ).abs() / steps
        self.metrics["vel_error"] += (
            torch.linalg.vector_norm(
                com_velocity_yaw(self._env) - self.cmd[:, 1:3], dim=-1
            )
            / steps
        )


@dataclass
class HopRanges:
    apex: tuple[float, float] = (0.05, 0.15)
    vx: tuple[float, float] = (0.0, 0.0)
    vy: tuple[float, float] = (0.0, 0.0)
    yaw_rate: tuple[float, float] = (0.0, 0.0)


@dataclass(kw_only=True)
class HopCommandCfg(CommandTermCfg):
    ranges: HopRanges = field(default_factory=HopRanges)
    rel_in_place: float = 0.2
    com_rest: float = 0.0  # system COM height in the mounted pose [m]; H = 0 there
    apex_cap: float = 0.0  # ceiling curriculum limit [m]; <= ranges.apex[1] disables

    def build(self, env):
        return HopCommand(self, env)


def hop_command(env):
    return env.command_manager.get_command("hop")


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


def imu_acc(env):
    """Pelvis accelerometer (specific force) in g units; ~0 in free flight."""
    return env.scene["robot/imu_lin_acc"].data / G


# (kind, dim) in order; kinds tell symmetry.py how each block mirrors.
PRIVILEGED_LAYOUT = [
    ("scalar", 1),  # compression
    ("scalar", 1),  # compression velocity
    ("scalar", 1),  # energy height
    ("scalar", 1),  # tip contact
    ("polar", 3),  # tip force (yaw frame)
    ("scalar", 1),  # tip clearance
    ("polar", 3),  # pogo projected gravity (pogo frame)
    ("polar", 3),  # pogo linear velocity (yaw frame)
    ("axial", 3),  # pogo angular velocity (yaw frame)
    ("polar", 3),  # pogo position relative to pelvis (yaw frame)
    ("polar", 3),  # pelvis linear velocity (yaw frame)
    ("planar", 2),  # system COM velocity (yaw frame)
    ("pair", 2),  # foot-pedal error (left, right)
    ("pair", 2),  # grip force (left, right)
]


def yaw_rotate(env, v):
    """World-frame 3-vectors into the robot heading frame."""
    h = env.scene["robot"].data.heading_w
    c, s = torch.cos(h), torch.sin(h)
    return torch.stack(
        [c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1
    )


def privileged(env):
    """Critic-only state: pogo and contact quantities the actor cannot sense."""
    robot, pogo = env.scene["robot"].data, env.scene["pogo"].data
    tip_force = env.scene["tip_contact"].data.force.reshape(env.num_envs, -1, 3).sum(1)
    return torch.cat(
        [
            compression(env)[:, None] * 10,
            compression_vel(env)[:, None],
            energy_height(env)[:, None] * 10,
            tip_contact(env).float()[:, None],
            yaw_rotate(env, tip_force) / 1000.0,
            tip_clearance(env)[:, None] * 10,
            pogo.projected_gravity_b,
            yaw_rotate(env, pogo.root_link_lin_vel_w),
            yaw_rotate(env, pogo.root_link_ang_vel_w),
            yaw_rotate(env, pogo.root_link_pos_w - robot.root_link_pos_w),
            yaw_rotate(env, robot.root_link_lin_vel_w),
            com_velocity_yaw(env),
            foot_error(env) * 10,
            grip_wrench(env)[0] / 100.0,
        ],
        -1,
    )


# --------------------------------------------------------------------------- #
# Rewards
# --------------------------------------------------------------------------- #


def track_com_velocity(env, std: float):
    err = com_velocity_yaw(env) - hop_command(env)[:, 1:3]
    return torch.exp(-err.square().sum(-1) / std**2)


def track_yaw_rate(env, std: float):
    err = yaw_rate(env) - hop_command(env)[:, 3]
    return torch.exp(-((err / std) ** 2))


def yaw_rate_error_l2(env):
    """Squared instantaneous yaw-rate error. Unlike the Gaussian guide it does
    not saturate, so it keeps penalising a large within-hop twist."""
    return (yaw_rate(env) - hop_command(env)[:, 3]).square()


def upright(env, name: str, std: float):
    # 1 - cos(tilt) ~ tilt^2 / 2
    return torch.exp(-(((1 - tilt_cos(env, name)) / std) ** 2))


def feet_on_pedals(env, std: float):
    return torch.exp(-((foot_error(env).max(-1).values / std) ** 2))


def height_score(error, std: float):
    """Bounded tracking reward with a broader tail than the legacy Gaussian."""
    return 1 / (1 + (error / std).square())


def track_height_energy(env, std: float):
    """Weak dense guide; in flight, retain the command used at takeoff."""
    hops(env)
    target = torch.where(env._was_contact, hop_command(env)[:, 0], env._air_command)
    return height_score(energy_height(env) - target, std)


def height_potential(env, std: float, gamma: float):
    """Potential-based shaping of the height guide (Ng et al. 1999):
    F = gamma * Phi(s') - Phi(s), Phi = -|H - h| / std, with the command (the
    takeoff command in flight) part of the state, Phi = 0 at termination and
    no shaping on the first step of an episode. Preserves optimal policies."""
    hops(env)
    target = torch.where(env._was_contact, hop_command(env)[:, 0], env._air_command)
    phi = -(energy_height(env) - target).abs() / std
    if not hasattr(env, "_phi_prev"):
        env._phi_prev = phi.clone()
    first = env.episode_length_buf <= 1
    end = env.termination_manager.terminated
    shaped = gamma * torch.where(end, torch.zeros_like(phi), phi) - env._phi_prev
    env._phi_prev = phi.clone()
    return torch.where(first, torch.zeros_like(phi), shaped)


def landed_height(env, std: float):
    """Actual apex, scored at valid landing against the takeoff command.

    Pay for elapsed cycle time, not hop count: equal accuracy at a slower
    cadence earns the same undiscounted reward rate. Cap credit at the existing
    2 s continuity limit so waiting cannot accumulate an arbitrary jackpot.
    RewardManager multiplies by dt, hence the division here.
    """
    hop = hops(env)
    score = height_score(env._land_apex - env._land_command, std)
    valid = hop * ~env.termination_manager.terminated
    return valid * score * env._land_period.clamp(max=2.0) / env.step_dt


def apex_ceiling(
    env,
    band: float = 0.05,
    raise_by: float = 0.05,
    min_reach: float = 0.85,
    max_falls_per_landing: float = 0.02,
    settle_s: float = 3.0,
    window: int = 20000,
):
    """Training curriculum: raise the apex ceiling once its top band is practised.

    Over each window of landings commanded within `band` of the ceiling, raise
    by `raise_by` if the policy reaches the band (mean measured apex >= 85% of
    the mean command, over landings at least `settle_s` after the command last
    changed: a 0.15 -> 0.45 m change settles in ~2.2 s, and counting the
    climbing hops held the ceiling at 0.55 m although steady 0.50 m hops were
    on target) and falls per landing are <= 2%, so about two thirds of 20 s
    training episodes at the top band run to the time limit. Falls are counted
    under exploration noise (action std ~0.9-1.0), which inflates them 10-100x
    over the deployed deterministic policy (0.65 m: 2.2% vs 0%; 0.75 m: 5.1%
    vs 0.05% per landing), so deployed safety is judged by evaluation, not by
    this practice gate. The final 3 cm accuracy is left to the reward and
    evaluation. Runs before resets. Returns the ceiling.
    """
    term = env.command_manager.get_term("hop")
    if term.cfg.apex_cap > term.cfg.ranges.apex[1]:
        hop = hops(env).bool()
        top = term.ceiling - band
        land = hop & (env._land_command > top - 1e-6) & (term.held >= settle_s)
        fall = env.termination_manager.terminated & (
            hop_command(env)[:, 0] > top - 1e-6
        )
        term.window += torch.stack(
            [
                land.sum(),
                env._land_apex[land].sum(),
                env._land_command[land].sum(),
                fall.sum(),
            ]
        )
        n, apex, cmd, f = term.window
        full = n >= window
        ok = full & (apex >= min_reach * cmd) & (f <= max_falls_per_landing * n)
        term.ceiling.copy_(
            torch.where(
                ok, (term.ceiling + raise_by).clamp(max=term.cfg.apex_cap), term.ceiling
            )
        )
        term.window.mul_(~full)
    return term.ceiling.expand(env.num_envs)


def landed_velocity(env, std: float):
    """Hop-cycle average planar COM velocity (yaw frame) against the command.

    Scored at each valid landing over landing-to-landing, so the stance and
    flight swing that steering a pogo requires is not penalised, only the
    motion it produces. Paid for elapsed cycle time like `landed_height`.
    """
    hop = hops(env)
    err = torch.linalg.vector_norm(
        env._land_motion[:, :2] - hop_command(env)[:, 1:3], dim=-1
    )
    valid = hop * ~env.termination_manager.terminated
    return (
        valid * height_score(err, std) * env._land_period.clamp(max=2.0) / env.step_dt
    )


def landed_yaw_rate(env, std: float):
    """Hop-cycle average yaw rate against the command, scored like `landed_velocity`."""
    hop = hops(env)
    err = env._land_motion[:, 2] - hop_command(env)[:, 3]
    valid = hop * ~env.termination_manager.terminated
    return (
        valid * height_score(err, std) * env._land_period.clamp(max=2.0) / env.step_dt
    )


def step_velocity(env, std: float):
    """Per-step counterpart of `landed_velocity`: the same kernel and tolerance
    on the instantaneous planar COM velocity, paid every step (equal reward
    per second for equal error)."""
    err = torch.linalg.vector_norm(
        com_velocity_yaw(env) - hop_command(env)[:, 1:3], dim=-1
    )
    return height_score(err, std)


def step_yaw_rate(env, std: float):
    """Per-step counterpart of `landed_yaw_rate`."""
    return height_score(yaw_rate(env) - hop_command(env)[:, 3], std)


def step_centroidal_yaw_rate(env, std: float):
    """Per step, on the centroidal yaw rate instead of the pelvis's."""
    return height_score(centroidal_yaw_rate(env) - hop_command(env)[:, 3], std)


def window_motion(env, tau: float):
    """Exponential moving average of [vx, vy, yaw rate] (time constant tau),
    restarted at each episode's first step; computed once per step."""
    x = torch.cat([com_velocity_yaw(env), yaw_rate(env)[:, None]], -1)
    if getattr(env, "_ema_step", None) == env.common_step_counter:
        return env._ema
    if not hasattr(env, "_ema"):
        env._ema = x.clone()
    a = env.step_dt / tau
    first = (env.episode_length_buf <= 1)[:, None]
    env._ema = torch.where(first, x, env._ema + a * (x - env._ema))
    env._ema_step = env.common_step_counter
    return env._ema


def window_velocity(env, std: float, tau: float):
    """Per-step score of the moving-average planar velocity (the cycle average
    delivered densely)."""
    err = torch.linalg.vector_norm(
        window_motion(env, tau)[:, :2] - hop_command(env)[:, 1:3], dim=-1
    )
    return height_score(err, std)


def window_yaw_rate(env, std: float, tau: float):
    """Per-step score of the moving-average yaw rate."""
    return height_score(window_motion(env, tau)[:, 2] - hop_command(env)[:, 3], std)


def bottom_out(env, stroke: float, margin: float = pogo.BOTTOM_OUT_MARGIN):
    return (compression(env) > stroke - margin).float()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def hops(env):
    """1 at each touchdown after >= 0.1 s of flight with >= 2 cm peak clearance.

    Also records, per landing, the measured tip apex, the takeoff command and
    the average [vx, vy, yaw rate] since the previous valid landing (or reset).
    """
    if not hasattr(env, "_air_time"):
        env._air_time = torch.zeros(env.num_envs, device=env.device)
        env._air_peak = torch.zeros(env.num_envs, device=env.device)
        env._land_apex = torch.zeros(env.num_envs, device=env.device)
        env._air_command = torch.zeros(env.num_envs, device=env.device)
        env._land_command = torch.zeros(env.num_envs, device=env.device)
        env._hop_elapsed = torch.zeros(env.num_envs, device=env.device)
        env._land_period = torch.zeros(env.num_envs, device=env.device)
        env._cycle_motion = torch.zeros(env.num_envs, 3, device=env.device)
        env._land_motion = torch.zeros(env.num_envs, 3, device=env.device)
        env._was_contact = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        env._hop_step = -1
        env._hop = torch.zeros(env.num_envs, device=env.device)
    if env._hop_step == env.common_step_counter:
        return env._hop
    contact = tip_contact(env)
    takeoff = ~contact & env._was_contact
    env._air_command = torch.where(takeoff, hop_command(env)[:, 0], env._air_command)
    landing = contact & ~env._was_contact
    env._hop = (landing & (env._air_time >= 0.1) & (env._air_peak >= 0.02)).float()
    env._land_apex = torch.where(landing, env._air_peak, 0.0)
    env._land_command = torch.where(landing, env._air_command, 0.0)
    env._hop_elapsed += env.step_dt
    motion = torch.cat([com_velocity_yaw(env), yaw_rate(env)[:, None]], -1)
    env._cycle_motion += motion * env.step_dt
    hop = env._hop.bool()
    env._land_period = torch.where(hop, env._hop_elapsed, 0.0)
    env._land_motion = torch.where(
        hop[:, None], env._cycle_motion / env._hop_elapsed[:, None], 0.0
    )
    env._cycle_motion = torch.where(hop[:, None], 0.0, env._cycle_motion)
    env._hop_elapsed = torch.where(hop, 0.0, env._hop_elapsed)
    env._air_time = torch.where(contact, 0.0, env._air_time + env.step_dt)
    env._air_peak = torch.where(
        contact, 0.0, torch.maximum(env._air_peak, tip_clearance(env))
    )
    env._was_contact = contact
    env._hop_step = env.common_step_counter
    return env._hop


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


def reset_mixed(
    env,
    env_ids,
    airborne_prob: float,
    lift: tuple[float, float],
    vz: tuple[float, float],
    vxy: float,
    yaw_rate: float,
):
    """Mounted default state; a fraction start mid-hop (lifted, moving vertically).

    Both objects start with matched velocities and the mounted finger pose. They
    remain independent free bodies in contact runs; no attachment is created.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "_spawn_height"):
        # Correct the native generator's conservative patch height to the actual
        # surface under the mounted tip. One CPU ray per unique static origin,
        # once per environment construction; no host/device copy in the step loop.
        origins = env.scene.env_origins.cpu().numpy()
        unique, inverse = np.unique(origins, axis=0, return_inverse=True)
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        offsets = []
        for origin in unique:
            point = origin.astype(np.float64) + [0, 0, 2.0]
            distance = mujoco.mj_ray(
                model,
                data,
                point,
                np.array([0.0, 0.0, -1.0]),
                np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                True,
                -1,
                np.zeros(1, dtype=np.int32),
            )
            if distance < 0:
                raise ValueError("terrain has no support under the reset origin")
            offsets.append(2.0 - distance)
        env._spawn_height = torch.as_tensor(
            np.asarray(offsets)[inverse], device=env.device, dtype=torch.float32
        )
    n = len(env_ids)
    air = torch.rand(n, device=env.device) < airborne_prob
    dz = torch.where(air, torch.empty(n, device=env.device).uniform_(*lift), 0.0)
    v = torch.zeros(n, 3, device=env.device)
    v[:, 2] = torch.where(air, torch.empty(n, device=env.device).uniform_(*vz), 0.0)
    v[:, :2] = torch.where(
        air[:, None], torch.empty(n, 2, device=env.device).uniform_(-vxy, vxy), 0.0
    )
    w = torch.zeros(n, 3, device=env.device)
    w[:, 2] = torch.where(
        air, torch.empty(n, device=env.device).uniform_(-yaw_rate, yaw_rate), 0.0
    )
    for name in ["robot", "pogo"]:
        e = env.scene[name]
        s = e.data.default_root_state[env_ids].clone()
        s[:, :3] += env.scene.env_origins[env_ids]
        s[:, 2] += dz + env._spawn_height[env_ids]
        s[:, 7:10] = v
        s[:, 10:13] = w
        e.write_root_state_to_sim(s, env_ids=env_ids)
        e.write_joint_state_to_sim(
            e.data.default_joint_pos[env_ids].clone(),
            e.data.default_joint_vel[env_ids].clone(),
            env_ids=env_ids,
        )
        if name == "robot":
            ids = [i for i, name in enumerate(e.joint_names) if is_finger(name)]
            ids = [i for i in ids if e.joint_names[i].split("_", 1)[1] in ACTIVE_JOINTS]
            if ids:
                target = torch.tensor(
                    [finger_target(e.joint_names[i]) for i in ids],
                    device=env.device,
                ).expand(n, -1)
                e.set_joint_position_target(
                    target,
                    joint_ids=torch.tensor(ids, device=env.device),
                    env_ids=env_ids,
                )
    if hasattr(env, "_air_time"):
        # The first landing after an airborne reset ends a flight the policy did
        # not create: no valid hop, so no height credit.
        env._air_time[env_ids] = torch.where(air, -1.0, 0.0)
        env._air_peak[env_ids] = 0
        env._land_apex[env_ids] = 0
        env._air_command[env_ids] = 0
        env._land_command[env_ids] = 0
        env._hop_elapsed[env_ids] = 0
        env._land_period[env_ids] = 0
        env._cycle_motion[env_ids] = 0
        env._land_motion[env_ids] = 0
        env._hop[env_ids] = 0
        env._was_contact[env_ids] = True


# --------------------------------------------------------------------------- #
# Terminations
# --------------------------------------------------------------------------- #


def tilted(env, name: str, max_deg: float):
    return tilt_cos(env, name) < torch.cos(torch.deg2rad(torch.tensor(max_deg)))


def feet_off_pedals(env, limit: float):
    return foot_error(env).max(-1).values > limit


def illegal_contact(env, sensor: str):
    return env.scene[sensor].data.found.reshape(env.num_envs, -1).any(-1)


def nonfinite(env):
    return ~torch.isfinite(env.sim.data.qpos).all(-1)
