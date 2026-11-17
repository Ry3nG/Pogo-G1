"""mjlab environment config for G1 riding a passive pogo stick (DESIGN.md sections 3-4)."""

import ast
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mj
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import BuiltinSensorCfg, ContactMatch, ContactSensorCfg, ObjRef
from mjlab.sensor.raycast_sensor import GridPatternCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pogo_g1 import assembly, contact, mdp, pogo, robot, terrain

# Per-world constraint buffers, sized from the CPU scene (tests/test_physics.py).
NCONMAX = 160
NJMAX = 1000


@dataclass
class TaskCfg:
    """Knobs that define an experiment. Everything else is fixed by the plan."""

    params: pogo.PogoParams = field(default_factory=pogo.PogoParams)
    contacts: contact.ContactParams = field(default_factory=contact.ContactParams)
    terrain: str = "flat"
    terrain_height: float = 0.03
    terrain_slope: float = 0.08  # rise/run, not degrees
    asset_dir: Path = pogo.DEFAULT_ASSET
    worlds: int = 4096
    episode_s: float = 20.0
    timestep: float = pogo.PHYSICS_TIMESTEP
    decimation: int = 10  # 50 Hz policy
    history: int = 5
    base_velocity: bool = True  # actor sees the pelvis velocimeter (a state estimate)
    # Steering scored per hop cycle ("hop"), per step ("step"), per step on
    # a moving average with time constant steer_tau ("window"), or per step
    # with the yaw rate taken from angular momentum ("centroidal"); the height
    # guide is a Gaussian score ("gaussian") or potential-based shaping
    # ("potential"). yaw_weight scales the yaw-rate score (gain-matched control).
    steer_timescale: str = "hop"
    steer_tau: float = 0.5
    yaw_weight: float = 2.0
    height_guide: str = "gaussian"
    airborne_prob: float = 0.5
    apex: tuple[float, float] = (0.05, 0.15)  # upper end is the initial ceiling
    apex_cap: float = 1.5  # curriculum limit; <= apex[1] disables the curriculum
    vx: tuple[float, float] = (-0.4, 0.5)
    vy: tuple[float, float] = (-0.4, 0.4)
    yaw_rate: tuple[float, float] = (-0.6, 0.6)
    robust: bool = False  # domain randomisation, pushes, actuator latency
    symmetry: bool = True  # rsl_rl mirror data augmentation
    pose_path: str = ""  # mounted pose saved with the run (reproducible evaluation)
    in_place_frac: float = 0.2  # fraction of commands with zero planar/yaw velocity
    eval_push: float = 0.0  # evaluation: push every ~3 s by +-eval_push m/s (x, y)
    latency_s: float = -1.0  # evaluation: fixed actuator command delay [s]
    grip_strength: float = 1.0  # finger servo torque limits, fraction of rating
    seed: int = 0


def parse_set(task: TaskCfg, items):
    """Apply CLI field=value overrides shared by training and playback."""
    for item in items:
        key, value = item.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        if "." in key:
            parent, child = key.split(".", 1)
            if not hasattr(task, parent) or not hasattr(getattr(task, parent), child):
                raise KeyError(key)
            setattr(
                task,
                parent,
                dataclasses.replace(getattr(task, parent), **{child: value}),
            )
        else:
            if not hasattr(task, key):
                raise KeyError(key)
            setattr(task, key, value)
    return task


def _scene(task: TaskCfg, pose: assembly.MountedPose) -> SceneCfg:
    robot_cfg = robot.get_robot_cfg()
    if task.robust:
        # 0-20 ms command latency (physics steps of task.timestep).
        lag = int(round(0.02 / task.timestep))
        robot_cfg.articulation.actuators = tuple(
            dataclasses.replace(a, delay_min_lag=0, delay_max_lag=lag)
            for a in robot_cfg.articulation.actuators
        )
    if task.latency_s >= 0:
        lag = int(round(task.latency_s / task.timestep))
        robot_cfg.articulation.actuators = tuple(
            dataclasses.replace(a, delay_min_lag=lag, delay_max_lag=lag)
            for a in robot_cfg.articulation.actuators
        )
    if task.grip_strength != 1.0:
        robot_cfg.articulation.actuators = tuple(
            dataclasses.replace(a, effort_limit=a.effort_limit * task.grip_strength)
            if any(f in a.target_names_expr[0] for f in ("thumb", "index"))
            else a
            for a in robot_cfg.articulation.actuators
        )
    robot_cfg.init_state = EntityCfg.InitialStateCfg(
        pos=tuple(pose.robot_pos),
        rot=tuple(pose.robot_quat),
        joint_pos=dict(pose.joints),
        joint_vel={".*": 0.0},
    )

    def pogo_spec():
        return pogo.build_spec(task.params, task.asset_dir)

    pogo_cfg = EntityCfg(
        spec_fn=pogo_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(pose.pogo_pos),
            rot=tuple(pose.pogo_quat),
            joint_pos={"compression": 0.0},
            joint_vel={".*": 0.0},
        ),
    )

    def finish(spec: mujoco.MjSpec):
        ground = [g.name for g in spec.body("terrain").geoms]
        # Reserve ray group 0 for terrain, independently of imported robot groups.
        for geom in spec.geoms:
            geom.group = (
                0
                if geom.name in ground
                else (3 if geom.contype or geom.conaffinity else 2)
            )
        contact.assembly_pairs(spec, task.contacts, ground)

    floor = ContactMatch(mode="subtree", pattern="terrain")
    sensors = (
        TerrainHeightSensorCfg(
            name="tip_ground_height",
            frame=ObjRef(type="geom", name="pogo/tip"),
            pattern=GridPatternCfg(size=(0.0, 0.0)),
            ray_alignment="world",
            include_geom_groups=(0,),
            max_distance=5.0,
        ),
        ContactSensorCfg(
            name="tip_contact",
            primary=ContactMatch(mode="geom", pattern="tip", entity="pogo"),
            secondary=floor,
            fields=("found", "force"),
            reduce="netforce",
        ),
        ContactSensorCfg(
            name="robot_ground",
            primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=floor,
            fields=("found",),
        ),
        ContactSensorCfg(
            name="frame_ground",
            primary=ContactMatch(
                mode="geom",
                pattern=(
                    "outer_tube",
                    "left_pedal",
                    "right_pedal",
                    "handlebar",
                    "shaft",
                ),
                entity="pogo",
            ),
            secondary=floor,
            fields=("found",),
        ),
        BuiltinSensorCfg(
            name="robot_com_vel",
            sensor_type="subtreelinvel",
            obj=ObjRef(type="body", name="robot/pelvis"),
        ),
        BuiltinSensorCfg(
            name="pogo_com_vel",
            sensor_type="subtreelinvel",
            obj=ObjRef(type="body", name="pogo/pogo_frame"),
        ),
    )
    sensors += tuple(
        ContactSensorCfg(
            name=f"{side}_grip_contact",
            primary=ContactMatch(
                mode="subtree", pattern=f"{side}_wrist_yaw_link", entity="robot"
            ),
            secondary=ContactMatch(mode="geom", pattern="handlebar", entity="pogo"),
            fields=("found", "force", "torque", "pos"),
            reduce="netforce",
        )
        for side in ("left", "right")
    )
    return SceneCfg(
        num_envs=task.worlds,
        env_spacing=2.5,
        entities={"robot": robot_cfg, "pogo": pogo_cfg},
        terrain=terrain.terrain_cfg(
            task.terrain, task.terrain_height, task.terrain_slope, task.seed
        ),
        sensors=sensors,
        spec_fn=finish,
    )


def make_env_cfg(task: TaskCfg) -> ManagerBasedRlEnvCfg:
    pose = (
        assembly.MountedPose.load(Path(task.pose_path))
        if task.pose_path
        else assembly.mounted_pose(task.params, task.asset_dir)
    )
    if not pose.report["passed"]:
        raise ValueError(f"mounted pose failed: {pose.report}")

    proprio = {
        "imu_ang_vel": ObservationTermCfg(
            func=mj.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "imu_acc": ObservationTermCfg(
            func=mdp.imu_acc, noise=Unoise(n_min=-0.05, n_max=0.05)
        ),
        # Horizontal velocity is unobservable in flight from 0.1 s of IMU and
        # encoders; legged RL (mjlab G1, legged_gym) feeds an estimated base
        # velocity, noised as in mjlab's G1 velocity task.
        "base_lin_vel": ObservationTermCfg(
            func=mj.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "gravity": ObservationTermCfg(
            func=mj.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        ),
        "joint_pos": ObservationTermCfg(
            func=mj.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=robot.BODY_JOINTS)
            },
        ),
        "joint_vel": ObservationTermCfg(
            func=mj.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=robot.BODY_JOINTS)
            },
        ),
        "actions": ObservationTermCfg(func=mj.last_action),
        "command": ObservationTermCfg(func=mdp.hop_command),
    }
    if not task.base_velocity:
        del proprio["base_lin_vel"]
    critic = {
        k: ObservationTermCfg(func=v.func, params=v.params, scale=v.scale)
        for k, v in proprio.items()
    }
    critic["privileged"] = ObservationTermCfg(func=mdp.privileged)
    observations = {
        "actor": ObservationGroupCfg(
            terms=proprio,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=task.history,
        ),
        "critic": ObservationGroupCfg(
            terms=critic, concatenate_terms=True, enable_corruption=False
        ),
    }

    stroke = task.params.stroke
    steer = {
        "hop": (mdp.landed_velocity, mdp.landed_yaw_rate, {}),
        "step": (mdp.step_velocity, mdp.step_yaw_rate, {}),
        "window": (mdp.window_velocity, mdp.window_yaw_rate, {"tau": task.steer_tau}),
        "centroidal": (mdp.step_velocity, mdp.step_centroidal_yaw_rate, {}),
    }
    if task.steer_timescale not in steer or task.height_guide not in (
        "gaussian",
        "potential",
    ):
        raise ValueError(f"unknown {task.steer_timescale=} or {task.height_guide=}")
    steer_vel, steer_yaw, steer_kw = steer[task.steer_timescale]
    rewards = {
        # Measured tip apex at each valid landing against the takeoff command,
        # plus a weak dense energy guide (DESIGN.md section 3).
        "apex": RewardTermCfg(func=mdp.landed_height, weight=4.0, params={"std": 0.03}),
        "energy": RewardTermCfg(
            func=mdp.track_height_energy, weight=1.0, params={"std": 0.08}
        )
        if task.height_guide == "gaussian"
        # Potential-based alternative; weight 8 makes correcting a 0.3 m error
        # worth about what the Gaussian guide pays over one hop.
        else RewardTermCfg(
            func=mdp.height_potential, weight=8.0, params={"std": 0.08, "gamma": 0.99}
        ),
        # Steering is scored like height: per hop cycle at the task tolerance
        # (0.15 m/s, 0.15 rad/s), plus a weak dense guide; as much per second.
        # steer_timescale="step" scores the same kernel on instantaneous
        # motion instead (the reward-timescale experiment).
        "com_vel": RewardTermCfg(
            func=steer_vel,
            weight=2.0,
            params={"std": 0.15} | steer_kw,
        ),
        "yaw_rate": RewardTermCfg(
            func=steer_yaw,
            weight=task.yaw_weight,
            params={"std": 0.15} | steer_kw,
        ),
        "com_vel_guide": RewardTermCfg(
            func=mdp.track_com_velocity, weight=0.5, params={"std": 0.5}
        ),
        "yaw_rate_guide": RewardTermCfg(
            func=mdp.track_yaw_rate, weight=0.5, params={"std": 0.5}
        ),
        "yaw_twist": RewardTermCfg(func=mdp.yaw_rate_error_l2, weight=-0.1),
        "pogo_upright": RewardTermCfg(
            func=mdp.upright, weight=1.0, params={"name": "pogo", "std": 0.1}
        ),
        "torso_upright": RewardTermCfg(
            func=mdp.upright, weight=0.5, params={"name": "robot", "std": 0.2}
        ),
        "feet": RewardTermCfg(
            func=mdp.feet_on_pedals, weight=1.0, params={"std": 0.03}
        ),
        "alive": RewardTermCfg(func=mj.is_alive, weight=1.0),
        "action_rate": RewardTermCfg(func=mj.action_rate_l2, weight=-0.02),
        "torque": RewardTermCfg(func=mj.joint_torques_l2, weight=-2e-5),
        "joint_acc": RewardTermCfg(func=mj.joint_acc_l2, weight=-2.5e-7),
        "joint_limits": RewardTermCfg(func=mj.joint_pos_limits, weight=-2.0),
        "bottom_out": RewardTermCfg(
            func=mdp.bottom_out, weight=-5.0, params={"stroke": stroke}
        ),
        "terminated": RewardTermCfg(func=mj.is_terminated, weight=-200.0),
    }
    terminations = {
        "timeout": TerminationTermCfg(func=mj.time_out, time_out=True),
        "pogo_tilt": TerminationTermCfg(
            func=mdp.tilted, params={"name": "pogo", "max_deg": 45.0}
        ),
        "torso_tilt": TerminationTermCfg(
            func=mdp.tilted, params={"name": "robot", "max_deg": 60.0}
        ),
        "feet_off": TerminationTermCfg(
            func=mdp.feet_off_pedals, params={"limit": 0.08}
        ),
        "robot_ground": TerminationTermCfg(
            func=mdp.illegal_contact, params={"sensor": "robot_ground"}
        ),
        "frame_ground": TerminationTermCfg(
            func=mdp.illegal_contact, params={"sensor": "frame_ground"}
        ),
        "grip_lost": TerminationTermCfg(func=mdp.grip_lost),
        "nonfinite": TerminationTermCfg(func=mdp.nonfinite),
        "terrain_missing": TerminationTermCfg(func=mdp.terrain_missing),
    }
    events = {
        "reset": EventTermCfg(
            func=mdp.reset_mixed,
            mode="reset",
            params={
                "airborne_prob": task.airborne_prob,
                "lift": (0.0, 0.2),
                "vz": (-1.0, 1.5),
                "vxy": 0.2,
                "yaw_rate": 0.3,
            },
        ),
    }
    if task.robust:
        pogo_joint = SceneEntityCfg("pogo", joint_names=("compression",))
        events |= {
            "spring_k": EventTermCfg(
                func=mdp.randomize_spring,
                mode="reset",
                params={"ranges": (0.8, 1.2)},
            ),
            "spring_c": EventTermCfg(
                func=dr.joint_damping,
                mode="reset",
                params={
                    "asset_cfg": pogo_joint,
                    "ranges": (0.5, 1.5),
                    "operation": "scale",
                },
            ),
            "seal_friction": EventTermCfg(
                func=dr.joint_friction,
                mode="reset",
                params={
                    "asset_cfg": pogo_joint,
                    "ranges": (0.5, 2.0),
                    "operation": "scale",
                },
            ),
            "pogo_mass": EventTermCfg(
                func=dr.pseudo_inertia,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "pogo", body_names=("pogo_frame", "pogo_plunger")
                    ),
                    "alpha_range": (-0.125, 0.255),  # mass x0.78-1.67
                },
            ),
            "contact_friction": EventTermCfg(
                func=mdp.randomize_contacts,
                mode="reset",
                params={
                    "tip_patch_radius": task.contacts.tip_patch_radius,
                    "grip_torsion": task.contacts.grip_torsion,
                },
            ),
            "robot_com": EventTermCfg(
                func=dr.body_com_offset,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                    "ranges": {0: (-0.03, 0.03), 1: (-0.02, 0.02), 2: (-0.03, 0.03)},
                },
            ),
            "robot_mass": EventTermCfg(
                func=dr.pseudo_inertia,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
                    "alpha_range": (-0.025, 0.025),
                },  # +-5 %,
            ),
            "encoder_bias": EventTermCfg(
                func=dr.encoder_bias,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "bias_range": (-0.015, 0.015),
                },
            ),
            "motor_gains": EventTermCfg(
                func=dr.pd_gains,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "kp_range": (0.9, 1.1),
                    "kd_range": (0.9, 1.1),
                },
            ),
            "push": EventTermCfg(
                func=mj.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(2.0, 5.0),
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "velocity_range": {
                        "x": (-0.3, 0.3),
                        "y": (-0.3, 0.3),
                        "yaw": (-0.4, 0.4),
                    },
                },
            ),
        }
    if task.eval_push > 0:
        events["eval_push"] = EventTermCfg(
            func=mj.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.5, 3.5),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": (-task.eval_push, task.eval_push),
                    "y": (-task.eval_push, task.eval_push),
                },
            },
        )
    metrics = {
        "hops": MetricsTermCfg(func=mdp.hops, reduce="sum"),
        "energy_height": MetricsTermCfg(func=mdp.energy_height, reduce="mean"),
        "tip_clearance_max": MetricsTermCfg(func=mdp.tip_clearance, reduce="max"),
        "compression_max": MetricsTermCfg(func=mdp.compression, reduce="max"),
        "grip_force_max": MetricsTermCfg(func=mdp.grip_force_max, reduce="max"),
        "grip_force_mean": MetricsTermCfg(func=mdp.grip_force_max, reduce="mean"),
        "grip_torque_max": MetricsTermCfg(func=mdp.grip_torque_max, reduce="max"),
        "grip_distance_max": MetricsTermCfg(func=mdp.grip_distance_max, reduce="max"),
        "grip_contact_fraction": MetricsTermCfg(
            func=mdp.grip_contact_fraction, reduce="mean"
        ),
        "finger_torque_max": MetricsTermCfg(func=mdp.finger_torque_max, reduce="max"),
        "apex_ceiling": MetricsTermCfg(func=mdp.apex_ceiling, reduce="mean"),
    }
    commands = {
        "hop": mdp.HopCommandCfg(
            resampling_time_range=(4.0, 8.0),
            ranges=mdp.HopRanges(
                apex=task.apex, vx=task.vx, vy=task.vy, yaw_rate=task.yaw_rate
            ),
            rel_in_place=task.in_place_frac,
            com_rest=pose.report["com_height_m"],
            apex_cap=task.apex_cap,
        )
    }
    return ManagerBasedRlEnvCfg(
        decimation=task.decimation,
        scene=_scene(task, pose),
        observations=observations,
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=robot.BODY_JOINTS,
                scale=robot.ACTION_SCALE,
                use_default_offset=True,
            )
        },
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        sim=SimulationCfg(
            nconmax=NCONMAX,
            njmax=NJMAX,
            mujoco=MujocoCfg(
                timestep=task.timestep,
                integrator="implicitfast",
                cone="elliptic",
                impratio=3.0,
            ),
        ),
        episode_length_s=task.episode_s,
        seed=task.seed,
    )
