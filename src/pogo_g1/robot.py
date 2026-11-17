"""G1 with BrainCo Revo2 hands, following Unitree's official G1 integration.

The 29 body joints use the mjlab G1 actuators. Each hand has six torque-limited
position servos (URDF effort limits). A distal finger joint follows its proximal
joint through the hand's linkage; the URDF mimic ratio is modelled as a joint
equality. The hands hold the pogo only through contact and friction.
"""

import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from pogo_g1 import contact

# Fetched by scripts/fetch_assets.sh (relative to the repository root).
HAND_URDF = Path(
    "data/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf"
)
SIDES = ("left", "right")
_FINGER = re.compile(r"_(thumb|index|middle|ring|pinky)_")
_TIP = re.compile(r"_tip(_Link)?$")
BODY_JOINTS = (r"^(?!.*_(thumb|index|middle|ring|pinky)_).*",)
ACTIVE = ("thumb_metacarpal", "thumb_proximal") + tuple(
    f"{f}_proximal" for f in ("index", "middle", "ring", "pinky")
)
# Power grasp: thumb fully in opposition (URDF range 0-1.52 rad), all fingers
# closing. The handle starts at GRASP_START (wrist_yaw_link frame) and the grasp
# is wherever it settles (settled_grasp). The start was chosen by pull-out tests
# (scripts/grip_capacity.py).
THUMB_OPPOSITION = 1.5
GRASP_START = {"left": (0.14, -0.02, 0.0), "right": (0.14, 0.02, 0.0)}
FRICTION = (1.0, 0.005, 0.0001)
# Finger links weigh ~10 g; the default contact time constant (20 ms) lets a
# 2 N m servo push them centimetres into the handle. 4 ms = 2 x physics step.
SOLREF = (0.004, 1.0)
# Reflected rotor inertia of the geared finger motors [kg m^2]. Without it the
# ~1e-7 kg m^2 phalanges make the linkage constraint ill-conditioned.
ARMATURE = 0.001
MAX_HULL_VERT = 64


def _hand_dir() -> Path:
    return HAND_URDF.resolve().parent


def is_finger(name: str) -> bool:
    return _FINGER.search(name) is not None


@lru_cache(maxsize=1)
def _urdf() -> tuple[str, tuple, dict]:
    text = HAND_URDF.read_text().replace(
        'filename="meshes/', f'filename="{_hand_dir()}/meshes/'
    )
    mimic = tuple(
        (j, leader, float(ratio))
        for j, leader, ratio in re.findall(
            r'<joint name="(\w+)" type="revolute">(?:(?!</joint>).)*?'
            r'<mimic joint="(\w+)" multiplier="([\d.]+)"',
            text,
            re.S,
        )
    )
    effort = {
        j: float(e)
        for j, e in re.findall(
            r'<joint name="(\w+)" type="revolute">(?:(?!</joint>).)*?effort="([\d.]+)"',
            text,
            re.S,
        )
    }
    return text, mimic, effort


@lru_cache(maxsize=1)
def _hand_source() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_string(_urdf()[0])
    spec.compiler.fusestatic = False
    return spec.compile()


def add_linkage(spec: mujoco.MjSpec, side: str, prefix: str = ""):
    for joint, leader, ratio in _urdf()[1]:
        if joint.startswith(side):
            data = np.zeros(11)
            data[1] = ratio
            spec.add_equality(
                name=f"{prefix}{joint}_linkage",
                type=mujoco.mjtEq.mjEQ_JOINT,
                name1=f"{prefix}{joint}",
                name2=f"{prefix}{leader}",
                data=data,
                solref=list(SOLREF),
                solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
            )


def _copy_body(m, bid, parent, meshes):
    """Copy one URDF body (compiled frames and inertia) and its subtree.

    The fingertip marker links are skipped: they have no size and a fake
    fixed-angle joint.
    """
    name = m.body(bid).name
    body = parent.add_body(
        name=name,
        pos=m.body_pos[bid].copy(),
        quat=m.body_quat[bid].copy(),
        mass=m.body_mass[bid],
        ipos=m.body_ipos[bid].copy(),
        iquat=m.body_iquat[bid].copy(),
        inertia=m.body_inertia[bid].copy(),
        explicitinertial=True,
    )
    for j in range(m.body_jntadr[bid], m.body_jntadr[bid] + m.body_jntnum[bid]):
        body.add_joint(
            name=m.joint(j).name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=m.jnt_axis[j].copy(),
            range=m.jnt_range[j].copy(),
            limited=True,
            armature=ARMATURE,
            damping=0.0,
        )
    for i, g in enumerate(
        range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid])
    ):
        mesh = m.mesh(m.geom_dataid[g]).name
        meshes.add(mesh)
        # URDF geometry origins are all identity; the mesh is re-centred on compile.
        if m.geom_contype[g] == 0:
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh,
                contype=0,
                conaffinity=0,
                group=2,
                density=0,
                rgba=[0.85, 0.85, 0.85, 1],
            )
        else:
            body.add_geom(
                name=f"{name}_{i}_contact",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh,
                group=3,
                density=0,
                contype=2,
                conaffinity=1,
                condim=4,
                friction=list(FRICTION),
                solref=list(SOLREF),
            )
    for child in range(m.nbody):
        if m.body_parentid[child] == bid and not _TIP.search(m.body(child).name):
            _copy_body(m, child, body, meshes)


def _graft_hand(spec: mujoco.MjSpec, side: str, meshes: set):
    m = _hand_source()
    dst = spec.body(f"{side}_wrist_yaw_link")
    for geom in list(dst.geoms):
        if (
            geom.meshname == f"{side}_rubber_hand"
            or geom.name == f"{side}_hand_collision"
        ):
            spec.delete(geom)
    w = m.body(f"{side}_wrist_yaw_link").id
    dst.mass, dst.ipos, dst.iquat = (
        m.body_mass[w],
        m.body_ipos[w].copy(),
        m.body_iquat[w].copy(),
    )
    dst.inertia, dst.explicitinertial = m.body_inertia[w].copy(), True
    for child in range(m.nbody):
        if m.body_parentid[child] == w:
            _copy_body(m, child, dst, meshes)
    add_linkage(spec, side)
    g = settled_grasp(side)
    dst.add_site(name=f"{side}_grasp", pos=g["pos"], quat=g["quat"], size=[0.01] * 3)


def _flexion_axis(side: str) -> np.ndarray:
    """Mean flexion axis of the four fingers in the wrist_yaw_link frame."""
    m = _hand_source()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    axes = []
    for f in ("index", "middle", "ring", "pinky"):
        j = m.joint(f"{side}_{f}_proximal_joint")
        axes.append(d.xmat[m.jnt_bodyid[j.id]].reshape(3, 3) @ m.jnt_axis[j.id])
    z = d.xmat[m.body(f"{side}_wrist_yaw_link").id].reshape(3, 3).T @ np.mean(axes, 0)
    return z / np.linalg.norm(z)


def get_spec() -> mujoco.MjSpec:
    spec = g1.get_spec()
    meshes: set[str] = set()
    for side in SIDES:
        _graft_hand(spec, side, meshes)
    for name in sorted(meshes):
        spec.add_mesh(
            name=name,
            file=str(_hand_dir() / "meshes" / f"{name}.STL"),
            maxhullvert=MAX_HULL_VERT,
        )
    return spec


def mass_kg() -> float:
    return float(get_spec().compile().body_subtreemass[1])


def finger_target(name: str) -> float:
    """Closing command: thumb in opposition, every other active joint fully flexed."""
    key = name.split("/")[-1].split("_", 1)[1].removesuffix("_joint")
    if key == "thumb_metacarpal":
        return THUMB_OPPOSITION
    return 1.4 if key == "thumb_proximal" else 1.5


def get_robot_cfg() -> EntityCfg:
    cfg = g1.get_g1_robot_cfg()
    cfg.spec_fn = get_spec
    effort = _urdf()[2]
    hand = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(rf".*_{key}_joint",),
            stiffness=20.0,
            damping=0.2,
            effort_limit=effort[f"left_{key}_joint"],
            armature=ARMATURE,
        )
        for key in ("thumb_metacarpal", "thumb_proximal")
    ) + (
        BuiltinPositionActuatorCfg(
            target_names_expr=(r".*_(index|middle|ring|pinky)_proximal_joint",),
            stiffness=20.0,
            damping=0.2,
            effort_limit=effort["left_index_proximal_joint"],
            armature=ARMATURE,
        ),
    )
    cfg.articulation = replace(
        cfg.articulation, actuators=cfg.articulation.actuators + hand
    )
    cfg.collisions += (
        CollisionCfg(
            geom_names_expr=(".*_contact",),
            contype=2,
            conaffinity=1,
            condim=4,
            priority=0,
            disable_other_geoms=False,
            friction=FRICTION,
        ),
    )
    return cfg


def grasp_fixture(side: str, radius: float) -> mujoco.MjSpec:
    """One hand fixed in space and a free 0.2 kg bar of the handle radius; no gravity.

    The bar starts at GRASP_START along the finger flexion axis. The fixture is only
    used to find the grasp before an episode (and to measure grip capacity); it is
    not part of the training scene.
    """
    m = _hand_source()
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.gravity = [0, 0, 0]
    wrist = spec.worldbody.add_body(name=f"{side}_wrist_yaw_link")
    meshes: set[str] = set()
    w = m.body(f"{side}_wrist_yaw_link").id
    for child in range(m.nbody):
        if m.body_parentid[child] == w:
            _copy_body(m, child, wrist, meshes)
    for name in sorted(meshes):
        spec.add_mesh(
            name=name,
            file=str(_hand_dir() / "meshes" / f"{name}.STL"),
            maxhullvert=MAX_HULL_VERT,
        )
    add_linkage(spec, side)
    effort = _urdf()[2]
    for key in ACTIVE:
        a = spec.add_actuator(
            name=f"{side}_{key}",
            target=f"{side}_{key}_joint",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
        )
        a.set_to_position(kp=20, kv=0.2)
        tau = effort[f"{side}_{key}_joint"]
        a.forcelimited, a.forcerange = True, [-tau, tau]
    axis = _flexion_axis(side)
    bar = spec.worldbody.add_body(name="bar", pos=GRASP_START[side])
    bar.add_freejoint()
    bar.add_geom(
        name="bar",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=np.r_[-0.09 * axis, 0.09 * axis],
        size=[radius, 0, 0],
        mass=0.2,
        contype=1,
        conaffinity=2,
        condim=4,
        friction=list(FRICTION),
        solref=list(SOLREF),
    )
    contact.configure_solver(spec)
    contact.hand_pairs(spec, "bar", hand_prefix=f"{side}_")
    return spec


def close_on_bar(m: mujoco.MjModel, d: mujoco.MjData, side: str):
    """Close every servo under its torque limit with the bar held, then release the
    bar and let the grasp settle into equilibrium."""
    target = np.array([finger_target(m.actuator(a).name) for a in range(m.nu)])
    q0 = np.r_[GRASP_START[side], 1.0, 0, 0, 0]
    for i in range(1000):
        d.ctrl[:] = target * min(1.0, i / 300)
        if i < 750:
            d.qpos[-7:], d.qvel[-6:] = q0, 0
        mujoco.mj_step(m, d)


@lru_cache(maxsize=2)
def settled_grasp(side: str) -> dict:
    """Finger angles and the bar pose (wrist frame) of the settled power grasp."""
    from pogo_g1 import pogo

    m = grasp_fixture(side, pogo.handle_radius()).compile()
    d = mujoco.MjData(m)
    close_on_bar(m, d, side)
    bar = m.body("bar").id
    touching = {
        m.body(m.geom_bodyid[g]).name
        for c in d.contact[: d.ncon]
        for g in (c.geom1, c.geom2)
        if m.geom_bodyid[g] != bar and bar in m.geom_bodyid[[c.geom1, c.geom2]]
    }
    moved = np.linalg.norm(d.xpos[bar] - GRASP_START[side])
    if len(touching) < 4 or moved > 0.03 or not np.isfinite(d.qpos).all():
        raise RuntimeError(
            f"{side} grasp: touches {sorted(touching)}, bar moved {moved}"
        )
    axis = d.xmat[bar].reshape(3, 3) @ _flexion_axis(side)
    x = np.array([1.0, 0, 0]) - axis[0] * axis
    x /= np.linalg.norm(x)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, np.cross(axis, x), axis]).ravel())
    fingers = {
        m.joint(j).name: float(d.qpos[m.jnt_qposadr[j]])
        for j in range(m.njnt)
        if is_finger(m.joint(j).name)
    }
    return {"pos": d.xpos[bar].tolist(), "quat": quat.tolist(), "fingers": fingers}


def contact_finger_pose() -> dict:
    """Finger joint angles of the settled grasp, both hands."""
    return settled_grasp("left")["fingers"] | settled_grasp("right")["fingers"]


ACTION_SCALE = g1.G1_ACTION_SCALE
