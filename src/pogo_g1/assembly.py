"""Robot + pogo assembly: mounted-pose IK and the CPU scene.

The mounted pose is kinematic (feet on pedals, hands around the handles, system
COM above the tip, no penetration). It is the evaluation start state; dynamic
balance is the policy's job. Hands and feet touch the pogo only through contact.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import Entity
from scipy.optimize import least_squares

from pogo_g1 import contact, pogo, robot
from pogo_g1.symmetry import joint_mirror

CACHE = Path("data/mounted")
SYMMETRY_WEIGHT = 50.0
COM_WEIGHT = 100.0

# Nominal "ready" crouch that regularises the IK (radians).
READY = {
    "hip_pitch": -0.45,
    "knee": 0.95,
    "ankle_pitch": -0.45,
    "shoulder_pitch": -0.2,
    "elbow": 0.6,
}


@dataclass
class MountedPose:
    robot_pos: list
    robot_quat: list
    joints: dict
    pogo_pos: list
    pogo_quat: list
    report: dict = field(default_factory=dict)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(Path(path).read_text()))


def _base_spec(params, asset_dir, contacts=contact.DEFAULT):
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1]
    )
    spec.attach(
        pogo.build_spec(params, asset_dir),
        prefix="pogo/",
        frame=spec.worldbody.add_frame(),
    )
    robot_spec = Entity(robot.get_robot_cfg()).spec
    for key in list(robot_spec.keys):
        robot_spec.delete(key)
    spec.attach(robot_spec, prefix="robot/", frame=spec.worldbody.add_frame())
    contact.assembly_pairs(spec, contacts)
    return spec


def _hinges(model):
    ids = [
        j
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        and model.joint(j).name.startswith("robot/")
        and not robot.is_finger(model.joint(j).name)
    ]
    return np.array(ids), model.jnt_qposadr[ids]


def solve_mounted(
    params: pogo.PogoParams = pogo.NOMINAL, asset_dir: Path = pogo.DEFAULT_ASSET
) -> MountedPose:
    spec = _base_spec(params, asset_dir)
    model = spec.compile()
    data = mujoco.MjData(model)
    fingers = robot.contact_finger_pose()
    for n, v in fingers.items():
        data.qpos[model.joint(f"robot/{n}").qposadr[0]] = v
    geo = pogo.load_geometry(asset_dir)
    rest = geo["pedal_top_above_tip_bottom_m"]
    # Static pogo pose: upright, plunger at the extension stop, tip on the ground.
    pogo_q = int(model.joint("pogo/pogo_root").qposadr[0])
    data.qpos[pogo_q : pogo_q + 7] = [0, 0, rest, 1, 0, 0, 0]
    root = int(model.joint("robot/floating_base_joint").qposadr[0])
    jids, qadr = _hinges(model)
    names = [model.joint(j).name.removeprefix("robot/") for j in jids]
    mperm, msign = joint_mirror(model, names)
    ready = np.zeros(len(names))
    for i, n in enumerate(names):
        for key, v in READY.items():
            if key in n:
                ready[i] = v
    data.qpos[qadr] = ready
    data.qpos[root : root + 7] = [-0.05, 0, rest + 0.62, 1, 0, 0, 0]

    sites = {
        f: model.site(f"robot/{f}").id
        for f in ["left_foot", "right_foot", "left_grasp", "right_grasp"]
    }
    grips = {s: model.site(f"pogo/{s}_grip").id for s in ["left", "right"]}
    tip = model.geom("pogo/tip").id
    pedal_top = rest
    targets = {}
    for side in ["left", "right"]:
        c = np.asarray(geo["pedal_centres"][side])
        # G1's foot site sits 2 mm below the sole capsules.
        targets[f"{side}_foot"] = np.array([c[0], c[1], pedal_top - 0.002])

    geoms = {model.geom(i).name: i for i in range(model.ngeom)}
    robot_coll = [
        i
        for n, i in geoms.items()
        if n.startswith("robot/") and n.endswith("_collision") and "foot" not in n
    ]
    pogo_coll = [geoms[f"pogo/{n}"] for n in ["outer_tube", "handlebar"]]
    pairs = [(a, b) for a in robot_coll for b in pogo_coll]
    # The hands must also clear the body (the arm links they are mounted on aside).
    hands = [
        i for n, i in geoms.items() if n.startswith("robot/") and n.endswith("_contact")
    ]
    body = [
        i
        for i in robot_coll
        if not any(k in model.geom(i).name for k in ("shoulder", "elbow", "wrist"))
    ]
    pairs += [(a, b) for a in hands for b in body]
    total_mass = model.body_subtreemass[0]

    def unpack(x):
        q = data.qpos.copy()
        q[root : root + 3] = x[:3]
        rot = np.empty(4)
        mujoco.mju_axisAngle2Quat(rot, [0, 1, 0], x[3])
        q[root + 3 : root + 7] = rot
        q[qadr] = x[4:]
        return q

    def residual(x):
        data.qpos[:] = unpack(x)
        mujoco.mj_forward(model, data)
        r = []
        for side in ["left", "right"]:
            f = sites[f"{side}_foot"]
            r.append(100 * (data.site_xpos[f] - targets[f"{side}_foot"]))
            r.append(10 * (data.site_xmat[f].reshape(3, 3) - np.eye(3)).ravel())
            g = sites[f"{side}_grasp"]
            r.append(100 * (data.site_xpos[g] - data.site_xpos[grips[side]]))
            # Grasp axis parallel to the handlebar (world y), either direction.
            z = data.site_xmat[g].reshape(3, 3)[:, 2]
            r.append(30 * z[[0, 2]])
        com = data.subtree_com[0]
        r.append(COM_WEIGHT * (com[:2] - data.geom_xpos[tip, :2]))
        r.append(
            [
                100
                * max(0.0, 0.01 - mujoco.mj_geomDistance(model, data, a, b, 0.1, None))
                for a, b in pairs
            ]
        )
        r.append(0.3 * (x[4:] - ready))
        # Left-right symmetric posture (exact mirror symmetry of the start state).
        r.append(SYMMETRY_WEIGHT * (x[4:] - msign * x[4:][mperm]))
        return np.concatenate([np.ravel(v) for v in r])

    lo = np.r_[[-0.4, -0.05, rest + 0.4, -0.2], model.jnt_range[jids, 0] + 0.05]
    hi = np.r_[[0.3, 0.05, rest + 0.9, 0.6], model.jnt_range[jids, 1] - 0.05]
    x0 = np.r_[data.qpos[root : root + 3], 0.1, ready]
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    sol = least_squares(residual, x0, bounds=(lo, hi), max_nfev=2000, xtol=1e-10)
    residual(sol.x)

    err = {
        k: float(np.linalg.norm(data.site_xpos[sites[k]] - targets[k]))
        for k in ["left_foot", "right_foot"]
    }
    for side in ["left", "right"]:
        err[f"{side}_grasp"] = float(
            np.linalg.norm(
                data.site_xpos[sites[f"{side}_grasp"]] - data.site_xpos[grips[side]]
            )
        )
    clearance = min(
        mujoco.mj_geomDistance(model, data, a, b, 0.1, None) for a, b in pairs
    )
    q = data.qpos[qadr]
    report = {
        "ik_cost": float(sol.cost),
        "max_asymmetry_rad": float(np.abs(q - msign * q[mperm]).max()),
        "site_errors_m": err,
        "min_robot_pogo_clearance_m": float(clearance),
        "com_minus_tip_xy_m": (
            data.subtree_com[0, :2] - data.geom_xpos[tip, :2]
        ).tolist(),
        "com_height_m": float(data.subtree_com[0, 2]),
        "torso_pitch_rad": float(sol.x[3]),
        "robot_mass_kg": float(model.body_subtreemass[model.body("robot/pelvis").id]),
        "pogo_mass_kg": params.mass,
        "total_mass_kg": float(total_mass),
        "params": asdict(params),
    }
    report["passed"] = bool(
        max(err.values()) < 0.003
        and clearance > -0.002
        and np.linalg.norm(report["com_minus_tip_xy_m"]) < 0.01
        and report["max_asymmetry_rad"] < 0.01
    )
    return MountedPose(
        robot_pos=data.qpos[root : root + 3].tolist(),
        robot_quat=data.qpos[root + 3 : root + 7].tolist(),
        joints={n: float(v) for n, v in zip(names, data.qpos[qadr], strict=True)}
        | fingers,
        pogo_pos=[0.0, 0.0, rest],
        pogo_quat=[1.0, 0.0, 0.0, 0.0],
        report=report,
    )


def mounted_pose(
    params: pogo.PogoParams = pogo.NOMINAL, asset_dir: Path = pogo.DEFAULT_ASSET
) -> MountedPose:
    """Cached on every input that changes the solution (masses move the COM target)."""
    key = hashlib.sha256(
        json.dumps(
            [
                asdict(params),
                str(asset_dir),
                pogo.load_geometry(asset_dir),
                READY,
                SYMMETRY_WEIGHT,
                COM_WEIGHT,
                [robot.settled_grasp(side) for side in robot.SIDES],
                "revo2-compound-inertia-v3",
            ]
        ).encode()
    ).hexdigest()[:12]
    path = CACHE / f"{key}.json"
    if path.exists():
        return MountedPose.load(path)
    pose = solve_mounted(params, asset_dir)
    pose.save(path)
    return pose


def build_scene(
    params: pogo.PogoParams = pogo.NOMINAL,
    asset_dir: Path = pogo.DEFAULT_ASSET,
    pose: MountedPose | None = None,
    timestep: float = pogo.PHYSICS_TIMESTEP,
    contacts: contact.ContactParams = contact.DEFAULT,
):
    """CPU MuJoCo scene (checks, evaluation replay, rendering)."""
    pose = pose or mounted_pose(params, asset_dir)
    spec = _base_spec(params, asset_dir, contacts)
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    model = spec.compile()
    q = model.qpos0.copy()
    pq = int(model.joint("pogo/pogo_root").qposadr[0])
    rq = int(model.joint("robot/floating_base_joint").qposadr[0])
    q[pq : pq + 7] = pose.pogo_pos + pose.pogo_quat
    q[pq + 7] = 0.0
    q[rq : rq + 3] = pose.robot_pos
    q[rq + 3 : rq + 7] = pose.robot_quat
    for n, v in pose.joints.items():
        q[model.joint(f"robot/{n}").qposadr[0]] = v
    ctrl = np.zeros(model.nu)
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        ctrl[a] = q[model.jnt_qposadr[j]]
        if robot.is_finger(model.joint(j).name):
            ctrl[a] = robot.finger_target(model.joint(j).name)
    spec.add_key(name="mounted", qpos=q, ctrl=ctrl)
    return spec, spec.compile(), pose
