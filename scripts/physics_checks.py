"""Physics checks that gate training (DESIGN.md section 2). Writes a JSON report."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from pogo_g1 import assembly, pogo, robot
from pogo_g1.env import NCONMAX, NJMAX

G = 9.81


def spring_law(params):
    spec = pogo.build_spec(params, asset_dir=pogo.DEFAULT_ASSET)
    spec.option.gravity = [0, 0, 0]
    m = spec.compile()
    d = mujoco.MjData(m)
    j = m.joint("compression")
    errs, gas_errors = [], []
    for x in np.linspace(0.0, params.stroke, 4097):
        d.qpos[:] = m.qpos0
        d.qpos[j.qposadr[0]] = x
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        measured = (
            -d.qfrc_spring[j.dofadr[0]]
            if hasattr(d, "qfrc_spring")
            else -d.qfrc_passive[j.dofadr[0]]
        )
        errs.append(abs(measured - params.spring_force(x)))
        gas_errors.append(abs(measured / params.gas_force(x) - 1))
    return {
        "max_abs_error_n": float(max(errs)),
        "max_gas_force_relative_error": float(max(gas_errors)),
        "compression_ratio": params.volume
        / (params.volume - params.area * params.stroke),
        "force_n_at_0_75_100_percent": [
            float(params.gas_force(x * params.stroke)) for x in (0, 0.75, 1)
        ],
        "gas_energy_j_at_75_100_percent": [
            float(params.gas_energy(x * params.stroke)) for x in (0.75, 1)
        ],
        "min_tangent_n_per_m_with_5mm_overtravel": float(
            np.gradient(
                params.spring_force(np.linspace(-0.005, params.stroke + 0.005, 4097)),
                (params.stroke + 0.01) / 4096,
            ).min()
        ),
        "passed": bool(max(errs) < 1e-6 and max(gas_errors) < 0.02),
    }


def drop(params, payload, height, timestep, seconds=4.0):
    """Frame constrained to vertical translation (diagnostic fixture), lumped payload.

    The slide coordinate is the height of the tip above the ground at full extension.
    """
    spec = pogo.build_spec(params, asset_dir=pogo.DEFAULT_ASSET, floor=True)
    frame = spec.body("pogo_frame")
    spec.delete(frame.joints[0])
    frame.add_joint(name="z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    frame.add_geom(
        name="payload",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0, 0],
        mass=payload,
        contype=0,
        conaffinity=0,
    )
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[0] = height
    tip = m.geom("tip").id
    apexes, stances, energy = [], [], []
    max_tip_penetration = 0.0
    max_compression = 0.0
    touching, t_touch, prev_v = False, 0.0, 0.0
    airborne_after_contact = False
    for _ in range(int(seconds / timestep)):
        mujoco.mj_step(m, d)
        max_compression = max(
            max_compression, float(d.qpos[m.joint("compression").qposadr[0]])
        )
        e = d.energy.sum()
        energy.append(e)
        contact = any(tip in (c.geom1, c.geom2) for c in d.contact[: d.ncon])
        for c in d.contact[: d.ncon]:
            if tip in (c.geom1, c.geom2):
                max_tip_penetration = max(max_tip_penetration, -float(c.dist))
        if contact and not touching:
            t_touch = d.time
        if touching and not contact:
            stances.append(d.time - t_touch)
            airborne_after_contact = True
        touching = contact
        v = d.qvel[0]
        if airborne_after_contact and prev_v > 0 >= v and not contact:
            apexes.append(float(d.geom_xpos[tip, 2] - m.geom_size[tip, 0]))
            airborne_after_contact = False
        prev_v = v
    energy = np.asarray(energy)
    rises = np.diff(energy)
    return {
        "drop_height_m": height,
        "max_tip_penetration_m": max_tip_penetration,
        "max_compression_m": max_compression,
        "tip_apex_m": apexes[:6],
        "stance_s": [float(s) for s in stances[:6]],
        "max_energy_increase_j": float(max(0.0, rises.max())),
        "energy_start_end_j": [float(energy[0]), float(energy[-1])],
        "monotone_decay": bool(all(np.diff(apexes) < 1e-4))
        if len(apexes) > 1
        else None,
    }


def mounted_rollout(pose, timestep, motors, seconds=3.0):
    """Mounted start with the body motors off or holding the pose (no policy)."""
    spec, m, _ = assembly.build_scene(pose=pose, timestep=timestep)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    fingers = [i for i in range(m.nu) if robot.is_finger(m.actuator(i).name)]
    if not motors:  # body motors off; fingers keep squeezing
        body = [i for i in range(m.nu) if i not in fingers]
        m.actuator_gainprm[body] = 0
        m.actuator_biasprm[body] = 0
    tip = m.geom("pogo/tip").id
    pelvis = m.body("robot/pelvis").id
    frame = m.body("pogo/pogo_frame").id
    hands = [m.site(f"robot/{s}_grasp").id for s in ("left", "right")]
    handles = [m.site(f"pogo/{s}_grip").id for s in ("left", "right")]
    mujoco.mj_forward(m, d)
    handle = m.geom("pogo/handlebar").id
    contacts = [c for c in d.contact[: d.ncon] if handle in (c.geom1, c.geom2)]
    initial = {
        "handle_contacts": len(contacts),
        "min_distance_m": float(min([c.dist for c in contacts], default=0)),
        "contact_bodies_per_hand": {
            side: len(
                {
                    int(m.geom_bodyid[g])
                    for c in contacts
                    for g in (c.geom1, c.geom2)
                    if g != handle and f"robot/{side}_" in m.body(m.geom_bodyid[g]).name
                }
            )
            for side in robot.SIDES
        },
    }
    finger_tau = 0.0
    ncon, nefc = d.ncon, d.nefc
    max_linkage = 0.0
    finite = True
    max_clear, t_fail = 0.0, None
    for _ in range(int(seconds / timestep)):
        mujoco.mj_step(m, d)
        ncon, nefc = max(ncon, d.ncon), max(nefc, d.nefc)
        finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        eq = np.abs(d.efc_pos[d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY])
        max_linkage = max(max_linkage, float(eq.max(initial=0)))
        distance = np.linalg.norm(
            d.site_xpos[hands] - d.site_xpos[handles], axis=-1
        ).max()
        if fingers:
            finger_tau = max(finger_tau, float(np.abs(d.actuator_force[fingers]).max()))
        tilt = np.degrees(np.arccos(np.clip(d.xmat[frame].reshape(3, 3)[2, 2], -1, 1)))
        if tilt > 45 or d.xpos[pelvis, 2] < 0.6 or distance > 0.05:
            t_fail = float(d.time)
            break
        max_clear = max(max_clear, d.geom_xpos[tip, 2] - m.geom_size[tip, 0])
    eq = np.abs(d.efc_pos[d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY])
    return {
        "motors": motors,
        "time_to_failure_s": t_fail,
        "max_tip_clearance_before_failure_m": float(max_clear),
        "max_linkage_residual_rad": float(eq.max()) if len(eq) else None,
        "peak_linkage_residual_rad": max_linkage,
        "finite": finite,
        "max_ncon": ncon,
        "max_nefc": nefc,
        "ncon_capacity": NCONMAX,
        "nefc_capacity": NJMAX,
        "buffer_headroom_passed": ncon < 0.75 * NCONMAX and nefc < 0.75 * NJMAX,
        "equality_count": m.neq,
        "actuator_count": m.nu,
        "initial": initial,
        "max_finger_torque_nm": finger_tau,
        "final_grip_distance_m": float(distance),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/physics_checks.json"))
    args = ap.parse_args()
    params = pogo.PogoParams()
    pose = assembly.mounted_pose(params)
    robot_mass = pose.report["robot_mass_kg"]
    report = {"params": asdict(params), "mounted_pose": pose.report}
    report["spring_law"] = spring_law(params)
    # Drops cover the full stroke (1.5 m bottoms out) and the pressure extremes.
    # The production 2 ms step must agree with 1 ms (half the timestep).
    report["drop"] = {}
    report["timestep_convergence"] = {}
    cases = {str(h): (params, h) for h in (0.15, 0.5, 1.0, 1.5)}
    cases["80kPa_n1.0_1m"] = (replace(params, pressure=80000, exponent=1.0), 1.0)
    cases["180kPa_n1.4_1m"] = (replace(params, pressure=180000, exponent=1.4), 1.0)
    for name, (case_params, height) in cases.items():
        runs = {
            str(dt): drop(case_params, robot_mass, height, dt) for dt in (0.002, 0.001)
        }
        report["drop"][name] = runs
        baseline, reference = runs["0.002"], runs["0.001"]
        report["timestep_convergence"][name] = {
            "second_apex_relative_error": abs(
                baseline["tip_apex_m"][1] / reference["tip_apex_m"][1] - 1
            ),
            "first_stance_relative_error": abs(
                baseline["stance_s"][0] / reference["stance_s"][0] - 1
            ),
        }
    report["mounted"] = {
        f"{'motors_on' if on else 'motors_off'}_dt_{dt}": mounted_rollout(pose, dt, on)
        for on in [False, True]
        for dt in [0.002, 0.001]
    }
    report["physics_version"] = pogo.PHYSICS_VERSION
    report["passed"] = bool(
        pose.report["passed"]
        and report["spring_law"]["passed"]
        and all(
            v["monotone_decay"]
            # Tunnelling guards only: compliance is judged by the dt convergence
            # below, and bottoming out on the physical stop is allowed.
            and v["max_tip_penetration_m"] < 0.01
            and v["max_compression_m"] < params.stroke + 0.005
            for runs in report["drop"].values()
            for v in runs.values()
        )
        and max(
            e for row in report["timestep_convergence"].values() for e in row.values()
        )
        < 0.03
        and all(
            v["finite"]
            and v["buffer_headroom_passed"]
            and v["initial"]["min_distance_m"] > -0.0015
            and min(v["initial"]["contact_bodies_per_hand"].values()) >= 4
            for v in report["mounted"].values()
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "params"}, indent=1))
    if not report["passed"]:
        raise SystemExit("Physics acceptance failed; see the saved report")


if __name__ == "__main__":
    main()
