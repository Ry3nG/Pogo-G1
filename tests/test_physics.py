import mujoco
import numpy as np
import pytest

pytest.importorskip("mjlab", reason="mjlab (train extra) not installed")

from pogo_g1 import assembly, env, pogo, robot  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    _, m, pose = assembly.build_scene()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d, pose


def test_small_pro_dimensions_and_full_compression_clearance():
    geo = pogo.load_geometry()
    assert geo["uncompressed_height_m"] == pytest.approx(1.2446, abs=1e-7)
    assert geo["handle_to_pedal_drawing_m"] == pytest.approx(0.762)
    assert geo["pedal_top_above_tip_bottom_m"] == pytest.approx(0.4826)
    assert geo["metal_bar_od_m"] == pytest.approx(0.022225)
    assert pogo.handle_radius() == pytest.approx(0.014576107105608307, abs=1e-14)
    m = pogo.build_spec(floor=True).compile()
    assert m.body_subtreemass[0] == pytest.approx(4.54)
    assert 2 * m.geom("outer_tube").size[0] > pogo.NOMINAL.bore
    d = mujoco.MjData(m)
    x = pogo.NOMINAL.stroke + 0.005
    d.qpos[2] -= x
    d.qpos[m.joint("compression").qposadr[0]] = x
    mujoco.mj_forward(m, d)
    floor = m.geom("floor").id
    for name in ("outer_tube", "left_pedal", "right_pedal", "handlebar"):
        assert mujoco.mj_geomDistance(m, d, floor, m.geom(name).id, 1, None) > 0.01


def test_only_attachments_are_the_hand_linkages(scene):
    m, _, pose = scene
    assert pose.report["passed"]
    assert m.neq == 10
    assert set(m.eq_type) == {int(mujoco.mjtEq.mjEQ_JOINT)}
    assert all("_distal_joint" in m.joint(j).name for j in m.eq_obj1id)
    assert np.count_nonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 2
    # 29 body motors + 6 per hand; every actuator drives a robot joint.
    assert m.nu == 41
    assert all(
        m.joint(int(j)).name.startswith("robot/") for j in m.actuator_trnid[:, 0]
    )
    for side in ("left", "right"):
        for i in range(1, 8):
            geom = m.geom(f"robot/{side}_foot{i}_collision")
            assert geom.contype[0] == 1 and geom.conaffinity[0] == 1


def test_mounted_grasp_is_contact_only_and_torque_limited(scene):
    m, d0, _ = scene
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, d0)
    handle = m.geom("pogo/handlebar").id
    contacts = [c for c in d.contact[: d.ncon] if handle in (c.geom1, c.geom2)]
    # Equilibrium compliance of the servo-loaded grasp, not tunnelling (bar r=14.6 mm).
    assert min(c.dist for c in contacts) > -0.0015
    for side in ("left", "right"):
        bodies = {
            m.body(m.geom_bodyid[g]).name
            for c in contacts
            for g in (c.geom1, c.geom2)
            if g != handle and f"robot/{side}_" in m.body(m.geom_bodyid[g]).name
        }
        assert len(bodies) >= 4
    fingers = [i for i in range(m.nu) if robot.is_finger(m.actuator(i).name)]
    assert len(fingers) == 12
    ncon, nefc = 0, 0
    foot_contacts = dict.fromkeys(("left", "right"), False)
    for _ in range(round(0.2 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        assert np.isfinite(d.qpos).all()
        limit = m.actuator_forcerange[fingers, 1] + 1e-9
        assert np.all(np.abs(d.actuator_force[fingers]) <= limit)
        ncon, nefc = max(ncon, d.ncon), max(nefc, d.nefc)
        for side in foot_contacts:
            pedal = m.geom(f"pogo/{side}_pedal").id
            foot_contacts[side] |= any(
                pedal in (c.geom1, c.geom2)
                and any(
                    f"robot/{side}_foot" in m.geom(g).name for g in (c.geom1, c.geom2)
                )
                for c in d.contact[: d.ncon]
            )
            # Same mdp.foot_error distance and existing 8 cm termination limit.
            normal = d.geom_xmat[pedal].reshape(3, 3)[:, 2]
            top = d.geom_xpos[pedal] + normal * (m.geom_size[pedal, 2] - 0.002)
            foot = d.site_xpos[m.site(f"robot/{side}_foot").id]
            assert np.linalg.norm(foot - top) < 0.08
    eq = d.efc_pos[d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    assert np.abs(eq).max() < 0.01
    assert ncon < 0.75 * env.NCONMAX and nefc < 0.75 * env.NJMAX
    for side in ("left", "right"):
        # Brief separation is valid; require contact during the window and
        # keep each foot within the task's pedal-distance limit throughout.
        assert foot_contacts[side]
        gap = (
            d.site_xpos[m.site(f"robot/{side}_grasp").id]
            - d.site_xpos[m.site(f"pogo/{side}_grip").id]
        )
        assert np.linalg.norm(gap) < 0.02
