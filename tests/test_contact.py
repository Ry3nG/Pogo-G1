"""Analytical contact fixtures; no robot, artwork, policy or GPU required."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from pogo_g1 import contact, pogo


def probe(mu=0.8, azimuth=0.0, slope=0.0, spin=False, mass=40, normal_load=None):
    """A loaded tip with constrained orientation, isolating sliding or spin."""
    s = mujoco.MjSpec()
    s.compiler.degree = False
    s.option.timestep = pogo.PHYSICS_TIMESTEP
    contact.configure_solver(s)
    angle = np.deg2rad(slope)
    normal = np.array([np.sin(angle), 0, np.cos(angle)])
    quat = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0])
    s.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.1],
        quat=quat,
        friction=[1, 0.02, 0.002],
    )
    b = s.worldbody.add_body(pos=normal * 0.0368)
    b.add_joint(name="normal", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=normal)
    if spin:
        b.add_joint(name="motion", type=mujoco.mjtJoint.mjJNT_HINGE, axis=normal)
    else:
        phi = np.deg2rad(azimuth)
        tangent = np.array(
            [np.cos(phi) * np.cos(angle), np.sin(phi), -np.cos(phi) * np.sin(angle)]
        )
        b.add_joint(name="motion", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=tangent)
    b.add_geom(
        name="tip",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.0368, 0, 0],
        mass=mass,
        friction=[1, 0.02, 0.002],
    )
    params = replace(contact.DEFAULT, ground_friction=mu)
    contact.tip_pair(s, "tip", "ground", params)
    m = s.compile()
    d = mujoco.MjData(m)
    if normal_load is not None:
        d.qfrc_applied[0] = mass * 9.81 * np.cos(angle) - normal_load
    for _ in range(round(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return m, d, params


@pytest.mark.parametrize("mu", [0.2, 0.8])
@pytest.mark.parametrize("azimuth", [0, 37, 90])
def test_slip_threshold_and_isotropy(mu, azimuth):
    m, d, _ = probe(mu, azimuth)
    assert d.ncon == 1
    assert d.contact[0].friction[:2] == pytest.approx([mu, mu])
    assert d.contact[0].dim == 6
    start = mujoco.MjData(m)
    mujoco.mj_copyData(start, m, d)
    for ratio in [0.8, 1.2]:
        mujoco.mj_copyData(d, m, start)
        d.qfrc_applied[1] = ratio * mu * 40 * 9.81
        for _ in range(round(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        if ratio < 1:
            assert abs(d.qpos[1]) < 0.002
        else:
            assert d.qpos[1] > 0.01


def test_finite_contact_patch_resists_spin_only_up_to_capacity():
    m, d, p = probe(spin=True)
    capacity = 2 / 3 * p.ground_friction * p.tip_patch_radius * 40 * 9.81
    start = mujoco.MjData(m)
    mujoco.mj_copyData(start, m, d)
    for ratio in [0.8, 1.2]:
        mujoco.mj_copyData(d, m, start)
        d.qfrc_applied[1] = ratio * capacity
        for _ in range(round(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        if ratio < 1:
            assert abs(d.qpos[1]) < 0.02
        else:
            assert d.qpos[1] > 0.1


@pytest.mark.parametrize("slope, slips", [(5, False), (20, True)])
def test_slope_slips_according_to_coulomb_limit(slope, slips):
    m, d, _ = probe(mu=0.2, slope=slope)
    for _ in range(round(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    if slips:
        assert d.qpos[1] > 0.05
    else:
        assert abs(d.qpos[1]) < 0.002


def test_contact_compliance_under_load():
    # Effective contact mass is the light plunger, not the whole rider. Testing
    # only a 40 kg single rigid body hides excessive soft-contact penetration.
    m, d, _ = probe(mass=0.9, normal_load=584)
    assert 0.0001 < -d.contact[0].dist < 0.003
    f = np.zeros(6)
    mujoco.mj_contactForce(m, d, 0, f)
    assert f[0] == pytest.approx(584, rel=0.01)
