"""Pogo model tests that need only MuJoCo (they run in CPU CI)."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from pogo_g1 import pogo


@pytest.mark.parametrize("exponent", [1.0, 1.3, 1.4])
@pytest.mark.parametrize("pressure", [80000, 128000, 180000])
@pytest.mark.parametrize("ratio", [None, 4.25, 4.77])
def test_native_spring_against_gas_law(exponent, pressure, ratio):
    p = pogo.PogoParams(exponent=exponent, pressure=pressure)
    if ratio is not None:
        p = replace(p, volume=p.area * p.stroke / (1 - 1 / ratio))
    spec = mujoco.MjSpec()
    spec.option.gravity = [0, 0, 0]
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    body = spec.worldbody.add_body()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=1)
    body.add_joint(
        name="compression",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
    )
    p.add_spring(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    energy_zero = d.energy[0]
    for x in np.linspace(0, p.stroke + 0.01, 97):
        d.qpos[0] = x
        mujoco.mj_forward(m, d)
        assert -d.qfrc_spring[0] == pytest.approx(p.gas_force(x), rel=0.0005)
        assert d.energy[0] - energy_zero == pytest.approx(p.spring_energy(x), abs=1e-8)
        assert p.spring_energy(x) == pytest.approx(
            p.gas_energy(x), rel=0.0005, abs=1e-8
        )
    x = np.linspace(-0.005, p.stroke + 0.01, 1001)
    assert np.all(np.diff(p.spring_force(x)) > 0)
    assert p.spring_force(0) == pytest.approx(p.preload)
    assert m.nu == 0 and m.neq == 0 and m.nq == 1
    assert np.all(p.gas_terms[1] > 0)


def test_isolated_spring_conserves_energy():
    p = pogo.PogoParams()
    s = mujoco.MjSpec()
    s.option.gravity = [0, 0, 0]
    s.option.timestep = 0.00025
    s.option.integrator = mujoco.mjtIntegrator.mjINT_RK4
    s.option.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    b = s.worldbody.add_body()
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=38)
    b.add_joint(name="compression", type=mujoco.mjtJoint.mjJNT_SLIDE)
    p.add_spring(s)
    m = s.compile()
    d = mujoco.MjData(m)
    d.qpos[0] = 0.15
    mujoco.mj_forward(m, d)
    initial = d.energy.sum()
    for _ in range(2000):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    assert d.energy.sum() == pytest.approx(initial, rel=1e-6)


def test_params_validation():
    with pytest.raises(ValueError):
        pogo.PogoParams(pressure=0)
    with pytest.raises(ValueError):
        pogo.PogoParams(damping=-1)
    with pytest.raises(ValueError):
        pogo.PogoParams(volume=0.0001)
    with pytest.raises(ValueError):
        pogo.PogoParams(exponent=1.5)
