"""Opt-in GPU gates: POGO_GPU_TESTS=1 uv run --locked --extra train pytest ..."""

import os

import mujoco
import numpy as np
import pytest

if os.environ.get("POGO_GPU_TESTS") != "1":
    pytest.skip("GPU tests are opt-in on shared machines", allow_module_level=True)

import mujoco_warp as mjwarp  # noqa: E402
import warp as wp  # noqa: E402
from test_contact import probe  # noqa: E402

from pogo_g1 import pogo  # noqa: E402


@pytest.mark.parametrize("spin,mu", [(False, 0.2), (False, 0.8), (True, 0.8)])
def test_native_gpu_contact_matches_cpu(spin, mu):
    m, initial, params = probe(mu=mu, spin=spin)
    capacity = mu * 40 * 9.81
    if spin:
        capacity *= 2 / 3 * params.tip_patch_radius
    expected = []
    for ratio in [0.8, 1.2]:
        d = mujoco.MjData(m)
        mujoco.mj_copyData(d, m, initial)
        d.qfrc_applied[1] = ratio * capacity
        for _ in range(round(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        expected.append(d.qpos[1])

    with wp.ScopedDevice("cuda:0"):
        model = mjwarp.put_model(m)
        data = mjwarp.put_data(m, initial, nworld=2, nconmax=16, njmax=64)
        applied = np.zeros((2, m.nv), dtype=np.float32)
        applied[:, 1] = np.array([0.8, 1.2]) * capacity
        data.qfrc_applied.assign(applied)
        for _ in range(round(0.5 / m.opt.timestep)):
            mjwarp.step(model, data)
        actual = data.qpos.numpy()[:, 1]
    np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.001)
    assert abs(actual[0]) < (0.02 if spin else 0.002)
    assert actual[1] > (0.1 if spin else 0.01)


@pytest.mark.parametrize("height", [0.15, 0.5])
def test_coupled_pneumatic_drop_matches_cpu(height):
    spec = pogo.build_spec(floor=True)
    frame = spec.body("pogo_frame")
    spec.delete(frame.joints[0])
    frame.add_joint(name="height", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    frame.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0, 0],
        mass=33.8068,
        contype=0,
        conaffinity=0,
    )
    spec.option.timestep = pogo.PHYSICS_TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m = spec.compile()
    assert m.body_subtreemass[0] == pytest.approx(38.3468)
    initial = mujoco.MjData(m)
    initial.qpos[0] = height
    mujoco.mj_forward(m, initial)
    cpu = mujoco.MjData(m)
    mujoco.mj_copyData(cpu, m, initial)
    expected = []
    steps = round(2 / m.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(m, cpu)
        expected.append(cpu.qpos.copy())
    with wp.ScopedDevice("cuda:0"):
        model = mjwarp.put_model(m)
        data = mjwarp.put_data(m, initial, nworld=1, nconmax=32, njmax=128)
        actual = []
        for _ in range(steps):
            mjwarp.step(model, data)
            actual.append(data.qpos.numpy()[0].copy())
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.003)
