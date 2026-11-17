"""CPU/Warp parity, material randomisation and terrain/command integration gates.

No learned controller is evaluated here. Zero body actions are a negative control.
"""

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from pogo_g1 import mdp, pogo
from pogo_g1.env import TaskCfg, make_env_cfg


def check(kind, worlds, steps):
    task = TaskCfg(worlds=worlds, robust=True, airborne_prob=0, terrain=kind)
    env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
    try:
        assert abs(env.sim.mj_model.opt.timestep - task.timestep) < 1e-12
        assert abs(env.step_dt - 0.02) < 1e-12
        env.reset()
        joint = env.sim.mj_model.joint("pogo/compression")
        x = torch.linspace(0, task.params.stroke, worlds, device=env.device)
        env.scene["pogo"].write_joint_state_to_sim(
            x[:, None], torch.zeros_like(x[:, None])
        )
        env.sim.forward()
        env.scene.update(0.0)
        env.sim.sense()
        cpu = copy.copy(env.sim.mj_model)
        data = mujoco.MjData(cpu)
        qpos = env.sim.data.qpos.cpu().numpy()
        spring_gpu = env.sim.data.qfrc_spring[:, joint.dofadr[0]].cpu().numpy()
        linear = env.sim.model.jnt_stiffness[:, joint.id].cpu().numpy()
        poly = env.sim.model.jnt_stiffnesspoly[:, joint.id].cpu().numpy()
        gas_ids = mdp._consts(env)["gas_ids"]
        gas_poly = env.sim.model.tendon_stiffnesspoly[:, gas_ids].cpu().numpy()
        errors = []
        for i in range(worlds):
            cpu.jnt_stiffness[joint.id] = linear[i]
            cpu.jnt_stiffnesspoly[joint.id] = poly[i]
            cpu.tendon_stiffnesspoly[gas_ids] = gas_poly[i]
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(cpu, data)
            force = data.qfrc_spring[joint.dofadr[0]]
            errors.append(abs(spring_gpu[i] - force) / max(1.0, abs(force)))
        assert max(errors) < 1e-5, errors
        base = task.params.spring[1]
        factors = linear / base[0]
        assert factors.min() >= 0.8 and factors.max() <= 1.2
        assert np.ptp(factors) > 0.05
        np.testing.assert_allclose(
            poly, factors[:, None] * np.array(base[1:]), rtol=1e-6
        )
        np.testing.assert_allclose(
            gas_poly[:, :, 1], factors[:, None] * task.params.gas_terms[1], rtol=1e-6
        )
        np.testing.assert_allclose(
            -mdp.spring_force(env).cpu().numpy(), spring_gpu, rtol=1e-5, atol=0.001
        )
        expected_energy = factors * task.params.spring_energy(
            x.cpu().numpy().astype(np.float64)
        )
        np.testing.assert_allclose(
            mdp.spring_energy(env).cpu().numpy(), expected_energy, rtol=1e-5, atol=0.001
        )
        torch.testing.assert_close(
            mdp.system_mass(env), env.sim.model.body_mass.sum(-1), rtol=1e-5, atol=1e-5
        )

        # Check actual per-world coefficients; low values must not be masked by
        # a geom default. Both tangential axes stay equal.
        pf = env.sim.model.pair_friction.cpu().numpy()
        ids = [
            i for i in range(cpu.npair) if cpu.pair(i).name.startswith("contact/tip/")
        ]
        assert ids
        assert np.allclose(pf[:, ids, 0], pf[:, ids, 1])
        assert pf[:, ids, 0].min() >= 0.4 and pf[:, ids, 0].max() <= 1.0

        env.reset()
        clearance = mdp.tip_clearance(env).clone()
        assert float(clearance.abs().max()) < 0.005, clearance
        assert not mdp.terrain_missing(env).any()
        # Sample away from the spawn, not just at its flat centre. Kinematic ray
        # probes are reset before the subsequent physical zero-action rollout.
        offset = torch.linspace(-2, 2, worlds, device=env.device)
        for name in ("robot", "pogo"):
            entity = env.scene[name]
            pose = entity.data.root_link_pose_w.clone()
            pose[:, 0] += offset
            pose[:, 1] += torch.sin(offset)
            pose[:, 2] += 0.5
            entity.write_root_state_to_sim(
                torch.cat((pose, torch.zeros_like(pose[:, :6])), dim=1)
            )
        env.sim.forward()
        env.scene.update(0.0)
        env.sim.sense()
        ray = env.scene["tip_ground_height"].data
        assert not mdp.terrain_missing(env).any()
        if kind != "flat":
            xs = ray.frame_pos_w[:, 0, 0]
            assert float(xs.max() - xs.min()) > 3.9
        # Native GPU height rays must agree with CPU rays through the same terrain.
        mujoco.mj_forward(cpu, data)
        ray_errors = []
        for i, point in enumerate(ray.frame_pos_w[:, 0].cpu().numpy()):
            dist = mujoco.mj_ray(
                cpu,
                data,
                point.astype(np.float64),
                np.array([0.0, 0.0, -1.0]),
                np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                True,
                -1,
                np.zeros(1, dtype=np.int32),
            )
            ray_errors.append(abs(dist - float(ray.distances[i, 0])))
        assert max(ray_errors) < 0.001, ray_errors

        commands = ((0.2, 0, 0), (0, 0.2, 0), (-0.2, 0, 0), (0, -0.2, 0), (0, 0, 0.3))
        term = env.command_manager.get_term("hop")
        term.set_command(0.1, *commands[0])
        env.reset()  # external input persists through reset and resampling
        expected = torch.tensor([0.1, 0.2, 0, 0], device=env.device).expand(worlds, -1)
        assert torch.allclose(term.command, expected)
        actions = torch.zeros(worlds, 29, device=env.device)
        resets = 0
        max_force = 0.0
        for i in range(steps):
            command = commands[min(len(commands) - 1, i * len(commands) // steps)]
            term.set_command(0.1, *command)
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
            assert not mdp.nonfinite(env).any()
            expected = torch.tensor(command, device=env.device).expand(worlds, -1)
            assert torch.allclose(term.command[:, 1:], expected)
            resets += int((terminated | truncated).sum())
            max_force = max(max_force, float(mdp.grip_force_max(env).max()))
        return {
            "terrain": kind,
            "worlds": worlds,
            "steps": steps,
            "spring_force_max_relative_cpu_gpu_error": max(errors),
            "terrain_ray_max_cpu_gpu_error_m": max(ray_errors),
            "initial_tip_clearance_max_abs_m": float(clearance.abs().max()),
            "ground_mu_range": [float(pf[:, ids, 0].min()), float(pf[:, ids, 0].max())],
            "resets_with_zero_actions": resets,
            "max_grip_force_n": max_force,
            "passed": True,
        }
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--terrains", nargs="+", default=["flat", "waves", "slope"])
    ap.add_argument("--out", type=Path, default=Path("artifacts/physics-v4-gpu.json"))
    args = ap.parse_args()
    report = {
        "physics_version": pogo.PHYSICS_VERSION,
        "conditions": [],
        "passed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for kind in args.terrains:
            result = check(kind, args.worlds, args.steps)
            report["conditions"].append(result)
            print(json.dumps(result), flush=True)
        report["passed"] = True
    finally:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
