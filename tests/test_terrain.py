"""Terrain presets: config checks on the CPU, spawn checks on the GPU (opt-in)."""

import os

import pytest

pytest.importorskip("mjlab", reason="mjlab (train extra) not installed")

from pogo_g1 import terrain  # noqa: E402


@pytest.mark.parametrize("kind", terrain.KINDS)
def test_every_kind_builds_a_config(kind):
    cfg = terrain.terrain_cfg(kind, 0.05, 0.1, seed=0)
    if kind == "flat":
        assert cfg.terrain_type == "plane"
    else:
        assert cfg.terrain_type == "generator"
        assert list(cfg.terrain_generator.sub_terrains) == [kind]


def test_bad_values_raise():
    with pytest.raises(ValueError):
        terrain.terrain_cfg("rough", 0.0, 0.1, 0)
    with pytest.raises(ValueError):
        terrain.terrain_cfg("slope", 0.1, 0.0, 0)
    with pytest.raises(ValueError):
        terrain.terrain_cfg("mesa", 0.1, 0.1, 0)


@pytest.mark.skipif(
    os.environ.get("POGO_GPU_TESTS") != "1",
    reason="GPU tests are opt-in on shared machines",
)
@pytest.mark.parametrize("kind", ["rough", "steps"])
def test_mounted_spawn_touches_the_ground(kind):
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from pogo_g1 import mdp
    from pogo_g1.env import TaskCfg, make_env_cfg

    task = TaskCfg(worlds=2, airborne_prob=0.0, terrain=kind, terrain_height=0.05)
    env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
    try:
        env.reset()
        zero = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device="cuda:0"
        )
        # Contact sensors report one policy step after a reset (MuJoCo 3.15).
        env.step(zero)
        env.step(zero)
        assert mdp.tip_clearance(env).abs().max().item() < 0.02
        assert mdp.tip_contact(env).all()
        assert not mdp.terrain_missing(env).any()
    finally:
        env.close()
