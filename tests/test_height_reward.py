"""Protect actual-height credit, command attribution and reward rate."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab", reason="requires train extra")
import torch  # noqa: E402

from pogo_g1 import mdp  # noqa: E402


def flight_env(monkeypatch, dt=0.02):
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=dt,
        common_step_counter=0,
        contact=torch.ones(2, dtype=torch.bool),
        clearance=torch.zeros(2),
        command=torch.tensor([[0.2, 0, 0, 0], [0.2, 0, 0, 0]]),
        termination_manager=SimpleNamespace(
            terminated=torch.zeros(2, dtype=torch.bool)
        ),
    )
    monkeypatch.setattr(mdp, "tip_contact", lambda e: e.contact.clone())
    monkeypatch.setattr(mdp, "tip_clearance", lambda e: e.clearance)
    # Deliberately wrong energy proxy: landing reward must use actual clearance.
    monkeypatch.setattr(mdp, "energy_height", lambda e: torch.full((2,), 0.8))
    monkeypatch.setattr(mdp, "hop_command", lambda e: e.command)
    env.motion = torch.zeros(2, 3)
    monkeypatch.setattr(mdp, "com_velocity_yaw", lambda e: e.motion[:, :2])
    monkeypatch.setattr(mdp, "yaw_rate", lambda e: e.motion[:, 2])
    return env


def step(env, contact, clearance=0.0):
    env.common_step_counter += 1
    env.contact[:] = contact
    env.clearance[:] = clearance
    return mdp.hops(env)


def test_actual_apex_and_takeoff_command_survive_midair_command_change(monkeypatch):
    env = flight_env(monkeypatch)
    step(env, True)
    step(env, False, 0.1)
    env.command[:, 0] = 0.35
    for _ in range(6):
        step(env, False, 0.2)
    assert step(env, True).tolist() == [1, 1]
    assert env._land_command.tolist() == pytest.approx([0.2, 0.2])
    assert env._land_apex.tolist() == pytest.approx([0.2, 0.2])
    reward = mdp.landed_height(env, 0.03) * env.step_dt
    torch.testing.assert_close(reward, env._land_period)
    # Repeated metric/reward reads must not advance the cached tracker.
    torch.testing.assert_close(mdp.landed_height(env, 0.03) * env.step_dt, reward)
    env.termination_manager.terminated[1] = True
    assert mdp.landed_height(env, 0.03)[1] == 0
    step(env, False)
    assert env._air_command.tolist() == pytest.approx([0.35, 0.35])


@pytest.mark.parametrize("dt", [0.01, 0.02])
def test_rate_is_cycle_time_scaled_and_standing_earns_no_landing_reward(
    monkeypatch, dt
):
    env = flight_env(monkeypatch, dt)
    for _ in range(round(0.3 / dt)):
        step(env, True)
        assert not mdp.landed_height(env, 0.03).any()
    for _ in range(round(0.2 / dt)):
        step(env, False, 0.2)
    step(env, True)
    reward = mdp.landed_height(env, 0.03) * dt
    torch.testing.assert_close(reward / env._land_period, torch.ones(2))
    env._land_period[:] = 10
    torch.testing.assert_close(mdp.landed_height(env, 0.03) * dt, torch.full((2,), 2.0))


def test_invalid_flights_get_no_credit(monkeypatch):
    env = flight_env(monkeypatch)
    step(env, True)
    step(env, False, 0.2)
    step(env, True)  # high but too short
    assert not mdp.landed_height(env, 0.03).any()
    for _ in range(8):
        step(env, False, 0.01)
    step(env, True)  # long enough but below the existing 2 cm definition
    assert not mdp.landed_height(env, 0.03).any()


def test_broad_tail_still_prefers_exact_tracking_without_rewarding_overshoot():
    error = torch.tensor([-0.2, -0.1, 0, 0.1, 0.2])
    score = mdp.height_score(error, 0.03)
    assert 0 < score[0] < score[1] < score[2] == 1
    torch.testing.assert_close(score, score.flip(0))


def test_steering_scores_the_hop_cycle_average_not_the_swing(monkeypatch):
    env = flight_env(monkeypatch)
    target = torch.tensor([0.4, 0.0, 0.5])
    env.command[:, 1:] = target
    # One cycle: 2 stance steps, 9 flight steps, landing. World 0 swings by
    # +-1.5 around the command (zero mean); world 1 holds still.
    for k in range(12):
        env.motion[0] = target + (1.5 if k % 2 else -1.5)
        hop = step(env, k < 2 or k == 11, 0.0 if k < 2 or k == 11 else 0.1)
    assert hop.tolist() == [1, 1]
    torch.testing.assert_close(env._land_motion[0], target)
    torch.testing.assert_close(env._land_motion[1], torch.zeros(3))
    vel = mdp.landed_velocity(env, 0.25) * env.step_dt
    yaw = mdp.landed_yaw_rate(env, 0.5) * env.step_dt
    torch.testing.assert_close(vel[0], env._land_period[0])  # exact average
    torch.testing.assert_close(yaw[0], env._land_period[0])
    assert vel[1] < 0.3 * vel[0]
    assert float(yaw[1]) == pytest.approx(0.5 * float(yaw[0]))  # one std off


def ceiling_env(monkeypatch, cap=0.3):
    term = SimpleNamespace(
        cfg=SimpleNamespace(apex_cap=cap, ranges=SimpleNamespace(apex=(0.05, 0.15))),
        ceiling=torch.tensor(0.15),
        window=torch.zeros(4),
        held=torch.full((4,), 5.0),
    )
    env = SimpleNamespace(
        num_envs=4,
        command=torch.full((4, 4), 0.0),
        _land_command=torch.zeros(4),
        _land_apex=torch.zeros(4),
        hop=torch.zeros(4),
        command_manager=SimpleNamespace(get_term=lambda name: term),
        termination_manager=SimpleNamespace(
            terminated=torch.zeros(4, dtype=torch.bool)
        ),
    )
    monkeypatch.setattr(mdp, "hops", lambda e: e.hop)
    monkeypatch.setattr(mdp, "hop_command", lambda e: e.command)
    return env, term


def land(env, command, apex, top_band_falls=0):
    env.hop[:] = 1
    env._land_command[:] = command
    env._land_apex[:] = apex
    env.command[:, 0] = command
    env.termination_manager.terminated[:] = False
    env.termination_manager.terminated[:top_band_falls] = True
    return mdp.apex_ceiling(env, window=8)


def test_ceiling_rises_only_when_its_top_band_is_reached_safely(monkeypatch):
    env, term = ceiling_env(monkeypatch)
    land(env, 0.05, 0.05)  # below the band: never counts
    land(env, 0.05, 0.05)
    assert term.window.tolist() == [0, 0, 0, 0]
    land(env, 0.14, 0.10)  # reaches 71% of the command
    assert land(env, 0.14, 0.10).tolist() == pytest.approx([0.15] * 4)
    assert term.window.tolist() == [0, 0, 0, 0]  # a full window restarts
    land(env, 0.14, 0.125)  # 89%: reached, though 1.5 cm short
    land(env, 0.14, 0.125, top_band_falls=1)  # 1 fall per 8 landings > 2%
    assert float(term.ceiling) == pytest.approx(0.15)
    land(env, 0.14, 0.125)
    land(env, 0.14, 0.125)
    assert float(term.ceiling) == pytest.approx(0.20)
    for _ in range(10):
        command = float(term.ceiling) - 0.01
        land(env, command, command)
    assert float(term.ceiling) == pytest.approx(0.3)  # capped


def test_ceiling_ignores_landings_while_height_settles(monkeypatch):
    env, term = ceiling_env(monkeypatch)
    term.held[:] = 1.0  # command changed 1 s ago: still climbing
    land(env, 0.14, 0.05)
    assert term.window.tolist() == [0, 0, 0, 0]


def test_ceiling_is_fixed_without_a_cap(monkeypatch):
    env, term = ceiling_env(monkeypatch, cap=0.0)
    for _ in range(4):
        land(env, 0.14, 0.14)
    assert float(term.ceiling) == pytest.approx(0.15)
    assert term.window.tolist() == [0, 0, 0, 0]


@pytest.mark.skipif(
    os.environ.get("POGO_GPU_TESTS") != "1", reason="GPU tests are opt-in"
)
def test_native_warm_start_and_partial_reset():
    """Protect policy reuse and keep reward history out of the next episode."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from pogo_g1.env import TaskCfg, make_env_cfg
    from pogo_g1.ppo import runner_cfg, to_dict

    ckpt = Path("artifacts/runs/0928-0000-v5-m1-s0/model_2999.pt")
    if not ckpt.exists():
        pytest.skip("local baseline checkpoint required")
    # The M1 baseline never saw direction commands: keep them at zero.
    task = TaskCfg(
        worlds=4,
        apex=(0.15, 0.15),
        apex_cap=0.0,
        vx=(0.0, 0.0),
        vy=(0.0, 0.0),
        yaw_rate=(0.0, 0.0),
        airborne_prob=0,
    )
    env = ManagerBasedRlEnv(make_env_cfg(task), device="cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env)
        runner = MjlabOnPolicyRunner(
            wrapped, to_dict(runner_cfg()), None, device="cuda:0"
        )
        critic_before = {
            k: v.clone() for k, v in runner.alg.critic.state_dict().items()
        }
        runner.load(str(ckpt), load_cfg={"actor": True})
        saved = torch.load(ckpt, weights_only=False, map_location="cuda:0")
        for k, v in runner.alg.actor.state_dict().items():
            torch.testing.assert_close(v, saved["actor_state_dict"][k])
        for k, v in runner.alg.critic.state_dict().items():
            torch.testing.assert_close(v, critic_before[k])
        assert runner.current_learning_iteration == 0
        assert not runner.alg.optimizer.state
        policy = runner.get_inference_policy(device="cuda:0")
        obs = wrapped.get_observations()
        with torch.inference_mode():
            for _ in range(150):
                obs, reward, _, _ = wrapped.step(policy(obs))
                assert torch.isfinite(reward).all()
            untouched = env._hop_elapsed[1:].clone()
            env.reset(env_ids=torch.tensor([0], device="cuda:0"))
            for name in [
                "_air_command",
                "_land_command",
                "_hop_elapsed",
                "_land_period",
            ]:
                assert getattr(env, name)[0] == 0
            torch.testing.assert_close(env._hop_elapsed[1:], untouched)
    finally:
        env.close()
