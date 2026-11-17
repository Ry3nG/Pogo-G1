import json
from dataclasses import asdict

import pytest

pytest.importorskip("mjlab", reason="requires train extra")
import torch  # noqa: E402

from pogo_g1.env import TaskCfg  # noqa: E402
from pogo_g1.evaluate import HoppingContinuity, load_task  # noqa: E402
from pogo_g1.pogo import PHYSICS_VERSION  # noqa: E402


def test_continuity_distinguishes_standing_late_hops_and_physical_failure():
    tracker = HoppingContinuity(4, "cpu")
    alive = torch.tensor([True, True, True, True])
    tracker.update(1.0, torch.tensor([True, False, True, True]), alive)
    alive[3] = False
    tracker.update(2.0, torch.tensor([True, False, False, False]), alive)
    # Third world resumes too late; its earlier gap remains a failure.
    tracker.update(3.5, torch.tensor([True, False, True, False]), alive)
    assert tracker.success.tolist() == [True, False, False, False]


def test_config_preserves_material_and_terrain_parameters(tmp_path):
    task = TaskCfg(terrain="waves", seed=17)
    cfg = {"physics_version": PHYSICS_VERSION, "task": asdict(task)}
    (tmp_path / "config.json").write_text(json.dumps(cfg, default=str))
    loaded = load_task(tmp_path)
    assert loaded.params == task.params
    assert loaded.contacts == task.contacts
    assert loaded.terrain == "waves" and loaded.seed == 17


@pytest.mark.parametrize("version", ["legacy-linear", "pneumatic-v2", "pneumatic-v3"])
def test_legacy_checkpoint_is_not_silently_reinterpreted(tmp_path, version):
    (tmp_path / "config.json").write_text(
        json.dumps({"physics_version": version, "task": {"params": {}}})
    )
    with pytest.raises(ValueError, match=version):
        load_task(tmp_path)


def test_command_sequence_keeps_failed_worlds_failed_and_observation_history(
    monkeypatch,
):
    """Transitions share the first episode; input appears on the next real step."""
    from types import SimpleNamespace

    from pogo_g1 import evaluate

    command = torch.tensor([[0.12, 0.0, 0.0, 0.0]]).repeat(2, 1)
    observed = []

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, *args, **kwargs):
            pass

        def get_inference_policy(self, **kwargs):
            def policy(obs):
                observed.append(obs.clone())
                return torch.zeros(2, 1)

            return policy

    class Env:
        num_envs, device, step_dt = 2, "cpu", 1.0
        _land_apex = torch.full((2,), 0.12)
        index = 0

        def get_observations(self):
            return command.clone()

        def step(self, action):
            dones = torch.tensor([self.index == 3, self.index == 1])
            self.last_dones = dones
            self.index += 1
            return command.clone(), torch.zeros(2), dones, {}

    env = Env()
    env.termination_manager = SimpleNamespace(
        active_terms=["timeout", "grip_lost"],
        get_term=lambda name: (
            torch.zeros(2, dtype=torch.bool) if name == "timeout" else env.last_dones
        ),
    )
    env.command_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(
            set_command=lambda *values: command.copy_(torch.tensor(values))
        )
    )
    env.scene = {
        "robot": SimpleNamespace(
            actuator_names=["wrist_pitch"],
            joint_names=["wrist_pitch"],
            data=SimpleNamespace(
                root_link_ang_vel_w=torch.tensor([[0.0, 0.0, 0.1]]).repeat(2, 1),
                actuator_force=torch.zeros(2, 1),
                joint_vel=torch.zeros(2, 1),
            ),
        )
    }
    env.sim = SimpleNamespace(
        mj_model=SimpleNamespace(
            actuator=lambda name: SimpleNamespace(forcerange=[-1, 1])
        )
    )
    monkeypatch.setattr(evaluate, "MjlabOnPolicyRunner", Runner)
    monkeypatch.setattr(evaluate, "RslRlVecEnvWrapper", lambda env, **kwargs: env)
    monkeypatch.setattr(evaluate.mdp, "hop_command", lambda env: command)
    monkeypatch.setattr(evaluate.mdp, "hops", lambda env: torch.ones(2))
    monkeypatch.setattr(
        evaluate.mdp,
        "com_velocity_yaw",
        lambda env: torch.tensor([[0.2, 0.0]]).repeat(2, 1),
    )
    monkeypatch.setattr(
        evaluate.mdp, "grip_wrench", lambda env: (torch.zeros(2, 2), torch.zeros(2, 2))
    )
    monkeypatch.setattr(evaluate.mdp, "compression", lambda env: torch.full((2,), 0.1))
    monkeypatch.setattr(evaluate.mdp, "centroidal_yaw_rate", lambda env: torch.zeros(2))
    steps = (
        (0.0, (0.12, 0, 0, 0)),
        (1.0, (0.12, 0.4, 0, 0)),
        (2.0, (0.12, 0, 0, 0)),
        (3.0, (0.12, 0, 0, 0.5)),
    )
    result = evaluate.Evaluator("unused", "model_0.pt")._rollout(
        env, TaskCfg(), 4.0, None, steps
    )
    assert result["success"] == result["survival"] == 0.0
    assert result["fall_cause"] == {"grip_lost": 2}
    assert result["stroke_frac_max"] == pytest.approx(0.1 / TaskCfg().params.stroke)
    segments = result["segments"]
    assert [s["alive_samples"] for s in segments] == [2, 1, 1, 0]
    assert [s["survival_end"] for s in segments] == [1.0, 0.5, 0.5, 0.0]
    assert [s["vel_err"] for s in segments[:3]] == pytest.approx([0.2] * 3)
    assert segments[-1]["vel_err"] is segments[-1]["yaw_err"] is None
    assert [s["observation_effective_s"] for s in segments] == [1, 2, 3, 4]
    assert [float(obs[0, 1]) for obs in observed] == pytest.approx([0, 0, 0.4, 0])


def test_command_conditions_run_by_default_with_the_fixed_protocol():
    import runpy
    from pathlib import Path

    suite = runpy.run_path(str(Path(__file__).parents[1] / "scripts/eval_suite.py"))
    extras = {"nominal_0.12", "fwd_0.2", "back_0.15", "turn_-0.5", "command_sequence"}
    assert extras <= suite["CONDITIONS"].keys()
    assert [t for t, _ in suite["SEQUENCE_STEPS"]] == [0, 5, 15, 20, 25]
    assert all(values[0] == 0.12 for _, values in suite["SEQUENCE_STEPS"])


def test_run_pose_takes_precedence_over_old_host_path(tmp_path):
    task = TaskCfg(pose_path="/old-host/run/mounted_pose.json")
    config = {"physics_version": PHYSICS_VERSION, "task": asdict(task)}
    (tmp_path / "config.json").write_text(json.dumps(config, default=str))
    pose = tmp_path / "mounted_pose.json"
    pose.write_text("{}")
    assert load_task(tmp_path).pose_path == str(pose)
    pose.unlink()
    assert load_task(tmp_path).pose_path == task.pose_path
