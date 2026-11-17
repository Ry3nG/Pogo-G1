"""Resume must preserve the experiment rather than silently restore defaults."""

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("mjlab", reason="requires train extra")

from pogo_g1.env import TaskCfg  # noqa: E402
from pogo_g1.pogo import PHYSICS_VERSION  # noqa: E402
from pogo_g1.ppo import runner_cfg, to_dict  # noqa: E402


def test_resume_inherits_recipe_seed_pose_and_starts_after_saved_iteration(
    tmp_path, monkeypatch
):
    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train", path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    parent = tmp_path / "parent"
    parent.mkdir()
    task = TaskCfg(seed=17, apex=(0.05, 0.35), apex_cap=1.2)
    agent = to_dict(runner_cfg(seed=17), symmetry=task.symmetry)
    agent["algorithm"].update(schedule="fixed", learning_rate=1e-4)
    (parent / "config.json").write_text(
        json.dumps(
            {"physics_version": PHYSICS_VERSION, "task": asdict(task), "agent": agent},
            default=str,
        )
    )
    (parent / "mounted_pose.json").write_text("saved pose")
    observed = {}

    class Env:
        def __init__(self, cfg, device):
            observed["task"] = cfg

        def close(self):
            pass

    class Runner:
        def __init__(self, wrapped, cfg, out, device):
            observed.update(agent=cfg, out=Path(out))

        def load(self, path):
            observed["checkpoint"] = path
            self.current_learning_iteration = 5999

        def learn(self, num_learning_iterations, init_at_random_ep_len):
            observed["start"] = self.current_learning_iteration
            observed["updates"] = num_learning_iterations

    monkeypatch.setattr(train, "make_env_cfg", lambda cfg: cfg)
    monkeypatch.setattr(train, "ManagerBasedRlEnv", Env)
    monkeypatch.setattr(train, "RslRlVecEnvWrapper", lambda env, clip_actions: env)
    monkeypatch.setattr(train, "MjlabOnPolicyRunner", Runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--name",
            "resume-test",
            "--iters",
            "30",
            "--resume",
            str(parent / "model_5999.pt"),
            "--root",
            str(tmp_path / "runs"),
            "--set",
            "apex=(0.05,0.60)",
        ],
    )
    train.main()
    assert observed["task"].seed == 17
    assert observed["task"].apex_cap == 1.2
    assert observed["task"].apex == (0.05, 0.60)
    assert observed["agent"]["algorithm"] == agent["algorithm"]
    assert observed["agent"]["seed"] == 17
    assert observed["start"] == 6000 and observed["updates"] == 30
    pose = observed["out"] / "mounted_pose.json"
    assert pose.read_text() == "saved pose"
    assert observed["task"].pose_path == str(pose)
    saved = json.loads((observed["out"] / "config.json").read_text())
    assert saved["task"]["apex_cap"] == 1.2
    assert saved["agent"]["algorithm"]["schedule"] == "fixed"
