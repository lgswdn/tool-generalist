from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
STEP_CACHE_PATH = (
    ROOT
    / "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
    "isaaclab_nonprehensile/mdp/step_cache.py"
)


def _load_step_cache_module():
    spec = importlib.util.spec_from_file_location("mdp_step_cache_under_test", STEP_CACHE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _CommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command
        self.calls = 0

    def get_command(self, name: str) -> torch.Tensor:
        assert name == "goal"
        self.calls += 1
        return self.command


class StepCacheTest(unittest.TestCase):
    def setUp(self):
        self.cache = _load_step_cache_module()
        root_pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        self.object = SimpleNamespace(
            data=SimpleNamespace(root_pos_w=root_pos, root_quat_w=root_quat)
        )
        command = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        self.manager = _CommandManager(command)
        self.env = SimpleNamespace(
            common_step_counter=7,
            command_manager=self.manager,
            scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
        )

    def test_goal_geometry_and_masks_are_shared_within_step(self):
        first = self.cache.object_goal_geometry(
            self.env, self.object, command_name="goal"
        )
        second = self.cache.object_goal_geometry(
            self.env, self.object, command_name="goal"
        )
        first_mask = self.cache.object_pose_success_mask(
            self.env,
            self.object,
            command_name="goal",
            threshold=0.1,
            rotation_threshold=0.1,
        )
        second_mask = self.cache.object_pose_success_mask(
            self.env,
            self.object,
            command_name="goal",
            threshold=0.1,
            rotation_threshold=0.1,
        )

        self.assertIs(first, second)
        self.assertIs(first_mask, second_mask)
        self.assertEqual(self.manager.calls, 1)
        self.assertTrue(torch.equal(first_mask, torch.tensor([True, False])))

    def test_step_increment_invalidates_cached_geometry(self):
        first = self.cache.object_goal_geometry(
            self.env, self.object, command_name="goal"
        )
        self.env.common_step_counter += 1
        second = self.cache.object_goal_geometry(
            self.env, self.object, command_name="goal"
        )

        self.assertIsNot(first, second)
        self.assertEqual(self.manager.calls, 2)

    def test_calls_without_step_counter_are_not_cached(self):
        env = SimpleNamespace()
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            return calls

        self.assertEqual(self.cache.get_or_compute_step_value(env, ("x",), factory), 1)
        self.assertEqual(self.cache.get_or_compute_step_value(env, ("x",), factory), 2)


class PhysParamsCacheWiringTest(unittest.TestCase):
    def test_observation_and_reset_paths_use_cache(self):
        observations = (
            ROOT
            / "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
            "isaaclab_nonprehensile/mdp/observations.py"
        ).read_text(encoding="utf-8")
        env_tool = (
            ROOT
            / "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
            "isaaclab_nonprehensile/env_tool.py"
        ).read_text(encoding="utf-8")
        bimanual_env = (
            ROOT
            / "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
            "isaaclab_nonprehensile/env_tool_bimanual_unstable.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def refresh_phys_params_cache(", observations)
        self.assertIn('getattr(env, "_phys_params_cache", None)', observations)
        self.assertIn("mdp.refresh_phys_params_cache(self, env_ids=env_ids)", env_tool)
        self.assertIn("mdp.refresh_phys_params_cache(", bimanual_env)
        self.assertIn('hand_cfg=SceneEntityCfg("robot_1")', bimanual_env)


if __name__ == "__main__":
    unittest.main()
