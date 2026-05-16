from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest

from configs.config_exp import ConfigValidationError, ExpCfg
from scripts.train import build_rl_runtime_spec
from utils.config.paths import load_project_paths
from utils.experiment.rl_runtime_spec import runtime_spec_contract, validate_runtime_spec


ROOT = Path(__file__).resolve().parents[1]
ASSET_ASSIGNMENT_PATH = (
    ROOT
    / "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
    / "isaaclab_nonprehensile/asset_assignment.py"
)


def _load_asset_assignment_module():
    spec = importlib.util.spec_from_file_location("asset_assignment_under_test", ASSET_ASSIGNMENT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _source(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_asset_assignment_deterministic_offsets_across_ranks():
    assignment = _load_asset_assignment_module()

    assert assignment.asset_indices_for_rank(
        4,
        0,
        3,
        randomize=False,
        seed=123,
        salt=assignment.TOOL_ASSIGNMENT_SALT,
    ) == [0, 1, 2, 0]
    assert assignment.asset_indices_for_rank(
        4,
        1,
        3,
        randomize=False,
        seed=123,
        salt=assignment.TOOL_ASSIGNMENT_SALT,
    ) == [1, 2, 0, 1]
    assert assignment.global_env_id(3, 4, 2) == 11


def test_asset_assignment_random_seed_and_salt_contract():
    assignment = _load_asset_assignment_module()

    tool_seed_7 = assignment.asset_indices_for_rank(
        64,
        0,
        7,
        randomize=True,
        seed=7,
        salt=assignment.TOOL_ASSIGNMENT_SALT,
    )
    assert tool_seed_7 == assignment.asset_indices_for_rank(
        64,
        0,
        7,
        randomize=True,
        seed=7,
        salt=assignment.TOOL_ASSIGNMENT_SALT,
    )
    assert tool_seed_7 != assignment.asset_indices_for_rank(
        64,
        0,
        7,
        randomize=True,
        seed=8,
        salt=assignment.TOOL_ASSIGNMENT_SALT,
    )
    assert tool_seed_7 != assignment.asset_indices_for_rank(
        64,
        0,
        7,
        randomize=True,
        seed=7,
        salt=assignment.OBJECT_ASSIGNMENT_SALT,
    )


def test_general_assignment_flags_validate_as_bool():
    cfg = ExpCfg(name="asset_assignment_bool_unit")
    cfg.general.randomize_tool_assignment = "yes"  # type: ignore[assignment]

    with pytest.raises(ConfigValidationError, match="randomize_tool_assignment must be a bool"):
        cfg.validate()


def test_runtime_spec_contains_asset_assignment_params(tmp_path):
    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text("{}\n", encoding="utf-8")
    paths = load_project_paths(paths_yaml)

    cfg = ExpCfg(name="asset_assignment_runtime_unit")
    cfg.general.seed = 42
    cfg.general.randomize_tool_assignment = True
    cfg.general.randomize_object_assignment = False
    cfg.rl.encoder_checkpoint = str(tmp_path / "encoder.pt")

    spec = build_rl_runtime_spec(cfg, paths, tmp_path / "artifacts")
    payload = asdict(spec)

    assert payload["asset_assignment_params"] == {
        "seed": 42,
        "randomize_tool_assignment": True,
        "randomize_object_assignment": False,
    }
    assert payload["seed"] == 42
    validate_runtime_spec(payload, tmp_path / "rl_runtime_spec.json")
    contract = runtime_spec_contract(payload)
    assert contract.seed == 42
    assert contract.asset_assignment.seed == 42
    assert contract.asset_assignment.randomize_tool_assignment is True
    assert contract.asset_assignment.randomize_object_assignment is False


def test_env_tool_uses_expanded_spawn_lists_with_deterministic_spawners():
    env_tool = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/env_tool.py"
    )

    assert "TOOL_ASSET_INDICES_BY_ENV" in env_tool
    assert "OBJECT_ASSET_INDICES_BY_ENV" in env_tool
    assert "TOOL_USD_PATHS_BY_ENV" in env_tool
    assert "OBJECT_ASSET_CFGS_BY_ENV" in env_tool
    assert "assets_cfg=OBJECT_ASSET_CFGS_BY_ENV" in env_tool
    assert "build_multi_tool_robot_cfg(TOOL_USD_PATHS_BY_ENV, random_choice=False)" in env_tool
    assert "random_choice=False" in env_tool


def test_launcher_sets_rank_env_vars_before_task_registration():
    launcher = _source("utils/experiment/isaac_rl_launcher.py")

    for name in (
        "TOOL_GENERALIST_GLOBAL_RANK",
        "TOOL_GENERALIST_LOCAL_RANK",
        "TOOL_GENERALIST_WORLD_SIZE",
    ):
        assert f'os.environ["{name}"]' in launcher
    assert launcher.index("_set_asset_assignment_rank_env(spec, app_launcher)") < launcher.index(
        "import IsaacLab_nonPrehensile.tasks"
    )
    assert 'seed = int(spec["seed"])' in launcher
    assert "env_cfg.seed = seed" in launcher
    assert "agent_cfg.seed = seed" in launcher


def test_eval_tools_materializes_runtime_spec_for_cli_num_envs():
    source = _source("scripts/eval_tools.py")

    assert 'eval_runtime_spec["num_envs"] = args_cli.num_envs' in source
    assert 'eval_runtime_spec["env_params"]["num_envs"] = args_cli.num_envs' in source
    assert 'os.environ[RUNTIME_SPEC_ENV_VAR] = eval_runtime_spec_path' in source
