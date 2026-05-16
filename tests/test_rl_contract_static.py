from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _top_imports(rel_path: str) -> list[str]:
    tree = ast.parse(_source(rel_path))
    return [
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def test_run_experiment_is_single_lightweight_entrypoint():
    source = _source("run_experiment.py")
    flags = [
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("--")
    ]

    assert flags == ["--config", "--mode"]
    assert "choices=(\"run\", \"plan\")" in source
    assert "default=\"run\"" in source
    for item in _top_imports("run_experiment.py"):
        assert "torch" not in item.lower()
        assert "isaac" not in item.lower()
        assert "rsl_rl" not in item
        assert "contact_generation" not in item
        assert "pretrain" not in item


def test_runtime_spec_allows_only_tg_and_point2vec_with_shared_head_fields():
    runtime_loader = _source("utils/experiment/rl_runtime_spec.py")
    train_script = _source("scripts/train.py")
    launcher = _source("utils/experiment/isaac_rl_launcher.py")

    assert "SUPPORTED_POLICY_CLASSES = {\"ActorCriticTG\", \"ActorCriticPoint2Vec\"}" in runtime_loader
    for key in (
        "sd_num_query",
        "sd_emb_dim",
        "cross_attn_heads",
        "cross_attn_layers",
        "fusion_hidden_dims",
        "actor_hidden_dims",
        "critic_hidden_dims",
        "activation",
        "init_noise_std",
        "noise_std_type",
    ):
        assert key in runtime_loader
        assert key in train_script
    assert "actor_critic_class=rl.actor_critic_class" in train_script
    assert "encoder_checkpoint_override" in train_script
    assert "or model.pretrained_encoder.checkpoint_path" in train_script
    assert "or model.encoder.checkpoint_path" in train_script
    assert "or exp_cfg.rl.encoder_checkpoint" in train_script
    assert "launch_from_runtime_spec" in train_script
    assert "Use run_experiment.py --config" in train_script
    assert "paths_yaml" in train_script
    assert "num_gpus" in train_script
    assert "num_gpus" in runtime_loader
    assert "launch_params" in runtime_loader
    assert "NotImplementedError" not in launcher
    assert "os.environ[RUNTIME_SPEC_ENV_VAR]" in launcher
    assert "TOOL_GENERALIST_PATHS_YAML" in launcher
    assert "OnPolicyRunner" in launcher
    assert "runner.learn(" in launcher
    assert "torch.distributed.run" in launcher
    assert "--nproc_per_node" in launcher
    assert "LOCAL_RANK" in launcher
    assert "setattr(args, \"distributed\"" in launcher
    assert "def _local_rank(" in launcher
    assert "def main(" in launcher
    assert "--runtime-spec" in launcher


def test_multi_gpu_env_count_is_documented_as_per_rank():
    readme = _source("README.md")
    config_rl = _source("configs/config_rl.py")

    assert "EXP_CFG.num_gpus = 4" in readme
    assert "EXP_CFG.rl.launch.distributed = True" in readme
    assert "EXP_CFG.rl.env.num_envs = 1024  # per GPU/rank" in readme
    assert "EXP_CFG.num_gpus * EXP_CFG.rl.env.num_envs" in readme
    assert "Per-GPU/per-rank environment count" in config_rl


def test_rl_defaults_match_original_training_contract():
    config_rl = _source("configs/config_rl.py")
    point2vec_cfg = _source("configs/experiments/point2vec.py")
    readme = _source("README.md")

    for token in (
        "max_iterations: int = 1000000",
        "learning_rate: float = 5.0e-5",
        "clip_param: float = 0.3",
        "entropy_coef: float = 0.005",
        "value_loss_coef: float = 0.5",
        "desired_kl: float = 0.016",
        "num_steps_per_env: int = 8",
        "save_interval: int = 200",
        "schedule: str = \"adaptive\"",
        "gamma: float = 0.99",
        "lam: float = 0.95",
        "use_clipped_value_loss: bool = True",
        "num_learning_epochs: int = 8",
        "num_mini_batches: int = 8",
        "max_grad_norm: float = 1.0",
        "episode_length_s: float = 30.0",
        "decimation: int = 8",
        "sim_dt: float = 1.0 / 80.0",
        "sim_dt * decimation == 0.1",
        "scale: float | list[float] = 0.1",
        "clip: tuple[float, float] = (-1.0, 1.0)",
        "enabled: bool = True",
        "enabled: bool = False",
        "class ObjectPoseSamplingCfg",
        "initial_position_range: float = 0.15",
        "xy_offset_range: float = 0.15",
    ):
        assert token in config_rl
    assert "total envs = ExpCfg.num_gpus * RLCfg.env.num_envs" in config_rl
    assert "EXP_CFG.rl.ppo.save_interval = 500" in point2vec_cfg
    assert "EXP_CFG.rl.ppo.entropy_coef = 0.006" in point2vec_cfg
    assert "EXP_CFG.rl.ppo.max_iterations" not in point2vec_cfg
    assert "default is `1000000`" in readme


def test_point2vec_config_is_pure_config_backend_selection():
    point2vec_cfg = _source("configs/experiments/point2vec.py")
    model_cfg = _source("configs/config_model.py")
    exp_cfg = _source("configs/config_exp.py")

    assert "p2v: P2VCfg = field(default_factory=P2VCfg)" in model_cfg
    assert "return self.p2v" in model_cfg
    assert "\"point2vec_native\"" in model_cfg
    assert "EXP_CFG.model.encoder_backend = \"point2vec\"" in point2vec_cfg
    assert "EXP_CFG.model.pretrained_encoder.name = \"point2vec\"" in point2vec_cfg
    assert "pre_point2vec-epoch.799-step.64800.ckpt" in point2vec_cfg
    assert "EXP_CFG.model.pretrained_encoder.schema = \"point2vec\"" in point2vec_cfg
    assert "EXP_CFG.model.pretrained_encoder.adapter = \"point2vec_native\"" in point2vec_cfg
    assert "EXP_CFG.rl.actor_critic_class = \"ActorCriticPoint2Vec\"" in point2vec_cfg
    assert "EXP_CFG.rl.table.enabled = True" in point2vec_cfg
    assert "EXP_CFG.rl.table.pose_xyz = [0.5, 0.0, -0.02]" in point2vec_cfg
    assert "EXP_CFG.rl.domain_randomization.ground.material.enabled = False" in point2vec_cfg
    assert "encoder_backend=point2vec requires" in exp_cfg
    assert "PretrainCfg.enabled currently supports only TCE with ActorCriticTG" in exp_cfg


def test_table_enabled_removes_ground_and_uses_shared_placement_helper():
    config_rl = _source("configs/config_rl.py")
    exp_cfg = _source("configs/config_exp.py")
    runtime_loader = _source("utils/experiment/rl_runtime_spec.py")
    env_tool = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/env_tool.py"
    )
    events = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/events.py"
    )
    commands = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/commands.py"
    )
    observations = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/observations.py"
    )
    table_placement = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/table_placement.py"
    )

    assert "placement_margin_xy: float = 0.02" in config_rl
    assert "placement_max_attempts: int = 64" in config_rl
    assert "class ObjectPoseSamplingCfg" in config_rl
    assert "initial_position_range: float = 0.15" in config_rl
    assert "xy_offset_range: float = 0.15" in config_rl
    assert "dr.ground.material.enabled and not self.table.enabled" in config_rl
    assert "TableCfg.enabled removes the ground plane" in exp_cfg
    assert "table_surface.material" in exp_cfg
    assert "\"object_pose_sampling_params\"" in runtime_loader
    assert "object_pose_sampling_params.{key} must be >= 0" in runtime_loader
    assert "field.startswith(\"ground_\")" in runtime_loader
    assert "cannot request ground_* physics fields" in runtime_loader
    assert "when table is" in runtime_loader

    assert "if not _RL_CONTRACT.table.enabled:" in env_tool
    assert "terrain = TerrainImporterCfg(" in env_tool
    assert "not _RL_CONTRACT.table.enabled" in env_tool
    assert "table_bounds_xy" in env_tool
    assert "table_placement_margin_xy" in env_tool
    assert "table_placement_max_attempts" in env_tool

    assert "_RL_CONTRACT.object_pose_sampling.xy_offset_range" in env_tool
    assert "_RL_CONTRACT.object_pose_sampling.initial_position_range" in env_tool

    for source in (events, commands):
        assert "mdp.table_placement import" in source
        assert "surface_z_for_points(" in source
        assert "_get_vertices_torch(" in source
        assert "sample_table_xy(" not in source
        assert "rotated_xy_half_extents(" not in source
        assert "placement_margin" not in source
    assert "tool_aabbs" not in events
    assert "overlap" not in events
    assert "torch.rand((), device=env.device) * 2.0 - 1.0) * xy_range" in events
    assert "torch.rand((), device=env.device) * 4.0 - 2.0) * xy_range" in events
    assert "mag_x = (0.5 + torch.rand(pos.shape[0], device=self.device) * 0.5) * r" in commands
    assert "mag_y = (0.5 + torch.rand(pos.shape[0], device=self.device) * 0.5) * r" in commands
    assert "pos[:, 0] = 0.5 + sign_x * mag_x" in commands
    assert "pos[:, 1] = 2.0 * (sign_y * mag_y)" in commands
    assert "tool_aabbs" not in events

    for token in (
        "def table_bounds_from_contract",
        "def rotated_xy_half_extents",
        "def table_safe_xy_range",
        "def sample_table_xy",
        "def surface_z_for_points",
    ):
        assert token in table_placement
    assert "phys_params requested ground_* fields" in observations


def test_tg_and_point2vec_use_shared_policy_common_and_bbox_centering():
    common = _source("rsl_rl/modules/tg_policy_common.py")
    tg_actor = _source("rsl_rl/modules/actor_critic_tg.py")
    p2v_actor = _source("rsl_rl/modules/actor_critic_point2vec.py")

    for token in (
        "class ObservationLayout",
        "def split_observations",
        "def center_clouds_by_bbox",
        "def build_context_vector",
        "tool_bbox_center - object_bbox_center",
        "def build_state_cross_attention",
        "def build_mlp",
        "def build_fusion_mlp",
    ):
        assert token in common
    for actor in (tg_actor, p2v_actor):
        assert "from rsl_rl.modules.tg_policy_common import" in actor
        assert "center_clouds_by_bbox(" in actor
        assert "build_context_vector(parts)" in actor
        assert "build_state_cross_attention(" in actor
        assert "build_fusion_mlp(" in actor
        assert "build_mlp(" in actor
        assert "strict=False" not in actor
    assert "object_tokens = self._encode_single_cloud(object_cloud_rel" in p2v_actor
    assert "tool_tokens = self._encode_single_cloud(tool_cloud_rel" in p2v_actor
    assert "torch.cat([tool_tokens, object_tokens], dim=1)" in p2v_actor
    assert "extra_state = obs[:, offset:]" not in p2v_actor
    assert "PointcloudCentering" not in p2v_actor
    assert "random initialization" not in p2v_actor


def test_policy_registry_and_isaac_bridge_are_whitelist_driven():
    modules = _source("rsl_rl/modules/__init__.py")
    agent_cfg = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/agents/config/rsl_rl_ppo_cfg.py"
    )

    assert "\"ActorCriticTG\": ActorCriticTG" in modules
    assert "\"ActorCriticPoint2Vec\": ActorCriticPoint2Vec" in modules
    assert "policy_class = eval" not in _source("rsl_rl/runners/on_policy_runner.py")
    assert "if _POLICY_CLASS_NAME == \"ActorCriticTG\"" in agent_cfg
    assert "if _POLICY_CLASS_NAME == \"ActorCriticPoint2Vec\"" in agent_cfg
    assert "class TGActorCriticCfg" in agent_cfg
    assert "class Point2VecActorCriticCfg" in agent_cfg
    assert "eval(" not in agent_cfg
    assert "point2vec_ckpt_path" in agent_cfg
    assert "patch_size" in agent_cfg
    assert "/mnt/project" not in agent_cfg
