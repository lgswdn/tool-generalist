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


def _class_node(rel_path: str, class_name: str) -> ast.ClassDef:
    for node in ast.walk(ast.parse(_source(rel_path))):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found in {rel_path}")


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


def test_runtime_spec_allows_tg_point2vec_and_legacy_icp_with_policy_specific_fields():
    runtime_loader = _source("utils/experiment/rl_runtime_spec.py")
    train_script = _source("scripts/train.py")
    launcher = _source("utils/experiment/isaac_rl_launcher.py")

    assert "\"ActorCriticTG\"" in runtime_loader
    assert "\"ActorCriticTGBimanual\"" in runtime_loader
    assert "\"ActorCriticPoint2Vec\"" in runtime_loader
    assert "\"ActorCriticICP\"" in runtime_loader
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
    assert "icp_weights_path" in runtime_loader
    assert "icp_weights_path" in train_script
    assert "policy_params.icp_weights_path must match encoder_checkpoint" in runtime_loader
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
    icp_cfg = _source("configs/experiments/icp_legacy_rl.py")
    model_cfg = _source("configs/config_model.py")
    exp_cfg = _source("configs/config_exp.py")

    assert "p2v: P2VCfg = field(default_factory=P2VCfg)" in model_cfg
    assert "return self.p2v" in model_cfg
    assert "\"point2vec_native\"" in model_cfg
    assert "class ICPCfg" in model_cfg
    assert "\"icp_legacy\"" in model_cfg
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
    assert "EXP_CFG.model.encoder_backend = \"icp\"" in icp_cfg
    assert "EXP_CFG.rl.actor_critic_class = \"ActorCriticICP\"" in icp_cfg
    assert "EXP_CFG.rl.env.robot_mode = \"bare_franka\"" in icp_cfg
    assert "EXP_CFG.rl.observation.include_tool_cloud = False" in icp_cfg
    assert "encoder_backend=icp requires" in exp_cfg
    assert "PretrainCfg.enabled currently supports only TCE with ActorCriticTG" in exp_cfg


def test_icp_separate_actor_critic_fusion_is_wired_to_policy():
    train_script = _source("scripts/train.py")
    runner_cfg = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/agents/config/rsl_rl_ppo_cfg.py"
    )
    actor = _source("rsl_rl/modules/actor_critic_icp.py")

    assert "\"separate_actor_critic_fusion\": rl.separate_actor_critic_fusion" in train_script
    assert "class ICPActorCriticCfg" in runner_cfg
    assert "separate_actor_critic_fusion: bool = bool(_policy(\"separate_actor_critic_fusion\", False))" in runner_cfg
    assert "separate_actor_critic_fusion: bool = False" in actor
    assert "self.separate_actor_critic_fusion = bool(separate_actor_critic_fusion)" in actor
    assert "self.critic_feature_fusion = copy.deepcopy(self.feature_fusion)" in actor
    assert "self.critic_state_cross = copy.deepcopy(self.state_cross)" in actor
    assert "branch: str = \"actor\"" in actor
    assert "branch=\"critic\"" in actor


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
        "object_velocity",
        "def build_state_cross_attention",
        "def build_mlp",
        "def build_fusion_mlp",
    ):
        assert token in common
    for actor in (tg_actor, p2v_actor):
        assert "from rsl_rl.modules.tg_policy_common import" in actor
        assert "center_clouds_by_bbox(" in actor
        assert "build_context_vector(parts)" in actor
        assert "build_fusion_mlp(" in actor
        assert "build_mlp(" in actor
        assert "strict=False" not in actor
    assert "build_state_cross_attention(" in p2v_actor
    assert "object_tokens = self._encode_single_cloud(object_cloud_rel" in p2v_actor
    assert "tool_tokens = self._encode_single_cloud(tool_cloud_rel" in p2v_actor
    assert "torch.cat([tool_tokens, object_tokens], dim=1)" in p2v_actor
    assert "extra_state = obs[:, offset:]" not in p2v_actor
    assert "PointcloudCentering" not in p2v_actor
    assert "random initialization" not in p2v_actor


def test_tg_policy_heads_share_mixin_noise_and_layout_helpers():
    common = _source("rsl_rl/modules/tg_policy_common.py")
    assert "def validate_observation_layout" in common
    assert "def initialize_action_noise" in common
    assert "Normal.set_default_validate_args(False)" in common

    duplicated_helpers = {
        "_get_features",
        "_action_std",
        "update_distribution",
        "act",
        "act_inference",
        "reset",
        "get_actions_log_prob",
        "evaluate",
        "get_cached_encoder_features",
        "act_from_cached_features",
        "evaluate_from_cached_features",
        "get_actions_log_prob_from_cached_features",
        "act_inference_from_cached_features",
        "action_mean",
        "action_std",
        "entropy",
    }
    actors = {
        "rsl_rl/modules/actor_critic_tg.py": "ActorCriticTG",
        "rsl_rl/modules/actor_critic_point2vec.py": "ActorCriticPoint2Vec",
        "rsl_rl/modules/actor_critic_tg_bimanual.py": "ActorCriticTGBimanual",
    }
    for rel_path, class_name in actors.items():
        source = _source(rel_path)
        class_def = _class_node(rel_path, class_name)
        assert "TGActorCriticHeadMixin" in {ast.unparse(base) for base in class_def.bases}
        declared = {
            node.name
            for node in class_def.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert duplicated_helpers.isdisjoint(declared)
        assert "initialize_action_noise(" in source
        assert "validate_observation_layout(" in source
        assert "from torch.distributions import Normal" not in source
        assert "Normal.set_default_validate_args(False)" not in source


def test_train_policy_params_payload_shapes_are_stable():
    from configs.config_exp import ExpCfg
    from scripts.train import _build_policy_params

    checkpoint = "/tmp/encoder.ckpt"
    shared_tg_keys = [
        "class_name",
        "num_points",
        "point_dim",
        "encoder_weights_path",
        "encoder_checkpoint_name",
        "encoder_checkpoint_schema",
        "encoder_checkpoint_adapter",
        "freeze_encoder",
        "freeze_point2vec",
        "separate_actor_critic_fusion",
        "sd_num_query",
        "sd_emb_dim",
        "relative_translation_query_tokens",
        "reuse_pretrain_pose_cross_attn",
        "sd_query_keys",
        "cross_attn_heads",
        "cross_attn_layers",
        "cross_attn_ff_dim",
        "cross_attn_dropout",
        "sd_cat_query",
        "sd_cat_ctx",
        "fusion_hidden_dims",
        "actor_hidden_dims",
        "critic_hidden_dims",
        "hand_state_dim",
        "robot_state_dim",
        "previous_action_dim",
        "relative_goal_dim",
        "object_velocity_dim",
        "physics_dim",
        "model_input_centering",
        "activation",
        "init_noise_std",
        "noise_std_type",
    ]
    tce_keys = shared_tg_keys + ["patch_size", "encoder_channel", "vit_depth", "vit_heads"]
    p2v_keys = shared_tg_keys + [
        "token_dim",
        "point2vec_ckpt_path",
        "tokenizer_num_groups",
        "tokenizer_group_size",
        "tokenizer_group_radius",
        "encoder_dim",
        "encoder_depth",
        "encoder_heads",
        "encoder_dropout",
        "encoder_attention_dropout",
        "encoder_drop_path_rate",
        "encoder_add_pos_at_every_layer",
        "train_transformations",
        "val_transformations",
    ]
    icp_keys = [
        "class_name",
        "num_points",
        "point_dim",
        "encoder_weights_path",
        "encoder_checkpoint_name",
        "encoder_checkpoint_schema",
        "encoder_checkpoint_adapter",
        "icp_weights_path",
        "icp_point_dim",
        "icp_num_points",
        "freeze_icp",
        "freeze_encoder",
        "separate_actor_critic_fusion",
        "sd_num_query",
        "sd_emb_dim",
        "relative_translation_query_tokens",
        "reuse_pretrain_pose_cross_attn",
        "sd_query_keys",
        "cross_attn_heads",
        "cross_attn_layers",
        "cross_attn_ff_dim",
        "cross_attn_dropout",
        "sd_cat_query",
        "sd_cat_ctx",
        "hand_state_dim",
        "robot_state_dim",
        "previous_action_dim",
        "relative_goal_dim",
        "object_velocity_dim",
        "physics_dim",
        "model_input_centering",
        "fusion_hidden_dims",
        "actor_hidden_dims",
        "critic_hidden_dims",
        "activation",
        "init_noise_std",
        "noise_std_type",
    ]

    def params_for(class_name: str) -> dict:
        cfg = ExpCfg()
        cfg.rl.actor_critic_class = class_name
        if class_name == "ActorCriticPoint2Vec":
            cfg.model.encoder_backend = "point2vec"
            cfg.model.pretrained_encoder.name = "point2vec"
            cfg.model.pretrained_encoder.schema = "point2vec"
            cfg.model.pretrained_encoder.adapter = "point2vec_native"
        elif class_name == "ActorCriticICP":
            cfg.model.encoder_backend = "icp"
            cfg.model.pretrained_encoder.name = "icp"
            cfg.model.pretrained_encoder.schema = "icp_legacy"
            cfg.model.pretrained_encoder.adapter = "icp_legacy"
        return _build_policy_params(cfg, checkpoint)

    for class_name in ("ActorCriticTG", "ActorCriticTGBimanual"):
        params = params_for(class_name)
        assert list(params) == tce_keys
        assert params["class_name"] == class_name
        assert params["encoder_weights_path"] == checkpoint
        assert params["freeze_point2vec"] is params["freeze_encoder"]
    p2v = params_for("ActorCriticPoint2Vec")
    assert list(p2v) == p2v_keys
    assert p2v["class_name"] == "ActorCriticPoint2Vec"
    assert p2v["encoder_weights_path"] == checkpoint
    assert p2v["point2vec_ckpt_path"] == checkpoint
    icp = params_for("ActorCriticICP")
    assert list(icp) == icp_keys
    assert icp["class_name"] == "ActorCriticICP"
    assert icp["encoder_weights_path"] == checkpoint
    assert icp["icp_weights_path"] == checkpoint
    assert "freeze_point2vec" not in icp


def test_unstable_task_adds_random_pose_dwell_success_and_velocity_observation():
    registration = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/__init__.py"
    )
    env = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/env_tool_unstable.py"
    )
    commands = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/commands.py"
    )
    rewards = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/rewards.py"
    )
    terminations = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/terminations.py"
    )
    obs = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/observations.py"
    )
    cfg = _source("configs/experiments/fork_unstable_diff_post.py")

    assert "id=\"tool-unstable-v0\"" in registration
    assert "class RandomPoseCommand" in commands
    assert "def _sample_uniform_quat_wxyz" in commands
    assert "object_velocity = ObsTerm(func=mdp.object_root_velocity)" in env
    assert "func=mdp.task_success_from_termination" in env
    assert "\"term_name\": \"reached\"" in env
    assert "func=mdp.object_within_goal_threshold" in env
    assert "func=mdp.object_reached_goal_dwell" in env
    assert "def object_root_velocity" in obs
    assert "def task_success_from_termination" in rewards
    assert "def object_within_goal_threshold" in rewards
    assert "def object_reached_goal_dwell" in terminations
    assert "EXP_CFG.pretrain_reuse = \"multitools_diff_post.py\"" in cfg
    assert "EXP_CFG.rl.isaac_task_id = \"tool-unstable-v0\"" in cfg
    assert "\"object_velocity\"" in cfg


def test_bimanual_unstable_task_is_additive_and_uses_three_cloud_layout():
    registration = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/__init__.py"
    )
    env = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/env_tool_bimanual_unstable.py"
    )
    obs = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/observations_bimanual.py"
    )
    rewards = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/mdp/rewards_bimanual.py"
    )
    actor = _source("rsl_rl/modules/actor_critic_tg_bimanual.py")
    common = _source("rsl_rl/modules/tg_bimanual_policy_common.py")
    cfg = _source("configs/experiments/bimanual_unstable_diff_post.py")

    assert "id=\"tool-bimanual-unstable-v0\"" in registration
    assert "class BimanualUnstableEnv" in env
    assert "mdp.RandomPoseCommandCfg" in env
    assert "robot_1" in env
    assert "robot_2" in env
    assert "tool1_cloud = ObsTerm" in env
    assert "tool2_cloud = ObsTerm" in env
    assert "object_velocity = ObsTerm(func=mdp.object_root_velocity)" in env
    assert "func=mdp.object_reached_goal_dwell" in env
    assert "func=mdp.bimanual_links_too_close" in env
    assert "func=mdp.bimanual_link_proximity_penalty" in env
    assert "def get_tool1_pointcloud_in_env_frame" in obs
    assert "def get_tool2_pointcloud_in_env_frame" in obs
    assert "def bimanual_object_goal_distance_tanh" in rewards
    assert "def bimanual_link_proximity_penalty" in rewards
    assert "class ActorCriticTGBimanual" in actor
    assert "class BimanualTCEPointCloudEncoder" in actor
    assert "expanded[1] = type_embed[0]" in actor
    assert "expanded[2] = type_embed[1]" in actor
    assert "class BimanualObservationLayout" in common
    assert "tool1_bbox_center - object_bbox_center" in common
    assert "tool2_bbox_center - object_bbox_center" in common
    assert "EXP_CFG.rl.actor_critic_class = \"ActorCriticTGBimanual\"" in cfg
    assert "EXP_CFG.rl.isaac_task_id = \"tool-bimanual-unstable-v0\"" in cfg
    assert "\"tool1_cloud_flat\"" in cfg
    assert "\"tool2_cloud_flat\"" in cfg


def test_policy_registry_and_isaac_bridge_are_whitelist_driven():
    modules = _source("rsl_rl/modules/__init__.py")
    agent_cfg = _source(
        "source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/"
        "isaaclab_nonprehensile/agents/config/rsl_rl_ppo_cfg.py"
    )

    assert "\"ActorCriticTG\": ActorCriticTG" in modules
    assert "\"ActorCriticTGBimanual\": ActorCriticTGBimanual" in modules
    assert "\"ActorCriticPoint2Vec\": ActorCriticPoint2Vec" in modules
    assert "\"ActorCriticICP\": ActorCriticICP" in modules
    assert "policy_class = eval" not in _source("rsl_rl/runners/on_policy_runner.py")
    assert "if _POLICY_CLASS_NAME == \"ActorCriticTG\"" in agent_cfg
    assert "if _POLICY_CLASS_NAME == \"ActorCriticTGBimanual\"" in agent_cfg
    assert "if _POLICY_CLASS_NAME == \"ActorCriticPoint2Vec\"" in agent_cfg
    assert "if _POLICY_CLASS_NAME == \"ActorCriticICP\"" in agent_cfg
    assert "class TGActorCriticCfg" in agent_cfg
    assert "class TGBimanualActorCriticCfg" in agent_cfg
    assert "class Point2VecActorCriticCfg" in agent_cfg
    assert "class ICPActorCriticCfg" in agent_cfg
    assert "eval(" not in agent_cfg
    assert "point2vec_ckpt_path" in agent_cfg
    assert "patch_size" in agent_cfg
    assert "/mnt/project" not in agent_cfg
