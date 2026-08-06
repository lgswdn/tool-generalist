"""RSL-RL runner config bridge backed by the experiment runtime spec.

The real Isaac/RSL-RL entrypoint must consume the ``rl_runtime_spec.json``
written by ``scripts/train.py``.  Missing or incompatible specs fail at import
time so policy/PPO dimensions cannot silently fall back to local defaults.
"""

from __future__ import annotations

from typing import Any

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from utils.experiment.rl_runtime_spec import (
    RUNTIME_SPEC_ENV_VAR,
    RUNTIME_SPEC_FILENAME,
    load_runtime_spec_from_env,
)


def _policy(name: str, default: Any = None) -> Any:
    return _POLICY.get(name, default)


def _ppo(name: str, default: Any = None) -> Any:
    return _PPO.get(name, default)


def _policy_for_class(name: str, class_name: str, default: Any) -> Any:
    """Read a policy field that is required only for one actor class.

    IsaacLab configclass evaluates every class body at import time, including
    inactive policy classes.  Do not let Point2Vec-only fields break TG runs,
    but still fail-fast if Point2Vec is the selected actor and a required field
    is missing.
    """

    value = _POLICY.get(name)
    if value is not None:
        return value
    if _POLICY_CLASS_NAME == class_name:
        raise RuntimeError(
            f"{class_name} requires policy_params.{name} in rl_runtime_spec.json"
        )
    return default


_RUNTIME_SPEC = load_runtime_spec_from_env()
_POLICY = _RUNTIME_SPEC["policy_params"]
_PPO = _RUNTIME_SPEC["ppo_params"]
_POLICY_CLASS_NAME = str(_RUNTIME_SPEC["actor_critic_class"])


@configclass
class TGActorCriticCfg:
    """Policy config sourced from rl_runtime_spec.json."""

    class_name: str = "ActorCriticTG"

    num_points: int = int(_policy("num_points"))
    point_dim: int = int(_policy("point_dim"))
    encoder_backend: str = str(_policy("encoder_backend", "tce"))
    num_patches: int = int(_policy("num_patches", 16))
    patch_size: int = int(_policy("patch_size", 32))
    encoder_channel: int = int(_policy("encoder_channel", 128))
    vit_depth: int = int(_policy("vit_depth", 12))
    vit_heads: int = int(_policy("vit_heads", 4))
    vit_attention_mode: str = str(_policy("vit_attention_mode", "joint_self"))
    kinematic_conditioning: bool = bool(_policy("kinematic_conditioning", False))
    kinematic_attention_layers: int = int(
        _policy("kinematic_attention_layers", 1)
    )
    oracle_contact_eps: float = float(_policy("oracle_contact_eps", 0.002))
    oracle_center_scale_m: float = float(_policy("oracle_center_scale_m", 0.30))
    oracle_distance_scale_m: float = float(_policy("oracle_distance_scale_m", 0.10))
    oracle_patch_relative_scale_m: float = float(
        _policy("oracle_patch_relative_scale_m", 0.05)
    )
    oracle_log_distance_resolution_m: float = float(
        _policy("oracle_log_distance_resolution_m", 0.005)
    )
    oracle_log_distance_cap_m: float = float(
        _policy("oracle_log_distance_cap_m", 0.05)
    )
    oracle_normalization_clip: float = float(
        _policy("oracle_normalization_clip", 5.0)
    )
    oracle_pointmesh_coordinate_scale_m: float = float(
        _policy("oracle_pointmesh_coordinate_scale_m", 0.30)
    )
    oracle_pointmesh_distance_scale_m: float = float(
        _policy("oracle_pointmesh_distance_scale_m", 0.10)
    )
    oracle_pointmesh_normalization_clip: float = float(
        _policy("oracle_pointmesh_normalization_clip", 5.0)
    )
    oracle_pointcloud_nearest_frame_batch_size: int = int(
        _policy("oracle_pointcloud_nearest_frame_batch_size", 64)
    )
    oracle_pointcloud_feature_mode: str = str(
        _policy("oracle_pointcloud_feature_mode", "fast11")
    )
    oracle_pointcloud_load_fitted_weights: bool = bool(
        _policy("oracle_pointcloud_load_fitted_weights", True)
    )
    oracle_pointcloud_use_rank10_bottleneck: bool = bool(
        _policy("oracle_pointcloud_use_rank10_bottleneck", True)
    )
    oracle_pointcloud_token_mode: str = str(
        _policy("oracle_pointcloud_token_mode", "patches")
    )
    oracle_pointcloud_input_normalization: str = str(
        _policy("oracle_pointcloud_input_normalization", "identity")
    )
    oracle_pointcloud_checkpoint_adapter: str = str(
        _policy(
            "oracle_pointcloud_checkpoint_adapter",
            "oracle_pointcloud_pointnet_strict",
        )
    )
    unicorn_token_source: str = str(_policy("unicorn_token_source", "encoder"))
    encoder_token_pca_rank: int = int(
        _policy("encoder_token_pca_rank", encoder_channel)
    )
    encoder_token_pca_path: str | None = _policy("encoder_token_pca_path")
    encoder_token_bottleneck_rank: int = int(
        _policy("encoder_token_bottleneck_rank", encoder_channel)
    )
    encoder_token_bottleneck_pca_path: str | None = _policy(
        "encoder_token_bottleneck_pca_path"
    )

    encoder_weights_path: str | None = _policy("encoder_weights_path")
    freeze_encoder: bool = bool(_policy("freeze_encoder"))
    separate_actor_critic_fusion: bool = bool(_policy("separate_actor_critic_fusion", False))

    use_learnable_query_tokens: bool = bool(_policy("use_learnable_query_tokens", False))
    sd_num_query: int = int(_policy("sd_num_query"))
    sd_num_query_object: int | None = None
    sd_emb_dim: int = int(_policy("sd_emb_dim"))
    relative_translation_query_tokens: int = int(_policy("relative_translation_query_tokens", 2))
    reuse_pretrain_pose_cross_attn: bool = bool(_policy("reuse_pretrain_pose_cross_attn", False))
    sd_cat_query: bool = bool(_policy("sd_cat_query", False))
    sd_cat_ctx: bool = bool(_policy("sd_cat_ctx", True))
    sd_query_keys: tuple[str, ...] = tuple(_policy("sd_query_keys", ("context",)))

    num_query_object_tokens: int | None = None
    num_query_tokens: int = int(_policy("sd_num_query"))
    cross_attn_heads: int = int(_policy("cross_attn_heads"))
    cross_attn_layers: int = int(_policy("cross_attn_layers"))
    cross_attn_ff_dim: int | None = _policy("cross_attn_ff_dim")
    cross_attn_dropout: float = float(_policy("cross_attn_dropout", 0.0))

    hand_state_dim: int = int(_policy("hand_state_dim"))
    robot_state_dim: int = int(_policy("robot_state_dim"))
    previous_action_dim: int = int(_policy("previous_action_dim"))
    relative_goal_dim: int = int(_policy("relative_goal_dim"))
    object_velocity_dim: int = int(_policy("object_velocity_dim", 0))
    task_embedding_dim: int = int(_policy("task_embedding_dim", 0))
    physics_dim: int = int(_policy("physics_dim"))
    model_input_centering: str = str(_policy("model_input_centering", "bbox_center"))

    fusion_hidden_dims: list[int] = list(_policy("fusion_hidden_dims"))
    actor_hidden_dims: list[int] = list(_policy("actor_hidden_dims"))
    critic_hidden_dims: list[int] = list(_policy("critic_hidden_dims"))

    activation: str = str(_policy("activation", "elu"))
    init_noise_std: float = float(_policy("init_noise_std", 1.0))
    noise_std_type: str = str(_policy("noise_std_type", "scalar"))


@configclass
class TGSMActorCriticCfg(TGActorCriticCfg):
    """Soft-Module TCE policy config sourced from rl_runtime_spec.json."""

    class_name: str = "ActorCriticTGSM"

    task_embedding_dim: int = int(_policy("task_embedding_dim", 2))
    sm_num_layers: int = int(_policy("sm_num_layers", 2))
    sm_num_modules: int = int(_policy("sm_num_modules", 4))
    sm_module_hidden: int = int(_policy("sm_module_hidden", 128))
    sm_gating_hidden: int = int(_policy("sm_gating_hidden", 128))
    sm_num_gating_layers: int = int(_policy("sm_num_gating_layers", 1))
    sm_cond_ob: bool = bool(_policy("sm_cond_ob", True))
    sm_add_bn: bool = bool(_policy("sm_add_bn", False))


@configclass
class TGHAMNetActorCriticCfg(TGActorCriticCfg):
    """HAMNet modular-hypernetwork policy using the existing TCE encoder."""

    class_name: str = "ActorCriticTGHAMNet"

    hamnet_num_modules: int = int(_policy("hamnet_num_modules", 4))
    hamnet_hidden_dims: list[int] = list(
        _policy("hamnet_hidden_dims", [256, 128, 128, 64])
    )
    hamnet_router_hidden_dims: list[int] = list(
        _policy("hamnet_router_hidden_dims", [256, 256])
    )


@configclass
class TGOutputGateActorCriticCfg(TGActorCriticCfg):
    """Two-expert output-gated TCE policy config sourced from rl_runtime_spec.json."""

    class_name: str = "ActorCriticTGOutputGate"

    expert_a_checkpoint: str = str(_policy("expert_a_checkpoint"))
    expert_b_checkpoint: str = str(_policy("expert_b_checkpoint"))
    output_gate_freeze_experts: bool = bool(_policy("output_gate_freeze_experts", True))
    output_gate_hidden_dims: list[int] = list(_policy("output_gate_hidden_dims", [64]))
    output_gate_initial_expert_a_weight: float = float(_policy("output_gate_initial_expert_a_weight", 0.8))
    output_gate_per_action: bool = bool(_policy("output_gate_per_action", False))


@configclass
class TGBimanualActorCriticCfg(TGActorCriticCfg):
    """Bimanual TCE policy config sourced from rl_runtime_spec.json."""

    class_name: str = "ActorCriticTGBimanual"


@configclass
class TGUnicornActorCriticCfg(TGActorCriticCfg):
    """Single-arm UniCORN policy config sourced from rl_runtime_spec.json."""

    class_name: str = "ActorCriticTGUnicorn"


@configclass
class Point2VecActorCriticCfg:
    """Point2Vec policy config sourced from rl_runtime_spec.json."""

    class_name: str = "ActorCriticPoint2Vec"

    num_points: int = int(_policy("num_points"))
    point_dim: int = int(_policy("point_dim"))
    encoder_weights_path: str = str(_policy("encoder_weights_path"))
    point2vec_ckpt_path: str = str(_policy_for_class("point2vec_ckpt_path", "ActorCriticPoint2Vec", ""))
    freeze_encoder: bool = bool(_policy("freeze_encoder"))
    freeze_point2vec: bool = bool(_policy("freeze_point2vec", _policy("freeze_encoder")))
    separate_actor_critic_fusion: bool = bool(_policy("separate_actor_critic_fusion", False))

    tokenizer_num_groups: int = int(_policy_for_class("tokenizer_num_groups", "ActorCriticPoint2Vec", 1))
    tokenizer_group_size: int = int(_policy_for_class("tokenizer_group_size", "ActorCriticPoint2Vec", 1))
    tokenizer_group_radius: float | None = _policy("tokenizer_group_radius")
    encoder_dim: int = int(_policy_for_class("encoder_dim", "ActorCriticPoint2Vec", 1))
    encoder_depth: int = int(_policy_for_class("encoder_depth", "ActorCriticPoint2Vec", 1))
    encoder_heads: int = int(_policy_for_class("encoder_heads", "ActorCriticPoint2Vec", 1))
    encoder_dropout: float = float(_policy("encoder_dropout", 0.0))
    encoder_attention_dropout: float = float(_policy("encoder_attention_dropout", 0.0))
    encoder_drop_path_rate: float = float(_policy("encoder_drop_path_rate", 0.2))
    encoder_add_pos_at_every_layer: bool = bool(_policy("encoder_add_pos_at_every_layer", True))
    train_transformations: list[str] = list(_policy("train_transformations", ("unit_sphere",)))
    val_transformations: list[str] = list(_policy("val_transformations", ("unit_sphere",)))

    sd_num_query: int = int(_policy("sd_num_query"))
    sd_emb_dim: int = int(_policy("sd_emb_dim"))
    sd_cat_query: bool = bool(_policy("sd_cat_query", False))
    sd_cat_ctx: bool = bool(_policy("sd_cat_ctx", True))
    sd_query_keys: tuple[str, ...] = tuple(_policy("sd_query_keys", ("context",)))
    cross_attn_heads: int = int(_policy("cross_attn_heads"))
    cross_attn_layers: int = int(_policy("cross_attn_layers"))
    cross_attn_ff_dim: int | None = _policy("cross_attn_ff_dim")
    cross_attn_dropout: float = float(_policy("cross_attn_dropout", 0.0))

    hand_state_dim: int = int(_policy("hand_state_dim"))
    robot_state_dim: int = int(_policy("robot_state_dim"))
    previous_action_dim: int = int(_policy("previous_action_dim"))
    relative_goal_dim: int = int(_policy("relative_goal_dim"))
    object_velocity_dim: int = int(_policy("object_velocity_dim", 0))
    physics_dim: int = int(_policy("physics_dim"))

    fusion_hidden_dims: list[int] = list(_policy("fusion_hidden_dims"))
    actor_hidden_dims: list[int] = list(_policy("actor_hidden_dims"))
    critic_hidden_dims: list[int] = list(_policy("critic_hidden_dims"))

    activation: str = str(_policy("activation", "elu"))
    init_noise_std: float = float(_policy("init_noise_std", 1.0))
    noise_std_type: str = str(_policy("noise_std_type", "scalar"))


@configclass
class ICPActorCriticCfg:
    """Legacy ICP policy config sourced from rl_runtime_spec.json."""

    class_name: str = "ActorCriticICP"

    icp_weights_path: str | None = _policy("icp_weights_path")
    freeze_icp: bool = bool(_policy("freeze_icp", True))
    icp_point_dim: int = int(_policy("icp_point_dim", 3))
    icp_num_points: int = int(_policy("icp_num_points", 512))
    separate_actor_critic_fusion: bool = bool(_policy("separate_actor_critic_fusion", False))

    fusion_hidden_dims: list[int] = list(_policy("fusion_hidden_dims", [512, 256, 128]))
    actor_hidden_dims: list[int] = list(_policy("actor_hidden_dims", [64]))
    critic_hidden_dims: list[int] = list(_policy("critic_hidden_dims", [64]))

    activation: str = str(_policy("activation", "elu"))
    init_noise_std: float = float(_policy("init_noise_std", 1.0))
    noise_std_type: str = str(_policy("noise_std_type", "scalar"))


def _policy_cfg():
    if _POLICY_CLASS_NAME == "ActorCriticTG":
        return TGActorCriticCfg()
    if _POLICY_CLASS_NAME == "ActorCriticTGOutputGate":
        return TGOutputGateActorCriticCfg()
    if _POLICY_CLASS_NAME == "ActorCriticTGSM":
        return TGSMActorCriticCfg()
    if _POLICY_CLASS_NAME == "ActorCriticTGHAMNet":
        return TGHAMNetActorCriticCfg()
    if _POLICY_CLASS_NAME == "ActorCriticTGUnicorn":
        return TGUnicornActorCriticCfg()
    if _POLICY_CLASS_NAME == "ActorCriticTGBimanual":
        return TGBimanualActorCriticCfg()
    if _POLICY_CLASS_NAME == "ActorCriticPoint2Vec":
        return Point2VecActorCriticCfg()
    if _POLICY_CLASS_NAME == "ActorCriticICP":
        return ICPActorCriticCfg()
    raise RuntimeError(f"Unsupported actor_critic_class in runtime spec: {_POLICY_CLASS_NAME!r}")


@configclass
class TGPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO config sourced from rl_runtime_spec.json."""

    num_steps_per_env = int(_RUNTIME_SPEC["num_steps_per_env"])
    max_iterations = int(_RUNTIME_SPEC["max_iterations"])
    save_interval = int(_ppo("save_interval"))

    _launch_params = _RUNTIME_SPEC.get("launch_params") or {}
    experiment_name = str(
        _launch_params.get("wandb_project")
        or _RUNTIME_SPEC.get("task_id")
        or "tool_generalist_rl"
    )
    run_name = str(
        _launch_params.get("run_name")
        or _RUNTIME_SPEC.get("artifact_dir")
        or experiment_name
    )
    empirical_normalization = False
    wandb_upload_files = bool(_launch_params.get("wandb_upload_files", False))
    policy = _policy_cfg()

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=float(_ppo("value_loss_coef")),
        use_clipped_value_loss=bool(_ppo("use_clipped_value_loss")),
        clip_param=float(_ppo("clip_param")),
        entropy_coef=float(_ppo("entropy_coef")),
        num_learning_epochs=int(_ppo("num_learning_epochs")),
        num_mini_batches=int(_ppo("num_mini_batches")),
        learning_rate=float(_ppo("learning_rate")),
        schedule=str(_ppo("schedule")),
        gamma=float(_ppo("gamma")),
        lam=float(_ppo("lam")),
        desired_kl=float(_ppo("desired_kl")),
        max_grad_norm=float(_ppo("max_grad_norm")),
    )
