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
    patch_size: int = int(_policy("patch_size", 32))
    encoder_channel: int = int(_policy("encoder_channel", 128))
    vit_depth: int = int(_policy("vit_depth", 12))
    vit_heads: int = int(_policy("vit_heads", 4))

    encoder_weights_path: str = str(_policy("encoder_weights_path"))
    freeze_encoder: bool = bool(_policy("freeze_encoder"))

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
    physics_dim: int = int(_policy("physics_dim"))
    model_input_centering: str = str(_policy("model_input_centering", "bbox_center"))

    fusion_hidden_dims: list[int] = list(_policy("fusion_hidden_dims"))
    actor_hidden_dims: list[int] = list(_policy("actor_hidden_dims"))
    critic_hidden_dims: list[int] = list(_policy("critic_hidden_dims"))

    activation: str = str(_policy("activation", "elu"))
    init_noise_std: float = float(_policy("init_noise_std", 1.0))
    noise_std_type: str = str(_policy("noise_std_type", "scalar"))


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
    physics_dim: int = int(_policy("physics_dim"))

    fusion_hidden_dims: list[int] = list(_policy("fusion_hidden_dims"))
    actor_hidden_dims: list[int] = list(_policy("actor_hidden_dims"))
    critic_hidden_dims: list[int] = list(_policy("critic_hidden_dims"))

    activation: str = str(_policy("activation", "elu"))
    init_noise_std: float = float(_policy("init_noise_std", 1.0))
    noise_std_type: str = str(_policy("noise_std_type", "scalar"))


def _policy_cfg():
    if _POLICY_CLASS_NAME == "ActorCriticTG":
        return TGActorCriticCfg()
    if _POLICY_CLASS_NAME == "ActorCriticPoint2Vec":
        return Point2VecActorCriticCfg()
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
