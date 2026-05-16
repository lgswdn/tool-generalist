"""Lightweight RL runtime-spec loader shared by Isaac and RSL-RL bridges."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


RUNTIME_SPEC_ENV_VAR = "TOOL_GENERALIST_RL_RUNTIME_SPEC"
RUNTIME_SPEC_FILENAME = "rl_runtime_spec.json"
SUPPORTED_POLICY_CLASSES = {"ActorCriticTG", "ActorCriticPoint2Vec"}


def load_runtime_spec_from_env() -> dict[str, Any]:
    raw_path = os.environ.get(RUNTIME_SPEC_ENV_VAR)
    if not raw_path:
        raise RuntimeError(
            f"{RUNTIME_SPEC_ENV_VAR} must point to the {RUNTIME_SPEC_FILENAME} "
            "written by scripts/train.py before importing the Isaac/RSL-RL bridge."
        )
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"RL runtime spec does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    if not isinstance(spec, dict):
        raise RuntimeError(f"RL runtime spec must contain a JSON object: {path}")
    validate_runtime_spec(spec, path)
    return spec


def validate_runtime_spec(spec: Mapping[str, Any], path: str | Path = "<runtime-spec>") -> None:
    required = {
        "actor_critic_class",
        "algorithm_class",
        "seed",
        "num_gpus",
        "num_envs",
        "action_dim",
        "physics_dim",
        "observation_dim",
        "physics_observation_fields",
        "policy_params",
        "ppo_params",
        "action_params",
        "observation_params",
        "launch_params",
        "table_params",
        "object_pose_sampling_params",
        "asset_assignment_params",
        "env_params",
        "reward_params",
        "domain_randomization_params",
        "encoder_checkpoint",
        "paths_yaml",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise RuntimeError(f"RL runtime spec missing required fields {missing}: {path}")
    if spec["actor_critic_class"] not in SUPPORTED_POLICY_CLASSES:
        raise RuntimeError(
            f"Unsupported actor_critic_class={spec['actor_critic_class']!r}; "
            f"supported policy classes are {sorted(SUPPORTED_POLICY_CLASSES)}."
        )
    if spec["algorithm_class"] != "PPO":
        raise RuntimeError(
            f"Unsupported algorithm_class={spec['algorithm_class']!r}; only PPO is supported."
        )
    if isinstance(spec["seed"], bool):
        raise RuntimeError("RL runtime spec seed must be an integer")
    try:
        seed = int(spec["seed"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("RL runtime spec seed must be an integer") from exc
    if seed < 0:
        raise RuntimeError("RL runtime spec seed must be >= 0")
    try:
        num_gpus = int(spec["num_gpus"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("RL runtime spec num_gpus must be an integer") from exc
    if num_gpus < 0:
        raise RuntimeError("RL runtime spec num_gpus must be >= 0")
    try:
        num_envs = int(spec["num_envs"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("RL runtime spec num_envs must be an integer") from exc
    if num_envs <= 0:
        raise RuntimeError("RL runtime spec num_envs must be > 0")
    launch = _require_mapping(spec, "launch_params", path)
    if bool(launch.get("distributed", False)) and num_gpus <= 0:
        raise RuntimeError("RL distributed launch requires num_gpus > 0")
    if not spec["encoder_checkpoint"]:
        raise RuntimeError(
            f"{spec['actor_critic_class']} requires encoder_checkpoint in RL runtime spec"
        )
    if len(tuple(spec["physics_observation_fields"])) != int(spec["physics_dim"]):
        raise RuntimeError(
            "RL runtime spec physics_dim must match len(physics_observation_fields)"
        )
    table = _require_mapping(spec, "table_params", path)
    pose_sampling = _require_mapping(spec, "object_pose_sampling_params", path)
    for key in ("initial_position_range", "xy_offset_range"):
        if key not in pose_sampling:
            raise RuntimeError(f"RL runtime spec object_pose_sampling_params missing {key!r}")
        try:
            value = float(pose_sampling[key])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"RL runtime spec object_pose_sampling_params.{key} must be a number") from exc
        if value < 0.0:
            raise RuntimeError(f"RL runtime spec object_pose_sampling_params.{key} must be >= 0")

    asset_assignment = _require_mapping(spec, "asset_assignment_params", path)
    seed = asset_assignment.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RuntimeError("RL runtime spec asset_assignment_params.seed must be an int")
    if seed < 0:
        raise RuntimeError("RL runtime spec asset_assignment_params.seed must be >= 0")
    for key in ("randomize_tool_assignment", "randomize_object_assignment"):
        if not isinstance(asset_assignment.get(key), bool):
            raise RuntimeError(f"RL runtime spec asset_assignment_params.{key} must be a bool")

    physics_fields = tuple(str(item) for item in spec["physics_observation_fields"])
    if bool(table.get("enabled", False)) and any(
        field.startswith("ground_") for field in physics_fields
    ):
        raise RuntimeError(
            "RL runtime spec cannot request ground_* physics fields when table is "
            "enabled because the ground plane is removed."
        )

    policy = _require_mapping(spec, "policy_params", path)
    if policy.get("class_name") != spec["actor_critic_class"]:
        raise RuntimeError("RL runtime spec policy_params.class_name must match actor_critic_class")

    required_policy = {
        "num_points",
        "point_dim",
        "encoder_weights_path",
        "sd_num_query",
        "sd_emb_dim",
        "relative_translation_query_tokens",
        "reuse_pretrain_pose_cross_attn",
        "cross_attn_heads",
        "cross_attn_layers",
        "sd_cat_query",
        "sd_cat_ctx",
        "fusion_hidden_dims",
        "actor_hidden_dims",
        "critic_hidden_dims",
        "hand_state_dim",
        "robot_state_dim",
        "previous_action_dim",
        "relative_goal_dim",
        "physics_dim",
        "model_input_centering",
        "activation",
        "init_noise_std",
        "noise_std_type",
    }
    if spec["actor_critic_class"] == "ActorCriticTG":
        required_policy.update(
            {
                "patch_size",
                "encoder_channel",
                "vit_depth",
                "vit_heads",
            }
        )
    elif spec["actor_critic_class"] == "ActorCriticPoint2Vec":
        required_policy.update(
            {
                "point2vec_ckpt_path",
                "tokenizer_num_groups",
                "tokenizer_group_size",
                "encoder_dim",
                "encoder_depth",
                "encoder_heads",
                "train_transformations",
                "val_transformations",
            }
        )
    if policy.get("encoder_weights_path") != spec["encoder_checkpoint"]:
        raise RuntimeError("policy_params.encoder_weights_path must match encoder_checkpoint")
    if (
        spec["actor_critic_class"] == "ActorCriticPoint2Vec"
        and policy.get("point2vec_ckpt_path") != spec["encoder_checkpoint"]
    ):
        raise RuntimeError("policy_params.point2vec_ckpt_path must match encoder_checkpoint")
    missing_policy = sorted(required_policy.difference(policy))
    if missing_policy:
        raise RuntimeError(f"RL runtime spec policy_params missing fields: {missing_policy}")

    ppo = _require_mapping(spec, "ppo_params", path)
    required_ppo = {
        "save_interval",
        "value_loss_coef",
        "use_clipped_value_loss",
        "clip_param",
        "entropy_coef",
        "num_learning_epochs",
        "num_mini_batches",
        "learning_rate",
        "schedule",
        "gamma",
        "lam",
        "desired_kl",
        "max_grad_norm",
    }
    missing_ppo = sorted(required_ppo.difference(ppo))
    if missing_ppo:
        raise RuntimeError(f"RL runtime spec ppo_params missing fields: {missing_ppo}")

    for key in (
        "action_params",
        "observation_params",
        "env_params",
        "reward_params",
        "domain_randomization_params",
    ):
        _require_mapping(spec, key, path)


def runtime_spec_contract(spec: Mapping[str, Any]) -> SimpleNamespace:
    """Return attr-style config used by legacy Isaac cfg declarations."""

    return _to_namespace(
        {
            "action": spec["action_params"],
            "observation": spec["observation_params"],
            "launch": spec["launch_params"],
            "table": spec["table_params"],
            "object_pose_sampling": spec["object_pose_sampling_params"],
            "asset_assignment": spec["asset_assignment_params"],
            "env": spec["env_params"],
            "reward": spec["reward_params"],
            "domain_randomization": spec["domain_randomization_params"],
            "physics_observation_fields": tuple(spec["physics_observation_fields"]),
            "seed": spec["seed"],
            "num_gpus": spec["num_gpus"],
            "num_envs": spec["num_envs"],
            "action_dim": spec["action_dim"],
            "physics_dim": spec["physics_dim"],
            "observation_dim": spec["observation_dim"],
        }
    )


def _require_mapping(spec: Mapping[str, Any], key: str, path: str | Path) -> Mapping[str, Any]:
    value = spec.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"RL runtime spec {key} must be an object: {path}")
    return value


def _to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{str(k): _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value
