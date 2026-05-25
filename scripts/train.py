"""Config-only RL training wrapper.

This module intentionally has no top-level Isaac, torch, or rsl_rl imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from configs.config_exp import ExpCfg
from utils.config.paths import ProjectPaths
from utils.experiment.effective_paths import materialize_runtime_paths_yaml
from utils.experiment.rl_runtime_spec import (
    RUNTIME_SPEC_ENV_VAR,
    RUNTIME_SPEC_FILENAME,
    validate_runtime_spec,
)
from utils.io import write_json


@dataclass(frozen=True)
class RLRuntimeSpec:
    mode: str
    seed: int
    task_id: str | None
    rsl_rl_cfg_entry_point: str | None
    train_entrypoint: str
    actor_critic_class: str
    algorithm_class: str
    num_gpus: int
    num_envs: int
    max_iterations: int
    num_steps_per_env: int
    action_dim: int
    physics_dim: int
    observation_dim: int
    observation_layout: list[str]
    physics_observation_fields: list[str]
    policy_params: dict[str, Any]
    ppo_params: dict[str, Any]
    action_params: dict[str, Any]
    observation_params: dict[str, Any]
    launch_params: dict[str, Any]
    table_params: dict[str, Any]
    object_pose_sampling_params: dict[str, Any]
    curriculum_params: dict[str, Any]
    asset_assignment_params: dict[str, Any]
    env_params: dict[str, Any]
    reward_params: dict[str, Any]
    domain_randomization_params: dict[str, Any]
    encoder_checkpoint: str | None
    freeze_encoder: bool
    artifact_dir: str
    paths_yaml: str
    runtime_spec_path: str
    runtime_spec_env_var: str


def build_rl_runtime_spec(
    exp_cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: str | Path,
    *,
    mode: str = "train",
    encoder_checkpoint_override: str | None = None,
) -> RLRuntimeSpec:
    rl = exp_cfg.rl
    checkpoint = resolve_encoder_checkpoint(
        exp_cfg,
        encoder_checkpoint_override=encoder_checkpoint_override,
    )
    policy_params = _build_policy_params(exp_cfg, checkpoint)
    runtime_paths_yaml = materialize_runtime_paths_yaml(
        exp_cfg,
        paths,
        Path(artifact_dir) / "paths.runtime.yaml",
    )
    return RLRuntimeSpec(
        mode=mode,
        seed=exp_cfg.general.seed,
        task_id=rl.resolved_task_id(),
        rsl_rl_cfg_entry_point=rl.rsl_rl_cfg_entry_point,
        train_entrypoint=rl.rsl_rl_train_entrypoint,
        actor_critic_class=rl.actor_critic_class,
        algorithm_class=rl.algorithm_class,
        num_gpus=exp_cfg.num_gpus,
        num_envs=rl.env.num_envs,
        max_iterations=rl.ppo.max_iterations,
        num_steps_per_env=rl.ppo.num_steps_per_env,
        action_dim=rl.effective_action_dim,
        physics_dim=rl.effective_physics_dim,
        observation_dim=rl.effective_observation_dim,
        observation_layout=list(rl.observation.layout),
        physics_observation_fields=list(rl.physics_observation_fields),
        policy_params=policy_params,
        ppo_params=asdict(rl.ppo),
        action_params=asdict(rl.action),
        observation_params=asdict(rl.observation),
        launch_params=asdict(rl.launch),
        table_params=asdict(rl.table),
        object_pose_sampling_params=asdict(rl.object_pose_sampling),
        curriculum_params=asdict(rl.curriculum),
        asset_assignment_params={
            "seed": exp_cfg.general.seed,
            "randomize_tool_assignment": exp_cfg.general.randomize_tool_assignment,
            "randomize_object_assignment": exp_cfg.general.randomize_object_assignment,
        },
        env_params=asdict(rl.env),
        reward_params=asdict(rl.reward),
        domain_randomization_params=asdict(rl.domain_randomization),
        encoder_checkpoint=checkpoint,
        freeze_encoder=rl.freeze_encoder,
        artifact_dir=str(Path(artifact_dir)),
        paths_yaml=str(runtime_paths_yaml),
        runtime_spec_path=str(Path(artifact_dir) / RUNTIME_SPEC_FILENAME),
        runtime_spec_env_var=RUNTIME_SPEC_ENV_VAR,
    )


def _build_policy_params(exp_cfg: ExpCfg, checkpoint: str | None) -> dict[str, Any]:
    rl = exp_cfg.rl
    model = exp_cfg.model
    pretrained = model.pretrained_encoder
    if rl.actor_critic_class == "ActorCriticICP":
        return {
            "class_name": "ActorCriticICP",
            "num_points": rl.observation.num_points,
            "point_dim": rl.observation.point_dim,
            "encoder_weights_path": checkpoint,
            "encoder_checkpoint_name": pretrained.name,
            "encoder_checkpoint_schema": pretrained.schema,
            "encoder_checkpoint_adapter": pretrained.adapter,
            "icp_weights_path": checkpoint,
            "icp_point_dim": rl.observation.point_dim,
            "icp_num_points": rl.observation.num_points,
            "freeze_icp": rl.freeze_encoder,
            "freeze_encoder": rl.freeze_encoder,
            "separate_actor_critic_fusion": rl.separate_actor_critic_fusion,
            "sd_num_query": model.policy_fusion.sd_num_query,
            "sd_emb_dim": model.policy_fusion.query_dim,
            "relative_translation_query_tokens": model.policy_fusion.relative_translation_query_tokens,
            "reuse_pretrain_pose_cross_attn": model.policy_fusion.reuse_pretrain_pose_cross_attn,
            "sd_query_keys": ("context",),
            "cross_attn_heads": model.policy_fusion.cross_attn_heads,
            "cross_attn_layers": model.policy_fusion.cross_attn_layers,
            "cross_attn_ff_dim": None,
            "cross_attn_dropout": 0.0,
            "sd_cat_query": False,
            "sd_cat_ctx": True,
            "hand_state_dim": rl.observation.hand_state_dim,
            "robot_state_dim": rl.observation.robot_state_dim,
            "previous_action_dim": rl.effective_action_dim,
            "relative_goal_dim": rl.observation.relative_goal_dim,
            "object_velocity_dim": rl.observation.object_velocity_dim,
            "physics_dim": rl.effective_physics_dim,
            "model_input_centering": rl.observation.model_input_centering,
            "fusion_hidden_dims": list(model.policy_fusion.fusion_hidden_dims),
            "actor_hidden_dims": list(model.policy_fusion.actor_hidden_dims),
            "critic_hidden_dims": list(model.policy_fusion.critic_hidden_dims),
            "activation": "elu",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
        }

    encoder = model.encoder
    common = {
        "class_name": rl.actor_critic_class,
        "num_points": encoder.num_points,
        "point_dim": rl.observation.point_dim,
        "encoder_weights_path": checkpoint,
        "encoder_checkpoint_name": pretrained.name,
        "encoder_checkpoint_schema": pretrained.schema,
        "encoder_checkpoint_adapter": pretrained.adapter,
        "freeze_encoder": rl.freeze_encoder,
        "freeze_point2vec": rl.freeze_encoder,
        "separate_actor_critic_fusion": rl.separate_actor_critic_fusion,
        "sd_num_query": model.policy_fusion.sd_num_query,
        "sd_emb_dim": model.policy_fusion.query_dim,
        "relative_translation_query_tokens": model.policy_fusion.relative_translation_query_tokens,
        "reuse_pretrain_pose_cross_attn": model.policy_fusion.reuse_pretrain_pose_cross_attn,
        "sd_query_keys": ("context",),
        "cross_attn_heads": model.policy_fusion.cross_attn_heads,
        "cross_attn_layers": model.policy_fusion.cross_attn_layers,
        "cross_attn_ff_dim": None,
        "cross_attn_dropout": 0.0,
        "sd_cat_query": False,
        "sd_cat_ctx": True,
        "fusion_hidden_dims": list(model.policy_fusion.fusion_hidden_dims),
        "actor_hidden_dims": list(model.policy_fusion.actor_hidden_dims),
        "critic_hidden_dims": list(model.policy_fusion.critic_hidden_dims),
        "hand_state_dim": rl.observation.hand_state_dim,
        "robot_state_dim": rl.observation.robot_state_dim,
        "previous_action_dim": rl.effective_action_dim,
        "relative_goal_dim": rl.observation.relative_goal_dim,
        "object_velocity_dim": rl.observation.object_velocity_dim,
        "physics_dim": rl.effective_physics_dim,
        "model_input_centering": rl.observation.model_input_centering,
        "activation": "elu",
        "init_noise_std": 1.0,
        "noise_std_type": "scalar",
    }
    if rl.actor_critic_class in {"ActorCriticTG", "ActorCriticTGBimanual"}:
        tce = model.tce
        common.update(
            {
                "patch_size": tce.patch_size,
                "encoder_channel": tce.encoder_channel,
                "vit_depth": tce.vit_depth,
                "vit_heads": tce.vit_heads,
            }
        )
        return common
    if rl.actor_critic_class == "ActorCriticPoint2Vec":
        p2v = model.p2v
        common.update(
            {
                "token_dim": p2v.token_dim,
                "point2vec_ckpt_path": checkpoint,
                "tokenizer_num_groups": p2v.tokenizer_num_groups,
                "tokenizer_group_size": p2v.tokenizer_group_size,
                "tokenizer_group_radius": p2v.tokenizer_group_radius,
                "encoder_dim": p2v.encoder_dim,
                "encoder_depth": p2v.encoder_depth,
                "encoder_heads": p2v.encoder_heads,
                "encoder_dropout": p2v.encoder_dropout,
                "encoder_attention_dropout": p2v.encoder_attention_dropout,
                "encoder_drop_path_rate": p2v.encoder_drop_path_rate,
                "encoder_add_pos_at_every_layer": p2v.encoder_add_pos_at_every_layer,
                "train_transformations": list(p2v.train_transformations),
                "val_transformations": list(p2v.val_transformations),
            }
        )
        return common
    raise ValueError(f"Unsupported RLCfg.actor_critic_class: {rl.actor_critic_class!r}")


def resolve_encoder_checkpoint(
    exp_cfg: ExpCfg,
    *,
    encoder_checkpoint_override: str | None = None,
) -> str | None:
    model = exp_cfg.model
    if exp_cfg.rl.actor_critic_class == "ActorCriticICP":
        return (
            encoder_checkpoint_override
            or exp_cfg.rl.encoder_checkpoint
            or model.icp.checkpoint_path
            or model.pretrained_encoder.checkpoint_path
        )
    return (
        encoder_checkpoint_override
        or model.pretrained_encoder.checkpoint_path
        or model.encoder.checkpoint_path
        or exp_cfg.rl.encoder_checkpoint
    )


def run_rl_training(
    exp_cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: str | Path,
    *,
    encoder_checkpoint_override: str | None = None,
    launch: bool = False,
) -> dict[str, Any]:
    """Write a config-derived RL runtime spec and optionally launch Isaac."""

    spec = build_rl_runtime_spec(
        exp_cfg,
        paths,
        artifact_dir,
        mode="train",
        encoder_checkpoint_override=encoder_checkpoint_override,
    )
    spec_path = Path(spec.runtime_spec_path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(spec)
    validate_runtime_spec(payload, spec_path)
    write_json(spec_path, payload)
    if launch:
        from utils.experiment.isaac_rl_launcher import launch_from_runtime_spec

        payload["launch_result"] = launch_from_runtime_spec(spec_path)
    return payload


def launch_isaac_training_from_spec(spec: RLRuntimeSpec) -> None:
    """Explicit heavy launcher for future runtime wiring."""

    from utils.experiment.isaac_rl_launcher import launch_from_runtime_spec

    launch_from_runtime_spec(spec.runtime_spec_path)


def main(argv: list[str] | None = None) -> int:
    raise SystemExit("Use run_experiment.py --config <experiment.py> [--mode run|plan].")


if __name__ == "__main__":
    raise SystemExit(main())
