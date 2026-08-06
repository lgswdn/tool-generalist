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
    rl_init_checkpoint: str | None
    rl_resume_checkpoint: str | None
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
    runtime_objects_manifest: str | Path | None = None,
    runtime_num_gpus: int | None = None,
    runtime_num_envs: int | None = None,
    runtime_rl_resume_checkpoint: str | Path | None = None,
    runtime_print_fine_grained_timing: bool = False,
) -> RLRuntimeSpec:
    rl = exp_cfg.rl
    num_gpus = int(runtime_num_gpus) if runtime_num_gpus is not None else exp_cfg.num_gpus
    num_envs = int(runtime_num_envs) if runtime_num_envs is not None else rl.env.num_envs
    env_params = asdict(rl.env)
    env_params["num_envs"] = num_envs
    launch_params = asdict(rl.launch)
    launch_params["print_fine_grained_timing"] = bool(
        runtime_print_fine_grained_timing
    )
    if runtime_num_gpus is not None:
        launch_params["distributed"] = num_gpus > 1
    checkpoint = resolve_encoder_checkpoint(
        exp_cfg,
        encoder_checkpoint_override=encoder_checkpoint_override,
    )
    policy_params = _build_policy_params(exp_cfg, checkpoint)
    runtime_paths_yaml = materialize_runtime_paths_yaml(
        exp_cfg,
        paths,
        Path(artifact_dir) / "paths.runtime.yaml",
        extra_overrides=(
            {"objects.candidates_json": Path(runtime_objects_manifest)}
            if runtime_objects_manifest is not None
            else None
        ),
    )
    return RLRuntimeSpec(
        mode=mode,
        seed=exp_cfg.general.seed,
        task_id=rl.resolved_task_id(),
        rsl_rl_cfg_entry_point=rl.rsl_rl_cfg_entry_point,
        train_entrypoint=rl.rsl_rl_train_entrypoint,
        actor_critic_class=rl.actor_critic_class,
        algorithm_class=rl.algorithm_class,
        num_gpus=num_gpus,
        num_envs=num_envs,
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
        launch_params=launch_params,
        table_params=asdict(rl.table),
        object_pose_sampling_params=asdict(rl.object_pose_sampling),
        curriculum_params=asdict(rl.curriculum),
        asset_assignment_params={
            "seed": exp_cfg.general.seed,
            "randomize_tool_assignment": exp_cfg.general.randomize_tool_assignment,
            "randomize_object_assignment": exp_cfg.general.randomize_object_assignment,
        },
        env_params=env_params,
        reward_params=asdict(rl.reward),
        domain_randomization_params=asdict(rl.domain_randomization),
        encoder_checkpoint=checkpoint,
        rl_init_checkpoint=(
            str(Path(rl.init_checkpoint))
            if rl.init_checkpoint is not None
            else None
        ),
        rl_resume_checkpoint=(
            str(Path(runtime_rl_resume_checkpoint))
            if runtime_rl_resume_checkpoint is not None
            else (
                str(Path(rl.resume_checkpoint))
                if rl.resume_checkpoint is not None
                else None
            )
        ),
        freeze_encoder=rl.freeze_encoder,
        artifact_dir=str(Path(artifact_dir)),
        paths_yaml=str(runtime_paths_yaml),
        runtime_spec_path=str(Path(artifact_dir) / RUNTIME_SPEC_FILENAME),
        runtime_spec_env_var=RUNTIME_SPEC_ENV_VAR,
    )


def _base_policy_params(
    exp_cfg: ExpCfg,
    checkpoint: str | None,
    *,
    class_name: str,
    num_points: int,
) -> dict[str, Any]:
    rl = exp_cfg.rl
    model = exp_cfg.model
    pretrained = model.pretrained_encoder
    return {
        "class_name": class_name,
        "num_points": num_points,
        "point_dim": rl.observation.point_dim,
        "encoder_weights_path": checkpoint,
        "encoder_checkpoint_name": pretrained.name,
        "encoder_checkpoint_schema": pretrained.schema,
        "encoder_checkpoint_adapter": pretrained.adapter,
    }


def _append_icp_policy_params(
    params: dict[str, Any],
    exp_cfg: ExpCfg,
    checkpoint: str | None,
) -> dict[str, Any]:
    rl = exp_cfg.rl
    model = exp_cfg.model
    params.update(
        {
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
            "cross_attn_token_order": model.policy_fusion.cross_attn_token_order,
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
    )
    return params


def _append_tg_shared_policy_params(params: dict[str, Any], exp_cfg: ExpCfg) -> dict[str, Any]:
    rl = exp_cfg.rl
    model = exp_cfg.model
    params.update(
        {
            "freeze_encoder": rl.freeze_encoder,
            "freeze_point2vec": rl.freeze_encoder,
            "vit_attention_mode": model.tce.vit_attention_mode,
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
            "task_embedding_dim": rl.observation.task_embedding_dim,
            "physics_dim": rl.effective_physics_dim,
            "model_input_centering": rl.observation.model_input_centering,
            "sm_num_layers": 2,
            "sm_num_modules": 4,
            "sm_module_hidden": 128,
            "sm_gating_hidden": 128,
            "sm_num_gating_layers": 1,
            "sm_cond_ob": True,
            "sm_add_bn": False,
            "hamnet_num_modules": rl.hamnet_num_modules,
            "hamnet_hidden_dims": list(rl.hamnet_hidden_dims),
            "hamnet_router_hidden_dims": list(rl.hamnet_router_hidden_dims),
            "activation": "elu",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
        }
    )
    return params


def _append_tce_policy_params(params: dict[str, Any], exp_cfg: ExpCfg) -> dict[str, Any]:
    _append_tg_shared_policy_params(params, exp_cfg)
    tce = exp_cfg.model.tce
    params.update(
        {
            "encoder_backend": "tce",
            "patch_size": tce.patch_size,
            "encoder_channel": tce.encoder_channel,
            "vit_depth": tce.vit_depth,
            "vit_heads": tce.vit_heads,
            "vit_attention_mode": tce.vit_attention_mode,
            "unicorn_token_source": tce.rl_token_source,
            "encoder_token_pca_rank": tce.encoder_token_pca_rank,
            "encoder_token_pca_path": tce.encoder_token_pca_path,
            "encoder_token_bottleneck_rank": tce.encoder_token_bottleneck_rank,
            "encoder_token_bottleneck_pca_path": tce.encoder_token_bottleneck_pca_path,
            "kinematic_conditioning": tce.kinematic_conditioning.enabled,
            "kinematic_attention_layers": (
                tce.kinematic_conditioning.attention_layers
            ),
        }
    )
    return params


def _append_tg_output_gate_policy_params(params: dict[str, Any], exp_cfg: ExpCfg) -> dict[str, Any]:
    _append_tce_policy_params(params, exp_cfg)
    rl = exp_cfg.rl
    params.update(
        {
            "expert_a_checkpoint": rl.output_gate_expert_a_checkpoint,
            "expert_b_checkpoint": rl.output_gate_expert_b_checkpoint,
            "output_gate_freeze_experts": rl.output_gate_freeze_experts,
            "output_gate_hidden_dims": list(rl.output_gate_hidden_dims),
            "output_gate_initial_expert_a_weight": rl.output_gate_initial_expert_a_weight,
            "output_gate_per_action": rl.output_gate_per_action,
        }
    )
    return params


def _append_unicorn_policy_params(params: dict[str, Any], exp_cfg: ExpCfg) -> dict[str, Any]:
    _append_tg_shared_policy_params(params, exp_cfg)
    unicorn = exp_cfg.model.unicorn
    params.update(
        {
            "encoder_backend": "unicorn",
            "num_patches": unicorn.num_patches,
            "patch_size": unicorn.patch_size,
            "encoder_channel": unicorn.encoder_channel,
            "vit_depth": unicorn.vit_depth,
            "vit_heads": unicorn.vit_heads,
            "unicorn_token_source": unicorn.rl_token_source,
        }
    )
    return params


def _append_patch_distance_pointnet_policy_params(
    params: dict[str, Any], exp_cfg: ExpCfg
) -> dict[str, Any]:
    _append_tg_shared_policy_params(params, exp_cfg)
    encoder = exp_cfg.model.patch_distance_pointnet
    params.update(
        {
            "encoder_backend": "patch_distance_pointnet",
            "num_patches": encoder.num_patches,
            "patch_size": encoder.patch_size,
            "encoder_channel": encoder.encoder_channel,
            # No transformer exists; these remain in the shared runtime schema.
            "vit_depth": 0,
            "vit_heads": 1,
            "patch_distance_point_scale_m": encoder.point_scale_m,
            "patch_distance_patch_center_scale_m": encoder.patch_center_scale_m,
        }
    )
    return params


def _append_oracle_patch_policy_params(params: dict[str, Any], exp_cfg: ExpCfg) -> dict[str, Any]:
    _append_tg_shared_policy_params(params, exp_cfg)
    oracle = exp_cfg.model.oracle_patch
    params.update(
        {
            "encoder_backend": "oracle_patch",
            "num_patches": oracle.num_patches,
            "patch_size": oracle.patch_size,
            "encoder_channel": oracle.encoder_channel,
            # Kept in the shared runtime schema; no ViT is constructed.
            "vit_depth": 0,
            "vit_heads": 1,
            "oracle_contact_eps": oracle.contact_eps,
            "oracle_center_scale_m": oracle.center_scale_m,
            "oracle_distance_scale_m": oracle.distance_scale_m,
            "oracle_patch_relative_scale_m": oracle.patch_relative_scale_m,
            "oracle_log_distance_resolution_m": oracle.log_distance_resolution_m,
            "oracle_log_distance_cap_m": oracle.log_distance_cap_m,
            "oracle_normalization_clip": oracle.normalization_clip,
        }
    )
    return params


def _append_oracle_pointmesh_pointnet_policy_params(
    params: dict[str, Any], exp_cfg: ExpCfg
) -> dict[str, Any]:
    _append_tg_shared_policy_params(params, exp_cfg)
    pointmesh = exp_cfg.model.oracle_pointmesh_pointnet
    params.update(
        {
            "encoder_backend": "oracle_pointmesh_pointnet",
            "num_patches": pointmesh.num_patches,
            "patch_size": pointmesh.patch_size,
            "encoder_channel": pointmesh.encoder_channel,
            "vit_depth": 0,
            "vit_heads": 1,
            "oracle_pointmesh_coordinate_scale_m": pointmesh.coordinate_scale_m,
            "oracle_pointmesh_distance_scale_m": pointmesh.distance_scale_m,
            "oracle_pointmesh_normalization_clip": pointmesh.normalization_clip,
        }
    )
    return params


def _append_oracle_pointcloud_pointnet_policy_params(
    params: dict[str, Any], exp_cfg: ExpCfg
) -> dict[str, Any]:
    _append_tg_shared_policy_params(params, exp_cfg)
    pointcloud = exp_cfg.model.oracle_pointcloud_pointnet
    params.update(
        {
            "encoder_backend": "oracle_pointcloud_pointnet",
            "num_patches": pointcloud.num_patches,
            "patch_size": pointcloud.patch_size,
            "encoder_channel": pointcloud.encoder_channel,
            "vit_depth": 0,
            "vit_heads": 1,
            "oracle_pointcloud_nearest_frame_batch_size": (
                pointcloud.nearest_frame_batch_size
            ),
            "oracle_pointcloud_feature_mode": pointcloud.feature_mode,
            "oracle_pointcloud_load_fitted_weights": (
                pointcloud.load_fitted_weights
            ),
            "oracle_pointcloud_use_rank10_bottleneck": (
                pointcloud.use_rank10_bottleneck
            ),
            "oracle_pointcloud_token_mode": pointcloud.token_mode,
            "oracle_pointcloud_input_normalization": (
                pointcloud.input_normalization
            ),
            "oracle_pointcloud_checkpoint_adapter": (
                exp_cfg.model.pretrained_encoder.adapter
            ),
        }
    )
    return params


def _append_oracle_pointcloud_patch_oracle_policy_params(
    params: dict[str, Any], exp_cfg: ExpCfg
) -> dict[str, Any]:
    _append_tg_shared_policy_params(params, exp_cfg)
    oracle = exp_cfg.model.oracle_pointcloud_patch_oracle
    params.update(
        {
            "encoder_backend": "oracle_pointcloud_patch_oracle",
            "num_patches": oracle.num_patches,
            "patch_size": oracle.patch_size,
            "encoder_channel": oracle.encoder_channel,
            "vit_depth": 0,
            "vit_heads": 1,
            "oracle_pointcloud_nearest_frame_batch_size": (
                oracle.nearest_frame_batch_size
            ),
        }
    )
    return params


def _append_point2vec_policy_params(
    params: dict[str, Any],
    exp_cfg: ExpCfg,
    checkpoint: str | None,
) -> dict[str, Any]:
    _append_tg_shared_policy_params(params, exp_cfg)
    p2v = exp_cfg.model.p2v
    params.update(
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
    return params


def _build_policy_params(exp_cfg: ExpCfg, checkpoint: str | None) -> dict[str, Any]:
    rl = exp_cfg.rl
    if rl.actor_critic_class == "ActorCriticICP":
        return _append_icp_policy_params(
            _base_policy_params(
                exp_cfg,
                checkpoint,
                class_name="ActorCriticICP",
                num_points=rl.observation.num_points,
            ),
            exp_cfg,
            checkpoint,
        )

    encoder = exp_cfg.model.encoder
    common = _base_policy_params(
        exp_cfg,
        checkpoint,
        class_name=rl.actor_critic_class,
        num_points=encoder.num_points,
    )
    if rl.actor_critic_class == "ActorCriticTGOutputGate":
        return _append_tg_output_gate_policy_params(common, exp_cfg)
    if rl.actor_critic_class in {
        "ActorCriticTG",
        "ActorCriticTGSM",
        "ActorCriticTGHAMNet",
        "ActorCriticTGBimanual",
    }:
        if rl.actor_critic_class == "ActorCriticTG" and exp_cfg.model.encoder_backend == "oracle_patch":
            return _append_oracle_patch_policy_params(common, exp_cfg)
        if (
            rl.actor_critic_class == "ActorCriticTG"
            and exp_cfg.model.encoder_backend == "oracle_pointmesh_pointnet"
        ):
            return _append_oracle_pointmesh_pointnet_policy_params(common, exp_cfg)
        if (
            rl.actor_critic_class == "ActorCriticTG"
            and exp_cfg.model.encoder_backend == "oracle_pointcloud_pointnet"
        ):
            return _append_oracle_pointcloud_pointnet_policy_params(common, exp_cfg)
        if (
            rl.actor_critic_class == "ActorCriticTG"
            and exp_cfg.model.encoder_backend == "oracle_pointcloud_patch_oracle"
        ):
            return _append_oracle_pointcloud_patch_oracle_policy_params(common, exp_cfg)
        if rl.actor_critic_class == "ActorCriticTG" and exp_cfg.model.encoder_backend == "unicorn":
            return _append_unicorn_policy_params(common, exp_cfg)
        if (
            rl.actor_critic_class == "ActorCriticTG"
            and exp_cfg.model.encoder_backend == "patch_distance_pointnet"
        ):
            return _append_patch_distance_pointnet_policy_params(common, exp_cfg)
        return _append_tce_policy_params(common, exp_cfg)
    if rl.actor_critic_class == "ActorCriticTGUnicorn":
        return _append_unicorn_policy_params(common, exp_cfg)
    if rl.actor_critic_class == "ActorCriticPoint2Vec":
        return _append_point2vec_policy_params(common, exp_cfg, checkpoint)
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
    runtime_objects_manifest: str | Path | None = None,
    runtime_num_gpus: int | None = None,
    runtime_num_envs: int | None = None,
    runtime_rl_resume_checkpoint: str | Path | None = None,
    runtime_print_fine_grained_timing: bool = False,
    launch: bool = False,
) -> dict[str, Any]:
    """Write a config-derived RL runtime spec and optionally launch Isaac."""

    spec = build_rl_runtime_spec(
        exp_cfg,
        paths,
        artifact_dir,
        mode="train",
        encoder_checkpoint_override=encoder_checkpoint_override,
        runtime_objects_manifest=runtime_objects_manifest,
        runtime_num_gpus=runtime_num_gpus,
        runtime_num_envs=runtime_num_envs,
        runtime_rl_resume_checkpoint=runtime_rl_resume_checkpoint,
        runtime_print_fine_grained_timing=runtime_print_fine_grained_timing,
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
