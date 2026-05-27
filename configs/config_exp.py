"""Top-level experiment config.

``ExpCfg`` is the only semantic config object accepted by the new automation
entrypoint.  Runtime-only flags may be provided to the CLI, but experiment
parameters should live in this dataclass tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .config_contact_gen import (
    ROTATION_SELECTION_MOST_DOWNWARD,
    ROTATION_SELECTION_RANDOM_LEGAL,
    TOOL_SOURCE_OBJECTS,
    TOOL_SOURCE_SELECTED_TOOLS,
    ContactGenCfg,
)
from .config_general import GeneralCfg
from .config_model import ModelCfg
from .config_pretrain import PretrainCfg
from .config_rl import RLCfg


ArtifactPolicy = Literal["reuse", "overwrite", "fail-if-exists"]


class ConfigValidationError(ValueError):
    """Raised when an experiment config is internally inconsistent."""


@dataclass
class ExpCfg:
    name: str = "default_exp"
    paths_yaml: str = "paths.yaml"
    num_gpus: int = 0
    artifact_policy: ArtifactPolicy = "reuse"
    pretrain_reuse: str | None = None
    general: GeneralCfg = field(default_factory=GeneralCfg)
    contact_gen: ContactGenCfg = field(default_factory=ContactGenCfg)
    pretrain: PretrainCfg = field(default_factory=PretrainCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    rl: RLCfg = field(default_factory=RLCfg)
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        errors: list[str] = []

        if not self.name:
            errors.append("ExpCfg.name must be non-empty")
        _require_name(errors, "ExpCfg.paths_yaml", self.paths_yaml)
        if self.num_gpus < 0:
            errors.append("ExpCfg.num_gpus must be >= 0")
        if self.artifact_policy not in {"reuse", "overwrite", "fail-if-exists"}:
            errors.append("ExpCfg.artifact_policy must be reuse, overwrite, or fail-if-exists")
        if self.pretrain_reuse is not None and (
            not isinstance(self.pretrain_reuse, str) or not self.pretrain_reuse.strip()
        ):
            errors.append("ExpCfg.pretrain_reuse must be a non-empty string or None")

        _require_name(errors, "GeneralCfg.name", self.general.name)
        _require_name(errors, "ContactGenCfg.name", self.contact_gen.name)
        _require_name(errors, "PretrainCfg.name", self.pretrain.name)
        _require_name(errors, "ModelCfg.name", self.model.name)
        _require_name(errors, "RLCfg.name", self.rl.name)

        _require_positive_int(errors, "GeneralCfg.seed", self.general.seed, allow_zero=True)
        _require_positive_int(errors, "GeneralCfg.num_points", self.general.num_points)
        for field_name in (
            "randomize_tool_assignment",
            "randomize_object_assignment",
        ):
            if not isinstance(getattr(self.general, field_name), bool):
                errors.append(f"GeneralCfg.{field_name} must be a bool")
        if self.general.dtype not in {"float16", "float32", "float64", "bfloat16"}:
            errors.append("GeneralCfg.dtype must be float16, float32, float64, or bfloat16")
        _require_name(errors, "ContactGenCfg.schema_version", self.contact_gen.schema_version)
        _require_positive_int(errors, "ContactGenCfg.num_pairs", self.contact_gen.num_pairs)
        _require_positive_int(
            errors, "ContactGenCfg.num_object_poses", self.contact_gen.num_object_poses
        )
        _require_positive_int(errors, "ContactGenCfg.B", self.contact_gen.B)
        _require_positive_int(errors, "ContactGenCfg.M", self.contact_gen.M)
        _require_positive_int(errors, "ContactGenCfg.chunk_B", self.contact_gen.chunk_B)
        _require_positive_int(
            errors, "ContactPhysicsCfg.num_workers", self.contact_gen.physics.num_workers
        )
        _require_positive_int(
            errors, "ContactPhysicsCfg.t_stabilize", self.contact_gen.physics.t_stabilize
        )
        _require_positive_int(
            errors, "ContactPhysicsCfg.t_postcontact", self.contact_gen.physics.t_postcontact
        )
        _require_positive_int(
            errors,
            "ContactGenCfg.max_contacts_per_pair",
            self.contact_gen.max_contacts_per_pair,
        )
        if self.contact_gen.rotation_selection not in {
            ROTATION_SELECTION_MOST_DOWNWARD,
            ROTATION_SELECTION_RANDOM_LEGAL,
        }:
            errors.append(
                "ContactGenCfg.rotation_selection must be "
                f"{ROTATION_SELECTION_MOST_DOWNWARD} or {ROTATION_SELECTION_RANDOM_LEGAL}"
            )
        if self.contact_gen.tool_source not in {TOOL_SOURCE_SELECTED_TOOLS, TOOL_SOURCE_OBJECTS}:
            errors.append(
                "ContactGenCfg.tool_source must be "
                f"{TOOL_SOURCE_SELECTED_TOOLS} or {TOOL_SOURCE_OBJECTS}"
            )
        if self.contact_gen.object_tool_manifest is not None and (
            not isinstance(self.contact_gen.object_tool_manifest, str)
            or not self.contact_gen.object_tool_manifest.strip()
        ):
            errors.append("ContactGenCfg.object_tool_manifest must be a non-empty string or None")
        if not isinstance(self.contact_gen.allow_self_object_tool_pairs, bool):
            errors.append("ContactGenCfg.allow_self_object_tool_pairs must be a bool")
        _require_positive_int(errors, "ContactGenCfg.shard_count", self.contact_gen.shard_count)
        _require_positive_int(
            errors,
            "ContactGenCfg.shard_index",
            self.contact_gen.shard_index,
            allow_zero=True,
        )
        if self.contact_gen.shard_index >= self.contact_gen.shard_count:
            errors.append("ContactGenCfg.shard_index must be < ContactGenCfg.shard_count")
        _require_positive_int(errors, "PretrainBatchCfg.batch_size", self.pretrain.batch.batch_size)
        _require_positive_int(
            errors,
            "PretrainCfg.num_precontact_steps",
            self.pretrain.num_precontact_steps,
            allow_zero=True,
        )
        _require_positive_int(
            errors,
            "PretrainCfg.legal_pose_max_tries",
            self.pretrain.legal_pose_max_tries,
        )
        for field_name in ("num_query_A", "num_query_B", "num_query_C", "num_query_D"):
            _require_positive_int(errors, f"PretrainCfg.{field_name}", getattr(self.pretrain, field_name))
        _require_positive_int(errors, "PretrainCfg.cross_attn_layers", self.pretrain.cross_attn_layers)
        _require_positive_int(errors, "PretrainCfg.cross_attn_heads", self.pretrain.cross_attn_heads)
        _require_positive_int(
            errors,
            "PretrainCfg.validation_noising_seed",
            self.pretrain.validation_noising_seed,
            allow_zero=True,
        )
        _require_positive_int(errors, "PretrainCfg.epochs", self.pretrain.epochs)
        _require_positive_int(errors, "RLEnvCfg.num_envs", self.rl.env.num_envs)
        _require_positive_int(errors, "RLEnvCfg.decimation", self.rl.env.decimation)
        _require_positive_int(
            errors,
            "RLEnvCfg.solver_position_iteration_count",
            self.rl.env.solver_position_iteration_count,
        )
        _require_positive_int(
            errors,
            "RLEnvCfg.solver_velocity_iteration_count",
            self.rl.env.solver_velocity_iteration_count,
        )
        _require_positive_int(errors, "PPOCfg.num_steps_per_env", self.rl.ppo.num_steps_per_env)
        _require_positive_int(errors, "PPOCfg.max_iterations", self.rl.ppo.max_iterations)
        _require_positive_int(errors, "PPOCfg.save_interval", self.rl.ppo.save_interval)
        _require_positive_int(errors, "PPOCfg.num_learning_epochs", self.rl.ppo.num_learning_epochs)
        _require_positive_int(errors, "PPOCfg.num_mini_batches", self.rl.ppo.num_mini_batches)
        _require_positive_int(errors, "ActionCfg.action_dim", self.rl.action.action_dim)
        _require_positive_int(
            errors,
            "RLCfg.effective_action_dim",
            self.rl.effective_action_dim,
        )
        _require_positive_int(errors, "ObservationCfg.num_points", self.rl.observation.num_points)
        _require_positive_int(errors, "ObservationCfg.point_dim", self.rl.observation.point_dim)
        _require_positive_int(
            errors,
            "RLCfg.effective_physics_dim",
            self.rl.effective_physics_dim,
            allow_zero=True,
        )

        if self.rl.env.sim_dt <= 0:
            errors.append("RLEnvCfg.sim_dt must be > 0")
        if self.rl.env.episode_length_s <= 0:
            errors.append("RLEnvCfg.episode_length_s must be > 0")
        if self.rl.env.robot_mode not in {"tool", "bare_franka"}:
            errors.append("RLEnvCfg.robot_mode must be tool or bare_franka")
        if not isinstance(self.rl.separate_actor_critic_fusion, bool):
            errors.append("RLCfg.separate_actor_critic_fusion must be a bool")
        if self.rl.reward.rotation_distance_divisor <= 0:
            errors.append("RewardCfg.rotation_distance_divisor must be > 0")
        if self.rl.reward.bimanual_arm_proximity_warning_distance <= 0:
            errors.append("RewardCfg.bimanual_arm_proximity_warning_distance must be > 0")
        if self.rl.reward.bimanual_arm_proximity_failure_distance <= 0:
            errors.append("RewardCfg.bimanual_arm_proximity_failure_distance must be > 0")
        if (
            self.rl.reward.bimanual_arm_proximity_warning_distance
            <= self.rl.reward.bimanual_arm_proximity_failure_distance
        ):
            errors.append(
                "RewardCfg.bimanual_arm_proximity_warning_distance must be greater than "
                "bimanual_arm_proximity_failure_distance"
            )
        if self.pretrain.optimizer.learning_rate <= 0:
            errors.append("PretrainOptimizerCfg.learning_rate must be > 0")
        if self.pretrain.loss.sdf_relative_eps <= 0:
            errors.append("PretrainLossCfg.sdf_relative_eps must be > 0")
        if self.pretrain.condition_normalization not in {None, True, False}:
            errors.append("PretrainCfg.condition_normalization must be None, True, or False")
        _require_positive_int(
            errors,
            "PretrainCfg.condition_norm_sample_files",
            self.pretrain.condition_norm_sample_files,
        )
        if self.pretrain.condition_norm_eps <= 0:
            errors.append("PretrainCfg.condition_norm_eps must be > 0")
        if self.pretrain.logger not in {"none", "wandb"}:
            errors.append("PretrainCfg.logger must be none or wandb")
        if self.pretrain.wandb_mode not in {"online", "offline", "disabled"}:
            errors.append("PretrainCfg.wandb_mode must be online, offline, or disabled")
        if not self.pretrain.enabled_heads:
            errors.append("PretrainCfg.enabled_heads must enable at least one head")
        else:
            allowed_heads = {"sdf", "diff", "postcontact", "contact"}
            unknown = sorted(set(self.pretrain.enabled_heads).difference(allowed_heads))
            if unknown:
                errors.append(f"PretrainCfg.enabled_heads contains unknown heads: {unknown}")
        if self.pretrain.mode not in {"tce_multitask", "unicorn_contact"}:
            errors.append("PretrainCfg.mode must be tce_multitask or unicorn_contact")
        if self.pretrain.mode == "unicorn_contact" and self.pretrain.enabled_heads != ["contact"]:
            errors.append("PretrainCfg.mode=unicorn_contact requires enabled_heads=['contact']")
        if self.pretrain.optimizer.name not in {"adamw", "sam"}:
            errors.append("PretrainOptimizerCfg.name must be adamw or sam")
        if self.pretrain.optimizer.sam_rho <= 0:
            errors.append("PretrainOptimizerCfg.sam_rho must be > 0")
        if (
            self.pretrain.unicorn.positive_patch_fraction <= 0
            or self.pretrain.unicorn.positive_patch_fraction >= 1
        ):
            errors.append("UnicornPretrainCfg.positive_patch_fraction must be in (0, 1)")
        if self.pretrain.unicorn.num_patches <= 0:
            errors.append("UnicornPretrainCfg.num_patches must be > 0")
        if self.pretrain.unicorn.label.contact_eps < 0:
            errors.append("UnicornLabelCfg.contact_eps must be >= 0")
        if self.pretrain.unicorn.label.patch_positive_rule not in {"any", "count"}:
            errors.append("UnicornLabelCfg.patch_positive_rule must be any or count")
        if self.pretrain.noise_schedule_mode not in {"interpolation", "slerp", "random_walk"}:
            errors.append(
                "PretrainCfg.noise_schedule_mode must be interpolation, slerp, or random_walk"
            )
        if self.pretrain.decoder_pooling not in {"mean", "min", "max"}:
            errors.append("PretrainCfg.decoder_pooling must be mean, min, or max")
        _require_range(
            errors,
            "PretrainCfg.translation_noise_range",
            self.pretrain.translation_noise_range,
        )
        _require_range(
            errors,
            "PretrainCfg.rotation_noise_range_deg",
            self.pretrain.rotation_noise_range_deg,
        )
        for field_name in (
            "translation_noise_range",
            "rotation_noise_range_deg",
        ):
            if any(item < 0 for item in getattr(self.pretrain, field_name)):
                errors.append(f"PretrainCfg.{field_name} values must be >= 0")
        for field_name in (
            "condition_mlp_hidden_dims",
            "sdf_head_hidden_dims",
            "denoise_head_hidden_dims",
            "postcontact_head_hidden_dims",
        ):
            dims = getattr(self.pretrain, field_name)
            if not dims:
                errors.append(f"PretrainCfg.{field_name} must be non-empty")
            for index, hidden_dim in enumerate(dims):
                _require_positive_int(
                    errors,
                    f"PretrainCfg.{field_name}[{index}]",
                    hidden_dim,
                )

        _require_len(errors, "GeneralCfg.tool_mount.scale_xyz", self.general.tool_mount.scale_xyz, 3)
        _require_all_positive(errors, "GeneralCfg.tool_mount.scale_xyz", self.general.tool_mount.scale_xyz)
        _require_len(errors, "GeneralCfg.tool_mount.translate", self.general.tool_mount.translate, 3)
        _require_len(errors, "GeneralCfg.tool_mount.rot_wxyz", self.general.tool_mount.rot_wxyz, 4)
        _require_len(errors, "GeneralCfg.tool_mount.pose_xyz", self.general.tool_mount.pose_xyz, 3)
        _require_len(
            errors,
            "GeneralCfg.tool_mount.pose_quat_wxyz",
            self.general.tool_mount.pose_quat_wxyz,
            4,
        )
        _require_len(errors, "TableCfg.size_xyz", self.rl.table.size_xyz, 3)
        _require_all_positive(errors, "TableCfg.size_xyz", self.rl.table.size_xyz)
        _require_len(errors, "TableCfg.pose_xyz", self.rl.table.pose_xyz, 3)
        _require_len(errors, "TableCfg.color_rgba", self.rl.table.color_rgba, 4)
        if self.rl.table.placement_margin_xy < 0:
            errors.append("TableCfg.placement_margin_xy must be >= 0")
        _require_positive_int(
            errors,
            "TableCfg.placement_max_attempts",
            self.rl.table.placement_max_attempts,
        )
        if self.rl.object_pose_sampling.initial_position_range < 0:
            errors.append("ObjectPoseSamplingCfg.initial_position_range must be >= 0")
        if self.rl.object_pose_sampling.xy_offset_range < 0:
            errors.append("ObjectPoseSamplingCfg.xy_offset_range must be >= 0")
        if not isinstance(self.rl.curriculum.enabled, bool):
            errors.append("RLCurriculumCfg.enabled must be a bool")
        _require_positive_int(
            errors,
            "RLCurriculumCfg.start_step",
            self.rl.curriculum.start_step,
            allow_zero=True,
        )
        _require_positive_int(
            errors,
            "RLCurriculumCfg.end_step",
            self.rl.curriculum.end_step,
            allow_zero=True,
        )
        if self.rl.curriculum.end_step < self.rl.curriculum.start_step:
            errors.append("RLCurriculumCfg.end_step must be >= start_step")
        for field_name in (
            "start_stable_pose_probability",
            "end_stable_pose_probability",
        ):
            value = getattr(self.rl.curriculum, field_name)
            if value < 0.0 or value > 1.0:
                errors.append(f"RLCurriculumCfg.{field_name} must be in [0, 1]")
        if self.rl.observation.object_velocity_dim < 0:
            errors.append("ObservationCfg.object_velocity_dim must be >= 0")
        if self.rl.reward.stable_success_dwell_steps < 1:
            errors.append("RewardCfg.stable_success_dwell_steps must be >= 1")
        _require_range(errors, "ActionCfg.clip", self.rl.action.clip)
        if self.rl.action.clip[0] < -1.0 or self.rl.action.clip[1] > 1.0:
            errors.append("ActionCfg.clip must stay within [-1, 1]")
        for field_name in (
            "object_cloud_centering",
            "tool_cloud_centering",
            "mesh_centering",
        ):
            if getattr(self.rl.observation, field_name) not in {"aabb_center", "centroid_mean", "none"}:
                errors.append(
                    f"ObservationCfg.{field_name} must be aabb_center, centroid_mean, or none"
                )
        if self.rl.observation.tool_cloud_source != "adjusted_decomposed_mesh":
            errors.append(
                "ObservationCfg.tool_cloud_source must be adjusted_decomposed_mesh"
            )
        if self.rl.observation.model_input_centering not in {"bbox_center", "object_center"}:
            errors.append("ObservationCfg.model_input_centering must be bbox_center or object_center")
        encoder_backend = self.model.encoder_backend.strip().lower()
        if encoder_backend in {"tg"}:
            encoder_backend = "tce"
        if encoder_backend in {"p2v"}:
            encoder_backend = "point2vec"
        if encoder_backend in {"corn"}:
            encoder_backend = "icp"
        if encoder_backend not in {"tce", "point2vec", "icp", "unicorn"}:
            errors.append("ModelCfg.encoder_backend must be tce, point2vec, icp, or unicorn")
        allowed_adapters = {"tce_strict", "point2vec_native", "icp_legacy", "unicorn_strict"}
        if self.model.pretrained_encoder.adapter not in allowed_adapters:
            errors.append(
                "ModelCfg.pretrained_encoder.adapter must be one of "
                f"{sorted(allowed_adapters)}"
            )
        allowed_actors = {"ActorCriticTG", "ActorCriticTGBimanual", "ActorCriticPoint2Vec", "ActorCriticICP"}
        if self.rl.actor_critic_class not in allowed_actors:
            errors.append(
                "RLCfg.actor_critic_class must be one of "
                f"{sorted(allowed_actors)}"
            )
        if encoder_backend == "tce":
            if self.rl.actor_critic_class not in {"ActorCriticTG", "ActorCriticTGBimanual"}:
                errors.append(
                    "encoder_backend=tce requires RLCfg.actor_critic_class=ActorCriticTG "
                    "or ActorCriticTGBimanual"
                )
            if self.model.pretrained_encoder.adapter != "tce_strict":
                errors.append("encoder_backend=tce requires pretrained_encoder.adapter=tce_strict")
        if encoder_backend == "point2vec":
            if self.rl.actor_critic_class != "ActorCriticPoint2Vec":
                errors.append(
                    "encoder_backend=point2vec requires "
                    "RLCfg.actor_critic_class=ActorCriticPoint2Vec"
                )
            if self.model.pretrained_encoder.adapter != "point2vec_native":
                errors.append(
                    "encoder_backend=point2vec requires "
                    "pretrained_encoder.adapter=point2vec_native"
                )
        if encoder_backend == "icp":
            if self.rl.actor_critic_class != "ActorCriticICP":
                errors.append("encoder_backend=icp requires RLCfg.actor_critic_class=ActorCriticICP")
            if self.model.pretrained_encoder.adapter != "icp_legacy":
                errors.append("encoder_backend=icp requires pretrained_encoder.adapter=icp_legacy")
            if self.rl.env.robot_mode != "bare_franka":
                errors.append("ActorCriticICP requires RLEnvCfg.robot_mode=bare_franka")
            expected_layout = [
                "object_cloud_flat",
                "hand_state",
                "robot_state",
                "previous_action",
                "relative_goal_pose",
                "physics",
            ]
            if self.rl.observation.layout != expected_layout:
                errors.append(f"ActorCriticICP requires ObservationCfg.layout={expected_layout!r}")
            if not self.rl.observation.include_object_cloud:
                errors.append("ActorCriticICP requires ObservationCfg.include_object_cloud=True")
            if self.rl.observation.include_tool_cloud:
                errors.append("ActorCriticICP requires ObservationCfg.include_tool_cloud=False")
            if self.rl.observation.include_bbox_centers:
                errors.append("ActorCriticICP requires ObservationCfg.include_bbox_centers=False")
        if encoder_backend == "unicorn" and self.model.pretrained_encoder.adapter != "unicorn_strict":
            errors.append("encoder_backend=unicorn requires pretrained_encoder.adapter=unicorn_strict")
        if self.pretrain.enabled and self.pretrain.mode == "unicorn_contact" and encoder_backend != "unicorn":
            errors.append("PretrainCfg.mode=unicorn_contact requires ModelCfg.encoder_backend=unicorn")
        if self.pretrain.enabled and (
            self.pretrain.mode != "unicorn_contact"
            and (
                encoder_backend != "tce"
                or self.rl.actor_critic_class not in {"ActorCriticTG", "ActorCriticTGBimanual"}
            )
        ):
            errors.append("PretrainCfg.enabled currently supports only TCE with ActorCriticTG/ActorCriticTGBimanual")
        if self.rl.algorithm_class != "PPO":
            errors.append("RLCfg.algorithm_class must be PPO")
        if self.rl.ppo.class_name != "PPO":
            errors.append("PPOCfg.class_name must be PPO")
        if self.rl.launch.logger not in {"tensorboard", "wandb", "neptune", "none"}:
            errors.append("RLLaunchCfg.logger must be tensorboard, wandb, neptune, or none")
        if self.rl.launch.device is not None and not str(self.rl.launch.device).strip():
            errors.append("RLLaunchCfg.device must be non-empty when set")
        if self.rl.launch.distributed and self.num_gpus <= 0:
            errors.append("RLLaunchCfg.distributed requires ExpCfg.num_gpus > 0")
        if (
            self.rl.table.enabled
            and self.rl.domain_randomization.enabled
            and self.rl.domain_randomization.ground.material.enabled
        ):
            errors.append(
                "TableCfg.enabled removes the ground plane; use "
                "DomainRandomizationCfg.table_surface.material instead of "
                "ground.material."
            )

        for field_name in (
            "object_mass_range",
            "tool_mass_range",
            "object_friction_range",
            "tool_friction_range",
            "ground_friction_range",
        ):
            _require_range(errors, f"ContactPhysicsCfg.{field_name}", getattr(self.contact_gen.physics, field_name))

        for field_name in (
            "epsilon",
            "floor_eps",
            "penetration_eps",
            "rotation_orth_eps",
        ):
            if getattr(self.contact_gen, field_name) < 0:
                errors.append(f"ContactGenCfg.{field_name} must be >= 0")
        if not isinstance(self.contact_gen.contact_mode_prob, dict):
            errors.append("ContactGenCfg.contact_mode_prob must be a mapping")
        else:
            if any(prob < 0 for prob in self.contact_gen.contact_mode_prob.values()):
                errors.append("ContactGenCfg.contact_mode_prob values must be >= 0")
            if sum(self.contact_gen.contact_mode_prob.values()) <= 0:
                errors.append("ContactGenCfg.contact_mode_prob must have positive total mass")

        dr = self.rl.domain_randomization
        for field_name, value in (
            ("object.mass.range", dr.object.mass.range),
            ("tool.mass.range", dr.tool.mass.range),
            ("object.material.static_friction_range", dr.object.material.static_friction_range),
            ("object.material.dynamic_friction_range", dr.object.material.dynamic_friction_range),
            ("tool.material.static_friction_range", dr.tool.material.static_friction_range),
            ("tool.material.dynamic_friction_range", dr.tool.material.dynamic_friction_range),
            ("ground.material.static_friction_range", dr.ground.material.static_friction_range),
            ("ground.material.dynamic_friction_range", dr.ground.material.dynamic_friction_range),
        ):
            _require_range(errors, f"DomainRandomizationCfg.{field_name}", value)

        if self.model.tce.output_dim < 0:
            errors.append("TCECfg.output_dim must be >= 0")
        for field_name in ("num_points", "patch_size", "encoder_channel", "vit_depth", "vit_heads"):
            _require_positive_int(errors, f"TCECfg.{field_name}", getattr(self.model.tce, field_name))
        if self.model.p2v.output_dim < 0:
            errors.append("P2VCfg.output_dim must be >= 0")
        for field_name in (
            "num_points",
            "token_dim",
            "tokenizer_num_groups",
            "tokenizer_group_size",
            "encoder_dim",
            "encoder_depth",
            "encoder_heads",
        ):
            _require_positive_int(errors, f"P2VCfg.{field_name}", getattr(self.model.p2v, field_name))
        _require_positive_int(
            errors,
            "PolicyFusionCfg.sd_num_query",
            self.model.policy_fusion.sd_num_query,
        )
        if self.model.policy_fusion.relative_translation_query_tokens < 0:
            errors.append("PolicyFusionCfg.relative_translation_query_tokens must be >= 0")
        if self.model.policy_fusion.relative_translation_query_tokens > self.model.policy_fusion.sd_num_query:
            errors.append(
                "PolicyFusionCfg.relative_translation_query_tokens must be <= "
                "PolicyFusionCfg.sd_num_query"
            )
        _require_positive_int(errors, "PolicyFusionCfg.query_dim", self.model.policy_fusion.query_dim)
        _require_positive_int(
            errors,
            "PolicyFusionCfg.cross_attn_heads",
            self.model.policy_fusion.cross_attn_heads,
        )
        _require_positive_int(
            errors,
            "PolicyFusionCfg.cross_attn_layers",
            self.model.policy_fusion.cross_attn_layers,
        )
        for index, hidden_dim in enumerate(self.model.policy_fusion.fusion_hidden_dims):
            _require_positive_int(errors, f"PolicyFusionCfg.fusion_hidden_dims[{index}]", hidden_dim)

        try:
            active_encoder_checkpoint = self.model.encoder.checkpoint_path
        except ValueError:
            active_encoder_checkpoint = None
        if self.rl.enabled and not (
            self.rl.encoder_checkpoint
            or self.model.pretrained_encoder.checkpoint_path
            or active_encoder_checkpoint
            or self.pretrain.enabled
            or self.pretrain_reuse
        ):
            errors.append(
                "RLCfg.enabled requires an encoder checkpoint or PretrainCfg.enabled"
            )

        if errors:
            raise ConfigValidationError("; ".join(errors))


def _require_name(errors: list[str], field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_name} must be a non-empty string")


def _require_positive_int(
    errors: list[str], field_name: str, value: int, *, allow_zero: bool = False
) -> None:
    if not isinstance(value, int):
        errors.append(f"{field_name} must be an int")
        return
    if allow_zero:
        if value < 0:
            errors.append(f"{field_name} must be >= 0")
    elif value <= 0:
        errors.append(f"{field_name} must be > 0")


def _require_len(errors: list[str], field_name: str, value: list[float], expected: int) -> None:
    if len(value) != expected:
        errors.append(f"{field_name} must have length {expected}")


def _require_all_positive(errors: list[str], field_name: str, value: list[float]) -> None:
    if any(item <= 0 for item in value):
        errors.append(f"{field_name} values must be > 0")


def _require_range(errors: list[str], field_name: str, value: tuple[float, float]) -> None:
    if len(value) != 2:
        errors.append(f"{field_name} must have two values")
        return
    low, high = value
    if low > high:
        errors.append(f"{field_name} low must be <= high")
