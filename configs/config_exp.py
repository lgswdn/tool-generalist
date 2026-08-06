"""Top-level experiment config.

``ExpCfg`` is the only semantic config object accepted by the new automation
entrypoint.  Runtime-only flags may be provided to the CLI, but experiment
parameters should live in this dataclass tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .config_contact_gen import (
    CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION,
    CONTACT_GEOMETRY_BBOX_TRANSLATION_NEAREST,
    CONTACT_GEOMETRY_INTERSECTING_ANCHORS,
    CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
    PENETRATION_CHECK_BIDIRECTIONAL,
    PENETRATION_CHECK_TOOL_INTO_OBJECT,
    ROTATION_SELECTION_MOST_CAVITY_CENTERED,
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
    paths_yaml: str = "configs/paths/default.yaml"
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
            ROTATION_SELECTION_MOST_CAVITY_CENTERED,
        }:
            errors.append(
                "ContactGenCfg.rotation_selection must be "
                f"{ROTATION_SELECTION_MOST_DOWNWARD}, "
                f"{ROTATION_SELECTION_RANDOM_LEGAL}, or "
                f"{ROTATION_SELECTION_MOST_CAVITY_CENTERED}"
            )
        if self.contact_gen.contact_geometry_mode not in {
            CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION,
            CONTACT_GEOMETRY_BBOX_TRANSLATION_NEAREST,
            CONTACT_GEOMETRY_INTERSECTING_ANCHORS,
            CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
        }:
            errors.append(
                "ContactGenCfg.contact_geometry_mode must be "
                f"{CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION} or "
                f"{CONTACT_GEOMETRY_BBOX_TRANSLATION_NEAREST} or "
                f"{CONTACT_GEOMETRY_INTERSECTING_ANCHORS} or "
                f"{CONTACT_GEOMETRY_TANGENT_GAUSSIAN}"
            )
        if self.contact_gen.penetration_check_mode not in {
            PENETRATION_CHECK_TOOL_INTO_OBJECT,
            PENETRATION_CHECK_BIDIRECTIONAL,
        }:
            errors.append(
                "ContactGenCfg.penetration_check_mode must be "
                f"{PENETRATION_CHECK_TOOL_INTO_OBJECT} or "
                f"{PENETRATION_CHECK_BIDIRECTIONAL}"
            )
        if self.contact_gen.tool_source not in {TOOL_SOURCE_SELECTED_TOOLS, TOOL_SOURCE_OBJECTS}:
            errors.append(
                "ContactGenCfg.tool_source must be "
                f"{TOOL_SOURCE_SELECTED_TOOLS} or {TOOL_SOURCE_OBJECTS}"
            )
        if not isinstance(self.contact_gen.geometry_only, bool):
            errors.append("ContactGenCfg.geometry_only must be a bool")
        if not isinstance(self.contact_gen.require_tool_tip_anchor, bool):
            errors.append(
                "ContactGenCfg.require_tool_tip_anchor must be a bool"
            )
        if self.contact_gen.require_tool_tip_anchor:
            if self.contact_gen.tool_source != TOOL_SOURCE_SELECTED_TOOLS:
                errors.append(
                    "ContactGenCfg.require_tool_tip_anchor requires "
                    f"tool_source={TOOL_SOURCE_SELECTED_TOOLS}"
                )
            if self.contact_gen.contact_geometry_mode not in {
                CONTACT_GEOMETRY_ANCHOR_PAIR_REJECTION,
                CONTACT_GEOMETRY_INTERSECTING_ANCHORS,
                CONTACT_GEOMETRY_TANGENT_GAUSSIAN,
            }:
                errors.append(
                    "ContactGenCfg.require_tool_tip_anchor currently supports "
                    "only paper tangent-Gaussian, raw intersecting-anchor, "
                    "and nonpenetrating anchor-rejection geometry"
                )
        if not isinstance(self.contact_gen.rejection_refill, bool):
            errors.append("ContactGenCfg.rejection_refill must be a bool")
        if not isinstance(
            self.contact_gen.rejection_apply_tangent_gaussian,
            bool,
        ):
            errors.append(
                "ContactGenCfg.rejection_apply_tangent_gaussian must be a bool"
            )
        if not isinstance(self.contact_gen.balanced_tool_pairs, bool):
            errors.append("ContactGenCfg.balanced_tool_pairs must be a bool")
        if not isinstance(self.contact_gen.require_complete, bool):
            errors.append("ContactGenCfg.require_complete must be a bool")
        if not isinstance(self.contact_gen.precompute_convex_union_labels, bool):
            errors.append(
                "ContactGenCfg.precompute_convex_union_labels must be a bool"
            )
        if not isinstance(self.contact_gen.precompute_mesh_sdf, bool):
            errors.append("ContactGenCfg.precompute_mesh_sdf must be a bool")
        if (
            self.contact_gen.precompute_convex_union_labels
            and self.contact_gen.precompute_mesh_sdf
        ):
            errors.append(
                "Contact generation must precompute exactly one label "
                "representation, not convex-union labels and mesh SDF together"
            )
        if (
            self.contact_gen.precompute_mesh_sdf
            and not self.contact_gen.geometry_only
        ):
            errors.append(
                "ContactGenCfg.precompute_mesh_sdf requires geometry_only=True"
            )
        _require_positive_int(
            errors,
            "ContactGenCfg.rejection_max_rounds",
            self.contact_gen.rejection_max_rounds,
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
        if not isinstance(self.pretrain.use_geometry_candidates, bool):
            errors.append("PretrainCfg.use_geometry_candidates must be a bool")
        if not isinstance(self.pretrain.max_contacts_per_file, int) or self.pretrain.max_contacts_per_file < 0:
            errors.append("PretrainCfg.max_contacts_per_file must be a non-negative int")
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
        _require_positive_int(
            errors,
            "RLEnvCfg.object_solver_position_iteration_count",
            self.rl.env.object_solver_position_iteration_count,
        )
        _require_positive_int(
            errors,
            "RLEnvCfg.object_solver_velocity_iteration_count",
            self.rl.env.object_solver_velocity_iteration_count,
            allow_zero=True,
        )
        _require_positive_int(
            errors,
            "RLEnvCfg.articulation_solver_position_iteration_count",
            self.rl.env.articulation_solver_position_iteration_count,
        )
        _require_positive_int(
            errors,
            "RLEnvCfg.articulation_solver_velocity_iteration_count",
            self.rl.env.articulation_solver_velocity_iteration_count,
            allow_zero=True,
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
        if self.rl.env.max_depenetration_velocity < 0:
            errors.append("RLEnvCfg.max_depenetration_velocity must be >= 0")
        if not isinstance(self.rl.env.enable_ccd, bool):
            errors.append("RLEnvCfg.enable_ccd must be a bool")
        if not isinstance(self.rl.env.visualize_tool_pointcloud, bool):
            errors.append("RLEnvCfg.visualize_tool_pointcloud must be a bool")
        if self.rl.env.generated_parallel_finger_velocity_limit_m_s <= 0:
            errors.append(
                "RLEnvCfg.generated_parallel_finger_velocity_limit_m_s must be > 0"
            )
        if self.rl.observation.object_cloud_source not in {"preprocessed", "mesh_sampled"}:
            errors.append(
                "ObservationCfg.object_cloud_source must be preprocessed or mesh_sampled"
            )
        if (
            self.rl.observation.object_cloud_source == "preprocessed"
            and (
                not isinstance(self.rl.observation.object_cloud_preprocessed_dir, str)
                or not self.rl.observation.object_cloud_preprocessed_dir.strip()
            )
        ):
            errors.append(
                "ObservationCfg.object_cloud_preprocessed_dir must be a non-empty string "
                "when object_cloud_source=preprocessed"
            )
        if self.rl.env.contact_offset is not None and self.rl.env.contact_offset < 0:
            errors.append("RLEnvCfg.contact_offset must be >= 0 or None")
        if self.rl.env.rest_offset is not None and self.rl.env.rest_offset < 0:
            errors.append("RLEnvCfg.rest_offset must be >= 0 or None")
        if (
            self.rl.env.contact_offset is not None
            and self.rl.env.rest_offset is not None
            and self.rl.env.contact_offset < self.rl.env.rest_offset
        ):
            errors.append("RLEnvCfg.contact_offset must be >= rest_offset")
        if self.rl.env.episode_length_s <= 0:
            errors.append("RLEnvCfg.episode_length_s must be > 0")
        if self.rl.env.robot_mode not in {
            "tool",
            "bare_franka",
            "official_panda_gripper",
            "generated_gripper",
            "one_dof_gripper",
            "cross_embodiment_gripper",
        }:
            errors.append(
                "RLEnvCfg.robot_mode must be tool, bare_franka, official_panda_gripper, "
                "generated_gripper, one_dof_gripper, or cross_embodiment_gripper"
            )
        if self.rl.env.robot_mode == "cross_embodiment_gripper" and (
            not self.rl.launch.distributed or self.num_gpus < 2 or self.num_gpus % 2 != 0
        ):
            errors.append(
                "cross_embodiment_gripper requires distributed=True and an even num_gpus >= 2 "
                "to provide an exact 50/50 environment split"
            )
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
        if self.rl.reward.bimanual_wrist_surface_warning_height <= 0:
            errors.append("RewardCfg.bimanual_wrist_surface_warning_height must be > 0")
        if self.rl.reward.bimanual_wrist_surface_contact_height < 0:
            errors.append("RewardCfg.bimanual_wrist_surface_contact_height must be >= 0")
        if (
            self.rl.reward.bimanual_wrist_surface_warning_height
            <= self.rl.reward.bimanual_wrist_surface_contact_height
        ):
            errors.append(
                "RewardCfg.bimanual_wrist_surface_warning_height must be greater than "
                "bimanual_wrist_surface_contact_height"
            )
        if self.rl.reward.bimanual_tool_proximity_warning_clearance <= 0:
            errors.append("RewardCfg.bimanual_tool_proximity_warning_clearance must be > 0")
        if self.rl.reward.bimanual_tool_proximity_contact_clearance < 0:
            errors.append("RewardCfg.bimanual_tool_proximity_contact_clearance must be >= 0")
        if (
            self.rl.reward.bimanual_tool_proximity_warning_clearance
            <= self.rl.reward.bimanual_tool_proximity_contact_clearance
        ):
            errors.append(
                "RewardCfg.bimanual_tool_proximity_warning_clearance must be greater than "
                "bimanual_tool_proximity_contact_clearance"
            )
        _require_positive_int(
            errors,
            "RewardCfg.bimanual_tool_proximity_num_points",
            self.rl.reward.bimanual_tool_proximity_num_points,
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
        if self.pretrain.mode not in {
            "tce_multitask",
            "unicorn_contact",
            "oracle_contact",
            "oracle_pointmesh_contact",
            "oracle_pointcloud_diffusion",
            "oracle_pointcloud_postcontact",
        }:
            errors.append(
                "PretrainCfg.mode must be tce_multitask, unicorn_contact, oracle_contact, "
                "oracle_pointmesh_contact, oracle_pointcloud_diffusion, or "
                "oracle_pointcloud_postcontact"
            )
        if (
            self.pretrain.mode in {
                "unicorn_contact", "oracle_contact", "oracle_pointmesh_contact"
            }
            and self.pretrain.enabled_heads != ["contact"]
        ):
            errors.append(
                f"PretrainCfg.mode={self.pretrain.mode} requires enabled_heads=['contact']"
            )
        if (
            self.pretrain.mode == "oracle_pointcloud_diffusion"
            and self.pretrain.enabled_heads != ["diff"]
        ):
            errors.append(
                "PretrainCfg.mode=oracle_pointcloud_diffusion requires "
                "enabled_heads=['diff']"
            )
        if (
            self.pretrain.mode == "oracle_pointcloud_postcontact"
            and self.pretrain.enabled_heads != ["postcontact"]
        ):
            errors.append(
                "PretrainCfg.mode=oracle_pointcloud_postcontact requires "
                "enabled_heads=['postcontact']"
            )
        if self.pretrain.optimizer.name not in {"adamw", "sam"}:
            errors.append("PretrainOptimizerCfg.name must be adamw or sam")
        if self.pretrain.optimizer.sam_rho <= 0:
            errors.append("PretrainOptimizerCfg.sam_rho must be > 0")
        if self.pretrain.optimizer.max_gradient_norm <= 0:
            errors.append("PretrainOptimizerCfg.max_gradient_norm must be > 0")
        if self.pretrain.unicorn.decoder_type not in {
            "relu_mlp",
            "paper_cmlp_cbn",
        }:
            errors.append(
                "UnicornPretrainCfg.decoder_type must be relu_mlp or "
                "paper_cmlp_cbn"
            )
        if not isinstance(
            self.pretrain.unicorn.augment.paper_pair_augmentation, bool
        ):
            errors.append(
                "UnicornAugmentCfg.paper_pair_augmentation must be a bool"
            )
        if (
            self.pretrain.unicorn.positive_patch_fraction <= 0
            or self.pretrain.unicorn.positive_patch_fraction >= 1
        ):
            errors.append("UnicornPretrainCfg.positive_patch_fraction must be in (0, 1)")
        if self.pretrain.unicorn.label.source not in {
            "mesh_sdf",
            "precomputed_convex_union",
            "precomputed_mesh_sdf",
        }:
            errors.append(
                "UnicornLabelCfg.source must be mesh_sdf, "
                "precomputed_convex_union, or precomputed_mesh_sdf"
            )
        if (
            self.pretrain.unicorn.label.source
            in {"precomputed_convex_union", "precomputed_mesh_sdf"}
            and self.pretrain.mode != "tce_multitask"
        ):
            errors.append(
                "Precomputed contact labels require "
                "PretrainCfg.mode=tce_multitask"
            )
        if (
            self.contact_gen.enabled
            and self.pretrain.use_geometry_candidates
            and self.pretrain.unicorn.label.source == "precomputed_mesh_sdf"
            and not self.contact_gen.precompute_mesh_sdf
        ):
            errors.append(
                "precomputed_mesh_sdf training requires contact generation "
                "with precompute_mesh_sdf=True"
            )
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
        if self.rl.object_pose_sampling.secondary_task not in {"random_pose", "grasp_lift"}:
            errors.append("ObjectPoseSamplingCfg.secondary_task must be random_pose or grasp_lift")
        if self.rl.object_pose_sampling.grasp_lift_height <= 0:
            errors.append("ObjectPoseSamplingCfg.grasp_lift_height must be > 0")
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
        if self.rl.env.robot_mode == "official_panda_gripper":
            if self.rl.observation.tool_cloud_source not in {
                "official_panda_gripper_kinematic_mesh",
                "official_panda_gripper_primitives",
                "official_panda_gripper_meshes",
            }:
                errors.append(
                    "official_panda_gripper requires "
                    "ObservationCfg.tool_cloud_source=official_panda_gripper_kinematic_mesh"
                )
        elif self.rl.env.robot_mode == "generated_gripper":
            if self.rl.observation.tool_cloud_source != "gripper_cloud_cache_v1":
                errors.append(
                    "generated_gripper requires "
                    "ObservationCfg.tool_cloud_source=gripper_cloud_cache_v1"
                )
        elif self.rl.env.robot_mode == "one_dof_gripper":
            if self.rl.observation.tool_cloud_source != "gripper_cloud_cache_v1":
                errors.append(
                    "one_dof_gripper requires "
                    "ObservationCfg.tool_cloud_source=gripper_cloud_cache_v1"
                )
        elif self.rl.env.robot_mode == "cross_embodiment_gripper":
            if self.rl.observation.tool_cloud_source != "gripper_cloud_cache_v1":
                errors.append(
                    "cross_embodiment_gripper requires "
                    "ObservationCfg.tool_cloud_source=gripper_cloud_cache_v1"
                )
        elif self.rl.observation.tool_cloud_source != "adjusted_decomposed_mesh":
            errors.append(
                "ObservationCfg.tool_cloud_source must be adjusted_decomposed_mesh"
            )
        if self.rl.observation.model_input_centering not in {"bbox_center", "object_center"}:
            errors.append("ObservationCfg.model_input_centering must be bbox_center or object_center")
        if self.pretrain.encoder_input_centering not in {"bbox_center", "object_center"}:
            errors.append("PretrainCfg.encoder_input_centering must be bbox_center or object_center")
        encoder_backend = self.model.encoder_backend.strip().lower()
        if encoder_backend in {"tg"}:
            encoder_backend = "tce"
        if encoder_backend in {"p2v"}:
            encoder_backend = "point2vec"
        if encoder_backend in {"corn"}:
            encoder_backend = "icp"
        if encoder_backend not in {
            "tce", "point2vec", "icp", "unicorn", "oracle_patch",
            "oracle_pointmesh_pointnet", "oracle_pointcloud_pointnet",
            "oracle_pointcloud_patch_oracle", "patch_distance_pointnet"
        }:
            errors.append(
                "ModelCfg.encoder_backend must be tce, point2vec, icp, unicorn, "
                "oracle_patch, oracle_pointmesh_pointnet, oracle_pointcloud_pointnet, "
                "oracle_pointcloud_patch_oracle, or patch_distance_pointnet"
            )
        allowed_adapters = {
            "tce_strict",
            "point2vec_native",
            "icp_legacy",
            "unicorn_strict",
            "oracle_none",
            "oracle_pointmesh_pointnet_strict",
            "oracle_pointcloud_pointnet_strict",
            "oracle_pointcloud_pointnet_pretrain_strict",
            "oracle_pointcloud_pointnet_normalized_pretrain_strict",
            "oracle_pointcloud_pointnet_rl_encoder_strict",
            "oracle_pointcloud_patch_oracle_strict",
            "patch_distance_pointnet_strict",
        }
        if self.model.pretrained_encoder.adapter not in allowed_adapters:
            errors.append(
                "ModelCfg.pretrained_encoder.adapter must be one of "
                f"{sorted(allowed_adapters)}"
            )
        allowed_actors = {
            "ActorCriticTG",
            "ActorCriticTGOutputGate",
            "ActorCriticTGSM",
            "ActorCriticTGHAMNet",
            "ActorCriticTGUnicorn",
            "ActorCriticTGBimanual",
            "ActorCriticPoint2Vec",
            "ActorCriticICP",
        }
        if self.rl.actor_critic_class not in allowed_actors:
            errors.append(
                "RLCfg.actor_critic_class must be one of "
                f"{sorted(allowed_actors)}"
            )
        if encoder_backend == "tce":
            if self.rl.actor_critic_class not in {
                "ActorCriticTG",
                "ActorCriticTGOutputGate",
                "ActorCriticTGSM",
                "ActorCriticTGHAMNet",
                "ActorCriticTGBimanual",
            }:
                errors.append(
                    "encoder_backend=tce requires RLCfg.actor_critic_class=ActorCriticTG, "
                    "ActorCriticTGOutputGate, ActorCriticTGSM, ActorCriticTGHAMNet, "
                    "or ActorCriticTGBimanual"
                )
            if self.model.pretrained_encoder.adapter != "tce_strict":
                errors.append("encoder_backend=tce requires pretrained_encoder.adapter=tce_strict")
            if self.model.tce.rl_token_source not in {"encoder", "contact_head_hidden"}:
                errors.append("TCECfg.rl_token_source must be encoder or contact_head_hidden")
            pca_rank = int(self.model.tce.encoder_token_pca_rank)
            if not 1 <= pca_rank <= int(self.model.tce.encoder_channel):
                errors.append("TCECfg.encoder_token_pca_rank must be in [1, encoder_channel]")
            if pca_rank < int(self.model.tce.encoder_channel) and not self.model.tce.encoder_token_pca_path:
                errors.append("TCE encoder-token PCA requires encoder_token_pca_path")
            bottleneck_rank = int(self.model.tce.encoder_token_bottleneck_rank)
            if not 1 <= bottleneck_rank <= int(self.model.tce.encoder_channel):
                errors.append(
                    "TCECfg.encoder_token_bottleneck_rank must be in [1, encoder_channel]"
                )
            if (
                bottleneck_rank < int(self.model.tce.encoder_channel)
                and not self.model.tce.encoder_token_bottleneck_pca_path
            ):
                errors.append(
                    "TCE encoder-token bottleneck requires encoder_token_bottleneck_pca_path"
                )
            if (
                bottleneck_rank < int(self.model.tce.encoder_channel)
                and pca_rank < int(self.model.tce.encoder_channel)
            ):
                errors.append(
                    "TCE fixed PCA and trainable encoder-token bottleneck cannot both be enabled"
                )
            if (
                bottleneck_rank < int(self.model.tce.encoder_channel)
                and self.model.tce.rl_token_source != "encoder"
            ):
                errors.append(
                    "TCE encoder-token bottleneck requires rl_token_source=encoder"
                )
            kinematic = self.model.tce.kinematic_conditioning
            if kinematic.enabled:
                if tuple(kinematic.state_fractions) != (0.0, 0.5, 1.0):
                    errors.append(
                        "TCE kinematic conditioning requires state_fractions=(0.0, 0.5, 1.0)"
                    )
                if int(kinematic.attention_layers) < 1:
                    errors.append(
                        "TCE kinematic conditioning attention_layers must be >= 1"
                    )
                if float(kinematic.delta_std) <= 0.0:
                    errors.append(
                        "TCE kinematic conditioning delta_std must be > 0"
                    )
                if self.model.tce.rl_token_source != "encoder":
                    errors.append(
                        "TCE kinematic conditioning requires rl_token_source=encoder"
                    )
                if self.rl.env.robot_mode not in {
                    "generated_gripper",
                    "one_dof_gripper",
                    "cross_embodiment_gripper",
                }:
                    errors.append(
                        "TCE kinematic conditioning requires generated, one-DoF, "
                        "or cross-embodiment gripper RL"
                    )
                if not self.rl.observation.include_kinematic_gripper_clouds:
                    errors.append(
                        "TCE kinematic conditioning requires kinematic gripper clouds"
                    )
                if (
                    "kinematic_gripper_clouds_flat"
                    not in self.rl.observation.layout
                ):
                    errors.append(
                        "TCE kinematic conditioning cloud is absent from observation layout"
                    )
                if set(self.pretrain.enabled_heads) != {"contact"}:
                    errors.append(
                        "TCE kinematic conditioning requires contact-only pretraining"
                    )
            elif self.rl.observation.include_kinematic_gripper_clouds:
                errors.append(
                    "Kinematic gripper clouds require TCE kinematic conditioning"
                )
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
            if self.rl.env.robot_mode not in {"bare_franka", "official_panda_gripper"}:
                errors.append(
                    "ActorCriticICP requires RLEnvCfg.robot_mode=bare_franka "
                    "or official_panda_gripper"
                )
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
        if encoder_backend == "unicorn":
            if self.rl.enabled and self.rl.actor_critic_class != "ActorCriticTG":
                errors.append("encoder_backend=unicorn requires shared RLCfg.actor_critic_class=ActorCriticTG")
            if self.model.pretrained_encoder.adapter != "unicorn_strict":
                errors.append("encoder_backend=unicorn requires pretrained_encoder.adapter=unicorn_strict")
            if self.pretrain.encoder_input_centering != "object_center":
                errors.append("encoder_backend=unicorn requires PretrainCfg.encoder_input_centering=object_center")
            if self.rl.observation.model_input_centering != "object_center":
                errors.append("encoder_backend=unicorn requires ObservationCfg.model_input_centering=object_center")
            if self.model.unicorn.rl_token_source not in {"encoder", "contact_head_hidden"}:
                errors.append(
                    "UnicornCfg.rl_token_source must be encoder or contact_head_hidden"
                )
        if encoder_backend == "patch_distance_pointnet":
            if self.rl.enabled and self.rl.actor_critic_class != "ActorCriticTG":
                errors.append(
                    "encoder_backend=patch_distance_pointnet requires "
                    "RLCfg.actor_critic_class=ActorCriticTG"
                )
            if self.model.pretrained_encoder.adapter != "patch_distance_pointnet_strict":
                errors.append(
                    "encoder_backend=patch_distance_pointnet requires "
                    "pretrained_encoder.adapter=patch_distance_pointnet_strict"
                )
            patch_distance = self.model.patch_distance_pointnet
            if min(
                patch_distance.num_points,
                patch_distance.num_patches,
                patch_distance.patch_size,
                patch_distance.encoder_channel,
                patch_distance.query_count,
                patch_distance.supervised_patches_per_cloud,
            ) <= 0:
                errors.append("PatchDistancePointNetCfg dimensions must be > 0")
            if patch_distance.supervised_patches_per_cloud > patch_distance.num_patches:
                errors.append(
                    "PatchDistancePointNetCfg.supervised_patches_per_cloud must be "
                    "<= num_patches"
                )
            if not (
                0.0
                < patch_distance.query_min_offset_m
                < patch_distance.query_max_offset_m
            ):
                errors.append(
                    "PatchDistancePointNetCfg query offsets must satisfy 0 < min < max"
                )
            if min(
                patch_distance.point_scale_m,
                patch_distance.distance_scale_m,
                patch_distance.patch_center_scale_m,
            ) <= 0.0:
                errors.append("PatchDistancePointNetCfg metric scales must be > 0")
        if encoder_backend == "oracle_patch":
            if self.rl.actor_critic_class != "ActorCriticTG":
                errors.append("encoder_backend=oracle_patch requires RLCfg.actor_critic_class=ActorCriticTG")
            if self.model.pretrained_encoder.adapter != "oracle_none":
                errors.append("encoder_backend=oracle_patch requires pretrained_encoder.adapter=oracle_none")
            if self.pretrain.enabled and self.pretrain.mode != "oracle_contact":
                errors.append("encoder_backend=oracle_patch pretraining requires mode=oracle_contact")
            if self.rl.enabled and self.rl.freeze_encoder:
                errors.append("encoder_backend=oracle_patch requires RLCfg.freeze_encoder=False")
            if self.rl.enabled and not self.rl.observation.include_oracle_mesh_sdf:
                errors.append("encoder_backend=oracle_patch requires exact oracle mesh SDF observation")
            if self.rl.enabled and "oracle_mesh_signed_sdf" not in self.rl.observation.layout:
                errors.append("encoder_backend=oracle_patch requires oracle_mesh_signed_sdf in observation layout")
            elif self.rl.enabled and self.rl.observation.layout.index("oracle_mesh_signed_sdf") != 2:
                errors.append(
                    "oracle_mesh_signed_sdf must follow object/tool clouds in observation layout"
                )
            oracle = self.model.oracle_patch
            if not isinstance(oracle.include_contact_feature, bool):
                errors.append("OraclePatchCfg.include_contact_feature must be a bool")
            if oracle.contact_eps < 0:
                errors.append("OraclePatchCfg.contact_eps must be >= 0")
            if oracle.center_scale_m <= 0 or oracle.distance_scale_m <= 0:
                errors.append("OraclePatchCfg metric normalization scales must be > 0")
            if oracle.patch_relative_scale_m <= 0:
                errors.append("OraclePatchCfg.patch_relative_scale_m must be > 0")
            if min(
                oracle.log_distance_resolution_m,
                oracle.log_distance_cap_m,
            ) <= 0:
                errors.append("OraclePatchCfg distance normalization scales must be > 0")
            if oracle.normalization_clip <= 0:
                errors.append("OraclePatchCfg.normalization_clip must be > 0")
        if encoder_backend == "oracle_pointmesh_pointnet":
            if self.rl.actor_critic_class != "ActorCriticTG":
                errors.append(
                    "encoder_backend=oracle_pointmesh_pointnet requires "
                    "RLCfg.actor_critic_class=ActorCriticTG"
                )
            if self.model.pretrained_encoder.adapter != "oracle_pointmesh_pointnet_strict":
                errors.append(
                    "encoder_backend=oracle_pointmesh_pointnet requires "
                    "pretrained_encoder.adapter=oracle_pointmesh_pointnet_strict"
                )
            if self.pretrain.enabled and self.pretrain.mode != "oracle_pointmesh_contact":
                errors.append(
                    "oracle_pointmesh_pointnet pretraining requires mode=oracle_pointmesh_contact"
                )
            if self.rl.enabled and not self.rl.observation.include_oracle_mesh_unsigned_distance:
                errors.append(
                    "oracle_pointmesh_pointnet requires unsigned mesh-distance observation"
                )
            if self.rl.enabled and "oracle_mesh_unsigned_distance" not in self.rl.observation.layout:
                errors.append(
                    "oracle_pointmesh_pointnet requires oracle_mesh_unsigned_distance in layout"
                )
            elif (
                self.rl.enabled
                and self.rl.observation.layout.index("oracle_mesh_unsigned_distance") != 2
            ):
                errors.append(
                    "oracle_mesh_unsigned_distance must follow object/tool clouds in layout"
                )
            pointmesh = self.model.oracle_pointmesh_pointnet
            if min(
                pointmesh.coordinate_scale_m,
                pointmesh.distance_scale_m,
                pointmesh.normalization_clip,
            ) <= 0:
                errors.append("OraclePointMeshPointNetCfg normalization scales must be > 0")
        if encoder_backend == "oracle_pointcloud_pointnet":
            if self.rl.actor_critic_class != "ActorCriticTG":
                errors.append(
                    "encoder_backend=oracle_pointcloud_pointnet requires "
                    "RLCfg.actor_critic_class=ActorCriticTG"
                )
            pointcloud = self.model.oracle_pointcloud_pointnet
            allowed_pointcloud_adapters = (
                {
                    "oracle_pointcloud_pointnet_strict",
                    "oracle_pointcloud_pointnet_pretrain_strict",
                    "oracle_pointcloud_pointnet_normalized_pretrain_strict",
                    "oracle_pointcloud_pointnet_rl_encoder_strict",
                }
                if pointcloud.load_fitted_weights
                else {
                    "oracle_none",
                    # The explicit scratch control may reuse only the fitted
                    # source's input normalization while leaving all learned
                    # PointNet parameters randomly initialized.
                    "oracle_pointcloud_pointnet_strict",
                }
            )
            if (
                self.model.pretrained_encoder.adapter
                not in allowed_pointcloud_adapters
            ):
                errors.append(
                    "encoder_backend=oracle_pointcloud_pointnet requires "
                    "pretrained_encoder.adapter in "
                    f"{sorted(allowed_pointcloud_adapters)}"
                )
            if (
                self.pretrain.enabled
                and self.pretrain.mode
                not in {
                    "oracle_pointcloud_diffusion",
                    "oracle_pointcloud_postcontact",
                }
            ):
                errors.append(
                    "oracle_pointcloud_pointnet pretraining requires "
                    "mode=oracle_pointcloud_diffusion or "
                    "oracle_pointcloud_postcontact"
                )
            if self.pretrain.enabled:
                if pointcloud.feature_mode != "fast11":
                    errors.append(
                        "native PointNet pretraining requires feature_mode=fast11"
                    )
                if pointcloud.use_rank10_bottleneck:
                    errors.append(
                        "native PointNet pretraining requires "
                        "use_rank10_bottleneck=False"
                    )
                if pointcloud.token_mode != "patches":
                    errors.append(
                        "native PointNet pretraining requires token_mode=patches"
                    )
            if min(
                pointcloud.num_points,
                pointcloud.num_patches,
                pointcloud.patch_size,
                pointcloud.encoder_channel,
                pointcloud.nearest_frame_batch_size,
            ) <= 0:
                errors.append("OraclePointCloudPointNetCfg dimensions must be > 0")
            if pointcloud.encoder_channel != 128:
                errors.append(
                    "OraclePointCloudPointNetCfg.encoder_channel must be 128"
                )
            if pointcloud.token_mode not in {"patches", "points"}:
                errors.append(
                    "OraclePointCloudPointNetCfg.token_mode must be patches or points"
                )
            if not isinstance(pointcloud.load_fitted_weights, bool):
                errors.append(
                    "OraclePointCloudPointNetCfg.load_fitted_weights must be a bool"
                )
            if not isinstance(pointcloud.use_rank10_bottleneck, bool):
                errors.append(
                    "OraclePointCloudPointNetCfg.use_rank10_bottleneck must be a bool"
                )
            if pointcloud.feature_mode not in {"fast11", "rich21"}:
                errors.append(
                    "OraclePointCloudPointNetCfg.feature_mode must be fast11 or rich21"
                )
            if pointcloud.input_normalization not in {
                "identity",
                "fast11_probe_v1",
            }:
                errors.append(
                    "OraclePointCloudPointNetCfg.input_normalization must be "
                    "identity or fast11_probe_v1"
                )
            if (
                pointcloud.input_normalization == "fast11_probe_v1"
                and pointcloud.feature_mode != "fast11"
            ):
                errors.append(
                    "fast11_probe_v1 input normalization requires feature_mode=fast11"
                )
            if pointcloud.load_fitted_weights:
                if pointcloud.feature_mode != "fast11":
                    errors.append(
                        "pretrained oracle point-cloud weights require "
                        "feature_mode=fast11"
                    )
                if (
                    self.model.pretrained_encoder.adapter
                    == "oracle_pointcloud_pointnet_strict"
                    and not pointcloud.use_rank10_bottleneck
                ):
                    errors.append(
                        "fitted probe weights require use_rank10_bottleneck=True"
                    )
                if (
                    self.model.pretrained_encoder.adapter
                    == "oracle_pointcloud_pointnet_pretrain_strict"
                    and pointcloud.use_rank10_bottleneck
                ):
                    errors.append(
                        "native PointNet pretrain weights require "
                        "use_rank10_bottleneck=False"
                    )
                if (
                    self.model.pretrained_encoder.adapter
                    == "oracle_pointcloud_pointnet_normalized_pretrain_strict"
                    and (
                        pointcloud.use_rank10_bottleneck
                        or pointcloud.input_normalization != "fast11_probe_v1"
                    )
                ):
                    errors.append(
                        "normalized native PointNet pretrain weights require "
                        "use_rank10_bottleneck=False and "
                        "input_normalization=fast11_probe_v1"
                    )
            if pointcloud.feature_mode == "rich21" and pointcloud.token_mode != "patches":
                errors.append("rich21 point-cloud features require token_mode=patches")
        if encoder_backend == "oracle_pointcloud_patch_oracle":
            if self.rl.actor_critic_class != "ActorCriticTG":
                errors.append(
                    "encoder_backend=oracle_pointcloud_patch_oracle requires "
                    "RLCfg.actor_critic_class=ActorCriticTG"
                )
            if (
                self.model.pretrained_encoder.adapter
                != "oracle_pointcloud_patch_oracle_strict"
            ):
                errors.append(
                    "encoder_backend=oracle_pointcloud_patch_oracle requires "
                    "pretrained_encoder.adapter=oracle_pointcloud_patch_oracle_strict"
                )
            if self.pretrain.enabled:
                errors.append(
                    "oracle_pointcloud_patch_oracle uses its fitted probe checkpoint "
                    "and does not run canonical pretraining"
                )
            oracle = self.model.oracle_pointcloud_patch_oracle
            if (
                oracle.num_points,
                oracle.num_patches,
                oracle.patch_size,
                oracle.encoder_channel,
            ) != (512, 16, 32, 128):
                errors.append(
                    "OraclePointCloudPatchOracleCfg requires num_points=512, "
                    "num_patches=16, patch_size=32, encoder_channel=128"
                )
            if oracle.nearest_frame_batch_size <= 0:
                errors.append(
                    "OraclePointCloudPatchOracleCfg.nearest_frame_batch_size must be > 0"
                )
        if self.pretrain.enabled and self.pretrain.mode == "unicorn_contact" and encoder_backend != "unicorn":
            errors.append("PretrainCfg.mode=unicorn_contact requires ModelCfg.encoder_backend=unicorn")
        if self.pretrain.enabled and self.pretrain.mode == "oracle_contact" and encoder_backend != "oracle_patch":
            errors.append("PretrainCfg.mode=oracle_contact requires ModelCfg.encoder_backend=oracle_patch")
        if (
            self.pretrain.enabled
            and self.pretrain.mode == "oracle_pointmesh_contact"
            and encoder_backend != "oracle_pointmesh_pointnet"
        ):
            errors.append(
                "PretrainCfg.mode=oracle_pointmesh_contact requires "
                "ModelCfg.encoder_backend=oracle_pointmesh_pointnet"
            )
        if self.pretrain.enabled and self.pretrain.mode == "tce_multitask" and encoder_backend != "tce":
            errors.append("PretrainCfg.enabled currently supports only TCE for tce_multitask")
        if (
            self.pretrain.enabled
            and self.pretrain.mode
            in {
                "oracle_pointcloud_diffusion",
                "oracle_pointcloud_postcontact",
            }
            and encoder_backend != "oracle_pointcloud_pointnet"
        ):
            errors.append(
                "PointNet pretraining requires "
                "ModelCfg.encoder_backend=oracle_pointcloud_pointnet"
            )
        if (
            self.pretrain.enabled
            and self.pretrain.mode == "tce_multitask"
            and self.rl.enabled
            and self.rl.actor_critic_class
            not in {
                "ActorCriticTG",
                "ActorCriticTGOutputGate",
                "ActorCriticTGSM",
                "ActorCriticTGHAMNet",
                "ActorCriticTGBimanual",
            }
        ):
            errors.append(
                "TCE RL requires ActorCriticTG/ActorCriticTGOutputGate/"
                "ActorCriticTGSM/ActorCriticTGHAMNet/ActorCriticTGBimanual"
            )
        if self.rl.actor_critic_class == "ActorCriticTGOutputGate":
            if not self.rl.output_gate_expert_a_checkpoint:
                errors.append("ActorCriticTGOutputGate requires RLCfg.output_gate_expert_a_checkpoint")
            if not self.rl.output_gate_expert_b_checkpoint:
                errors.append("ActorCriticTGOutputGate requires RLCfg.output_gate_expert_b_checkpoint")
            if len(tuple(self.rl.output_gate_hidden_dims)) == 0:
                errors.append("RLCfg.output_gate_hidden_dims must contain at least one hidden dim")
            if any(int(dim) <= 0 for dim in self.rl.output_gate_hidden_dims):
                errors.append("RLCfg.output_gate_hidden_dims entries must be > 0")
            if not 0.0 < float(self.rl.output_gate_initial_expert_a_weight) < 1.0:
                errors.append("RLCfg.output_gate_initial_expert_a_weight must be in (0, 1)")
        if self.rl.actor_critic_class == "ActorCriticTGSM":
            if not self.rl.separate_actor_critic_fusion:
                errors.append("ActorCriticTGSM requires RLCfg.separate_actor_critic_fusion=True")
            if self.rl.observation.task_embedding_dim != 2:
                errors.append("ActorCriticTGSM requires ObservationCfg.task_embedding_dim=2")
            if "task_embedding" not in self.rl.observation.layout:
                errors.append("ActorCriticTGSM requires 'task_embedding' in ObservationCfg.layout")
        if self.rl.actor_critic_class == "ActorCriticTGHAMNet":
            if not self.rl.separate_actor_critic_fusion:
                errors.append(
                    "ActorCriticTGHAMNet requires RLCfg.separate_actor_critic_fusion=True"
                )
            if self.rl.hamnet_num_modules < 2:
                errors.append("RLCfg.hamnet_num_modules must be >= 2")
            if not self.rl.hamnet_hidden_dims or any(
                int(dim) <= 0 for dim in self.rl.hamnet_hidden_dims
            ):
                errors.append("RLCfg.hamnet_hidden_dims must contain positive dimensions")
            if not self.rl.hamnet_router_hidden_dims or any(
                int(dim) <= 0 for dim in self.rl.hamnet_router_hidden_dims
            ):
                errors.append(
                    "RLCfg.hamnet_router_hidden_dims must contain positive dimensions"
                )
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
            "tangent_translation_noise_std",
            "tangent_rotation_noise_std_rad",
        ):
            if getattr(self.contact_gen, field_name) < 0:
                errors.append(f"ContactGenCfg.{field_name} must be >= 0")
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
        if self.model.policy_fusion.cross_attn_token_order not in {
            "joint",
            "tool_then_object",
        }:
            errors.append(
                "PolicyFusionCfg.cross_attn_token_order must be joint or "
                "tool_then_object"
            )
        if (
            self.model.policy_fusion.cross_attn_token_order == "tool_then_object"
            and self.model.policy_fusion.cross_attn_layers != 2
        ):
            errors.append(
                "PolicyFusionCfg.cross_attn_token_order=tool_then_object requires "
                "cross_attn_layers=2"
            )
        if (
            self.model.policy_fusion.cross_attn_token_order == "tool_then_object"
            and self.model.policy_fusion.reuse_pretrain_pose_cross_attn
        ):
            errors.append(
                "tool_then_object fusion cannot reuse joint pretrain cross-attention weights"
            )
        for index, hidden_dim in enumerate(self.model.policy_fusion.fusion_hidden_dims):
            _require_positive_int(errors, f"PolicyFusionCfg.fusion_hidden_dims[{index}]", hidden_dim)

        try:
            active_encoder_checkpoint = self.model.encoder.checkpoint_path
        except ValueError:
            active_encoder_checkpoint = None
        scratch_pointcloud = (
            encoder_backend == "oracle_pointcloud_pointnet"
            and not self.model.oracle_pointcloud_pointnet.load_fitted_weights
        )
        if self.rl.enabled and encoder_backend != "oracle_patch" and not scratch_pointcloud and not (
            self.rl.encoder_checkpoint
            or self.model.pretrained_encoder.checkpoint_path
            or active_encoder_checkpoint
            or self.pretrain.enabled
            or self.pretrain_reuse
        ):
            errors.append(
                "RLCfg.enabled requires an encoder checkpoint or PretrainCfg.enabled"
            )
        if self.rl.init_checkpoint and self.rl.resume_checkpoint:
            errors.append(
                "RLCfg.init_checkpoint and RLCfg.resume_checkpoint are mutually exclusive"
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
