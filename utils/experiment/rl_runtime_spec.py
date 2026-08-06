"""Lightweight RL runtime-spec loader shared by Isaac and RSL-RL bridges."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


RUNTIME_SPEC_ENV_VAR = "TOOL_GENERALIST_RL_RUNTIME_SPEC"
RUNTIME_SPEC_FILENAME = "rl_runtime_spec.json"
SUPPORTED_POLICY_CLASSES = {
    "ActorCriticTG",
    "ActorCriticTGOutputGate",
    "ActorCriticTGSM",
    "ActorCriticTGHAMNet",
    "ActorCriticTGUnicorn",
    "ActorCriticTGBimanual",
    "ActorCriticPoint2Vec",
    "ActorCriticICP",
}
SCRIPTED_GRASPGEN_MODES = {
    "graspgen_direct_grasp",
    "graspgen_direct_grasp_eval",
    "graspgen_lift_grasps_eval",
}


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
    fine_grained_timing = launch.get("print_fine_grained_timing", False)
    if not isinstance(fine_grained_timing, bool):
        raise RuntimeError(
            "RL runtime spec launch_params.print_fine_grained_timing must be a bool"
        )
    if bool(launch.get("distributed", False)) and num_gpus <= 0:
        raise RuntimeError("RL distributed launch requires num_gpus > 0")
    mode = str(spec.get("mode", ""))
    policy_preview = spec.get("policy_params")
    oracle_backend = isinstance(policy_preview, Mapping) and policy_preview.get("encoder_backend") == "oracle_patch"
    scratch_pointcloud = (
        isinstance(policy_preview, Mapping)
        and policy_preview.get("encoder_backend") == "oracle_pointcloud_pointnet"
        and policy_preview.get("oracle_pointcloud_load_fitted_weights") is False
    )
    checkpoint_optional = mode in SCRIPTED_GRASPGEN_MODES or oracle_backend or scratch_pointcloud
    if not spec["encoder_checkpoint"] and not checkpoint_optional:
        raise RuntimeError(
            f"{spec['actor_critic_class']} requires encoder_checkpoint in RL runtime spec"
        )
    resume_checkpoint = spec.get("rl_resume_checkpoint")
    if resume_checkpoint is not None and (
        not isinstance(resume_checkpoint, str) or not resume_checkpoint.strip()
    ):
        raise RuntimeError("RL runtime spec rl_resume_checkpoint must be a non-empty string or None")
    init_checkpoint = spec.get("rl_init_checkpoint")
    if init_checkpoint is not None and (
        not isinstance(init_checkpoint, str) or not init_checkpoint.strip()
    ):
        raise RuntimeError("RL runtime spec rl_init_checkpoint must be a non-empty string or None")
    if init_checkpoint is not None and resume_checkpoint is not None:
        raise RuntimeError(
            "RL runtime spec rl_init_checkpoint and rl_resume_checkpoint are mutually exclusive"
        )
    if len(tuple(spec["physics_observation_fields"])) != int(spec["physics_dim"]):
        raise RuntimeError(
            "RL runtime spec physics_dim must match len(physics_observation_fields)"
        )
    table = _require_mapping(spec, "table_params", path)
    env_params = _require_mapping(spec, "env_params", path)
    if not isinstance(env_params.get("visualize_tool_pointcloud", True), bool):
        raise RuntimeError(
            "RL runtime spec env_params.visualize_tool_pointcloud must be a bool"
        )
    try:
        generated_parallel_velocity_limit = float(
            env_params["generated_parallel_finger_velocity_limit_m_s"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "RL runtime spec "
            "env_params.generated_parallel_finger_velocity_limit_m_s "
            "must be a number"
        ) from exc
    if generated_parallel_velocity_limit <= 0:
        raise RuntimeError(
            "RL runtime spec "
            "env_params.generated_parallel_finger_velocity_limit_m_s must be > 0"
        )
    robot_mode = str(env_params.get("robot_mode", "tool"))
    if robot_mode not in {
        "tool",
        "bare_franka",
        "official_panda_gripper",
        "generated_gripper",
        "one_dof_gripper",
        "cross_embodiment_gripper",
    }:
        raise RuntimeError(
            "RL runtime spec env_params.robot_mode must be tool, bare_franka, "
            "official_panda_gripper, generated_gripper, one_dof_gripper, or "
            "cross_embodiment_gripper"
        )
    if spec["actor_critic_class"] == "ActorCriticICP" and robot_mode not in {
        "bare_franka",
        "official_panda_gripper",
    }:
        raise RuntimeError(
            "ActorCriticICP requires env_params.robot_mode=bare_franka "
            "or official_panda_gripper"
        )
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

    curriculum = spec.get("curriculum_params")
    if curriculum is not None:
        curriculum = _require_mapping(spec, "curriculum_params", path)
        if not isinstance(curriculum.get("enabled", False), bool):
            raise RuntimeError("RL runtime spec curriculum_params.enabled must be a bool")
        for key in ("start_step", "end_step"):
            try:
                value = int(curriculum.get(key, 0))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"RL runtime spec curriculum_params.{key} must be an integer") from exc
            if value < 0:
                raise RuntimeError(f"RL runtime spec curriculum_params.{key} must be >= 0")
        if int(curriculum.get("end_step", 0)) < int(curriculum.get("start_step", 0)):
            raise RuntimeError("RL runtime spec curriculum_params.end_step must be >= start_step")
        for key in ("start_stable_pose_probability", "end_stable_pose_probability"):
            try:
                value = float(curriculum.get(key, 0.0))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"RL runtime spec curriculum_params.{key} must be a number") from exc
            if value < 0.0 or value > 1.0:
                raise RuntimeError(f"RL runtime spec curriculum_params.{key} must be in [0, 1]")

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
    if robot_mode != "tool" and any(field.startswith("tool_") for field in physics_fields):
        raise RuntimeError("RL runtime spec cannot request tool_* physics fields unless robot_mode=tool")
    if bool(table.get("enabled", False)) and any(
        field.startswith("ground_") for field in physics_fields
    ):
        raise RuntimeError(
            "RL runtime spec cannot request ground_* physics fields when table is "
            "enabled because the ground plane is removed."
        )

    policy = _require_mapping(spec, "policy_params", path)
    cross_attn_token_order = str(policy.get("cross_attn_token_order", "joint"))
    if cross_attn_token_order not in {"joint", "tool_then_object"}:
        raise RuntimeError(
            "RL runtime spec policy_params.cross_attn_token_order must be joint or "
            "tool_then_object"
        )
    if (
        cross_attn_token_order == "tool_then_object"
        and int(policy.get("cross_attn_layers", 0)) != 2
    ):
        raise RuntimeError(
            "RL runtime spec tool_then_object fusion requires cross_attn_layers=2"
        )
    observation = _require_mapping(spec, "observation_params", path)
    object_cloud_source = str(observation.get("object_cloud_source", "preprocessed"))
    if object_cloud_source not in {"preprocessed", "mesh_sampled"}:
        raise RuntimeError(
            "RL runtime spec observation_params.object_cloud_source must be "
            "preprocessed or mesh_sampled"
        )
    object_cloud_preprocessed_dir = observation.get(
        "object_cloud_preprocessed_dir",
        "/mnt/project/world_model/tool_generalist/assets/DGN/first_hit_fps_pointclouds/npy",
    )
    if object_cloud_source == "preprocessed" and (
        not isinstance(object_cloud_preprocessed_dir, str)
        or not object_cloud_preprocessed_dir.strip()
    ):
        raise RuntimeError(
            "RL runtime spec observation_params.object_cloud_preprocessed_dir must be "
            "a non-empty string when object_cloud_source=preprocessed"
        )
    if robot_mode == "bare_franka" and bool(observation.get("include_tool_cloud", False)):
        raise RuntimeError("RL runtime spec cannot include tool cloud when robot_mode=bare_franka")
    include_kinematic_clouds = bool(
        observation.get("include_kinematic_gripper_clouds", False)
    )
    policy_kinematic = bool(policy.get("kinematic_conditioning", False))
    if include_kinematic_clouds != policy_kinematic:
        raise RuntimeError(
            "RL runtime spec kinematic cloud observation must exactly match "
            "policy_params.kinematic_conditioning"
        )
    if include_kinematic_clouds:
        if robot_mode not in {
            "generated_gripper",
            "one_dof_gripper",
            "cross_embodiment_gripper",
        }:
            raise RuntimeError(
                "Kinematic gripper-state clouds require robot_mode to be "
                "generated_gripper, one_dof_gripper, or cross_embodiment_gripper"
            )
        if "kinematic_gripper_clouds_flat" not in observation.get("layout", ()):
            raise RuntimeError(
                "Kinematic gripper-state clouds are missing from the observation layout"
            )
    if (
        robot_mode == "official_panda_gripper"
        and observation.get("tool_cloud_source")
        not in {
            "official_panda_gripper_kinematic_mesh",
            "official_panda_gripper_primitives",
            "official_panda_gripper_meshes",
        }
    ):
        raise RuntimeError(
            "official_panda_gripper requires "
            "observation_params.tool_cloud_source=official_panda_gripper_kinematic_mesh"
        )
    if (
        robot_mode == "generated_gripper"
        and observation.get("tool_cloud_source") != "gripper_cloud_cache_v1"
    ):
        raise RuntimeError(
            "generated_gripper requires "
            "observation_params.tool_cloud_source=gripper_cloud_cache_v1"
        )
    if (
        robot_mode == "one_dof_gripper"
        and observation.get("tool_cloud_source") != "gripper_cloud_cache_v1"
    ):
        raise RuntimeError(
            "one_dof_gripper requires "
            "observation_params.tool_cloud_source=gripper_cloud_cache_v1"
        )
    if (
        robot_mode == "cross_embodiment_gripper"
        and observation.get("tool_cloud_source") != "gripper_cloud_cache_v1"
    ):
        raise RuntimeError(
            "cross_embodiment_gripper requires "
            "observation_params.tool_cloud_source=gripper_cloud_cache_v1"
        )
    if policy.get("encoder_backend") == "oracle_patch":
        if not bool(observation.get("include_oracle_mesh_sdf", False)):
            raise RuntimeError("oracle_patch requires observation_params.include_oracle_mesh_sdf=True")
        layout = list(observation.get("layout", ()))
        if "oracle_mesh_signed_sdf" not in layout or layout.index("oracle_mesh_signed_sdf") != 2:
            raise RuntimeError(
                "oracle_patch requires oracle_mesh_signed_sdf immediately after object/tool clouds"
            )
    if policy.get("encoder_backend") == "oracle_pointmesh_pointnet":
        if not bool(observation.get("include_oracle_mesh_unsigned_distance", False)):
            raise RuntimeError(
                "oracle_pointmesh_pointnet requires "
                "observation_params.include_oracle_mesh_unsigned_distance=True"
            )
        layout = list(observation.get("layout", ()))
        if (
            "oracle_mesh_unsigned_distance" not in layout
            or layout.index("oracle_mesh_unsigned_distance") != 2
        ):
            raise RuntimeError(
                "oracle_pointmesh_pointnet requires oracle_mesh_unsigned_distance "
                "immediately after object/tool clouds"
            )
    if policy.get("class_name") != spec["actor_critic_class"]:
        raise RuntimeError("RL runtime spec policy_params.class_name must match actor_critic_class")

    if spec["actor_critic_class"] == "ActorCriticICP":
        required_policy = {
            "icp_weights_path",
            "icp_point_dim",
            "icp_num_points",
            "freeze_icp",
            "fusion_hidden_dims",
            "actor_hidden_dims",
            "critic_hidden_dims",
            "activation",
            "init_noise_std",
            "noise_std_type",
        }
        if policy.get("icp_weights_path") != spec["encoder_checkpoint"]:
            raise RuntimeError("policy_params.icp_weights_path must match encoder_checkpoint")
    else:
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

    if spec["actor_critic_class"] in {
        "ActorCriticTG",
        "ActorCriticTGOutputGate",
        "ActorCriticTGSM",
        "ActorCriticTGHAMNet",
        "ActorCriticTGUnicorn",
        "ActorCriticTGBimanual",
    }:
        required_policy.update(
            {
                "patch_size",
                "encoder_channel",
                "vit_depth",
                "vit_heads",
                "vit_attention_mode",
            }
        )
        if spec["actor_critic_class"] == "ActorCriticTGOutputGate":
            required_policy.update(
                {
                    "expert_a_checkpoint",
                    "expert_b_checkpoint",
                    "output_gate_freeze_experts",
                    "output_gate_hidden_dims",
                    "output_gate_initial_expert_a_weight",
                    "output_gate_per_action",
                }
            )
        if spec["actor_critic_class"] == "ActorCriticTGSM":
            required_policy.update(
                {
                    "task_embedding_dim",
                    "sm_num_layers",
                    "sm_num_modules",
                    "sm_module_hidden",
                    "sm_gating_hidden",
                    "sm_num_gating_layers",
                    "sm_cond_ob",
                    "sm_add_bn",
                }
            )
        if spec["actor_critic_class"] == "ActorCriticTGHAMNet":
            required_policy.update(
                {
                    "hamnet_num_modules",
                    "hamnet_hidden_dims",
                    "hamnet_router_hidden_dims",
                }
            )
        if spec["actor_critic_class"] == "ActorCriticTGUnicorn":
            required_policy.add("num_patches")
        if (
            spec["actor_critic_class"] == "ActorCriticTG"
            and policy.get("encoder_checkpoint_adapter") == "unicorn_strict"
        ):
            required_policy.update({"encoder_backend", "num_patches"})
        if (
            spec["actor_critic_class"] == "ActorCriticTG"
            and policy.get("encoder_backend") == "tce"
        ):
            # July runtime specs predate the token-source/PCA/bottleneck
            # experiments. Missing fields mean the original full encoder token.
            token_source = str(policy.get("unicorn_token_source", "encoder"))
            if token_source not in {"encoder", "contact_head_hidden"}:
                raise RuntimeError(
                    "RL runtime spec policy_params.unicorn_token_source must be "
                    "encoder or contact_head_hidden"
                )
            encoder_channel = int(policy["encoder_channel"])
            pca_rank = int(policy.get("encoder_token_pca_rank", encoder_channel))
            bottleneck_rank = int(
                policy.get("encoder_token_bottleneck_rank", encoder_channel)
            )
            if not 1 <= pca_rank <= encoder_channel:
                raise RuntimeError(
                    "RL runtime spec encoder_token_pca_rank must be in [1, encoder_channel]"
                )
            if not 1 <= bottleneck_rank <= encoder_channel:
                raise RuntimeError(
                    "RL runtime spec encoder_token_bottleneck_rank must be in "
                    "[1, encoder_channel]"
                )
            if pca_rank < encoder_channel and not policy.get("encoder_token_pca_path"):
                raise RuntimeError(
                    "RL runtime spec reduced encoder-token PCA requires "
                    "encoder_token_pca_path"
                )
            if (
                bottleneck_rank < encoder_channel
                and not policy.get("encoder_token_bottleneck_pca_path")
            ):
                raise RuntimeError(
                    "RL runtime spec reduced encoder-token bottleneck requires "
                    "encoder_token_bottleneck_pca_path"
                )
            if pca_rank < encoder_channel and bottleneck_rank < encoder_channel:
                raise RuntimeError(
                    "RL runtime spec cannot enable fixed PCA and trainable bottleneck together"
                )
        if (
            spec["actor_critic_class"] == "ActorCriticTG"
            and policy.get("encoder_backend") == "oracle_patch"
        ):
            required_policy.update(
                {
                    "encoder_backend",
                    "num_patches",
                    "oracle_contact_eps",
                    "oracle_center_scale_m",
                    "oracle_distance_scale_m",
                    "oracle_patch_relative_scale_m",
                    "oracle_log_distance_resolution_m",
                    "oracle_log_distance_cap_m",
                    "oracle_normalization_clip",
                }
            )
        if (
            spec["actor_critic_class"] == "ActorCriticTG"
            and policy.get("encoder_backend") == "oracle_pointmesh_pointnet"
        ):
            required_policy.update(
                {
                    "encoder_backend",
                    "num_patches",
                    "oracle_pointmesh_coordinate_scale_m",
                    "oracle_pointmesh_distance_scale_m",
                    "oracle_pointmesh_normalization_clip",
                }
            )
        if (
            spec["actor_critic_class"] == "ActorCriticTG"
            and policy.get("encoder_backend") == "oracle_pointcloud_pointnet"
        ):
            required_policy.update(
                {
                    "encoder_backend",
                    "num_patches",
                    "oracle_pointcloud_nearest_frame_batch_size",
                }
            )
            if (
                policy.get("oracle_pointcloud_checkpoint_adapter")
                == "oracle_pointcloud_pointnet_normalized_pretrain_strict"
            ):
                required_policy.add("oracle_pointcloud_input_normalization")
        if (
            spec["actor_critic_class"] == "ActorCriticTG"
            and policy.get("encoder_backend") == "oracle_pointcloud_patch_oracle"
        ):
            required_policy.update(
                {
                    "encoder_backend",
                    "num_patches",
                    "oracle_pointcloud_nearest_frame_batch_size",
                }
            )
        if (
            spec["actor_critic_class"] == "ActorCriticTG"
            and policy.get("encoder_backend") == "patch_distance_pointnet"
        ):
            required_policy.update(
                {
                    "encoder_backend",
                    "num_patches",
                    "patch_distance_point_scale_m",
                    "patch_distance_patch_center_scale_m",
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
    if (
        spec["actor_critic_class"] != "ActorCriticICP"
        and policy.get("encoder_weights_path") != spec["encoder_checkpoint"]
    ):
        raise RuntimeError("policy_params.encoder_weights_path must match encoder_checkpoint")
    if (
        spec["actor_critic_class"] == "ActorCriticPoint2Vec"
        and policy.get("point2vec_ckpt_path") != spec["encoder_checkpoint"]
    ):
        raise RuntimeError("policy_params.point2vec_ckpt_path must match encoder_checkpoint")
    missing_policy = sorted(required_policy.difference(policy))
    if missing_policy:
        raise RuntimeError(f"RL runtime spec policy_params missing fields: {missing_policy}")
    if (
        spec["actor_critic_class"] == "ActorCriticTG"
        and policy.get("encoder_checkpoint_adapter") == "unicorn_strict"
        and policy.get("encoder_backend") != "unicorn"
    ):
        raise RuntimeError(
            "ActorCriticTG with unicorn_strict checkpoint requires "
            "policy_params.encoder_backend='unicorn'"
        )
    if (
        spec["actor_critic_class"] == "ActorCriticTG"
        and policy.get("encoder_checkpoint_adapter")
        == "patch_distance_pointnet_strict"
        and policy.get("encoder_backend") != "patch_distance_pointnet"
    ):
        raise RuntimeError(
            "ActorCriticTG with patch_distance_pointnet_strict checkpoint "
            "requires policy_params.encoder_backend='patch_distance_pointnet'"
        )

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
        "reward_params",
        "domain_randomization_params",
    ):
        _require_mapping(spec, key, path)


def runtime_spec_contract(spec: Mapping[str, Any]) -> SimpleNamespace:
    """Return attr-style config used by legacy Isaac cfg declarations."""

    observation = dict(spec["observation_params"])
    observation.setdefault("object_cloud_source", "preprocessed")
    observation.setdefault(
        "object_cloud_preprocessed_dir",
        "/mnt/project/world_model/tool_generalist/assets/DGN/first_hit_fps_pointclouds/npy",
    )
    observation.setdefault("point_cloud_noise_enabled", True)
    observation.setdefault("include_kinematic_gripper_clouds", False)

    reward = dict(spec["reward_params"])
    reward.setdefault("object_goal_threshold_term_weight", 6.0)
    reward.setdefault("bimanual_arm_proximity_penalty_weight", -20.0)
    reward.setdefault("bimanual_arm_proximity_warning_distance", 0.20)
    reward.setdefault("bimanual_arm_proximity_failure_distance", 0.15)
    reward.setdefault("bimanual_wrist_surface_penalty_weight", -5.0)
    reward.setdefault("bimanual_wrist_surface_warning_height", 0.12)
    reward.setdefault("bimanual_wrist_surface_contact_height", 0.06)
    reward.setdefault("bimanual_tool_proximity_penalty_weight", -10.0)
    reward.setdefault("bimanual_tool_proximity_warning_clearance", 0.02)
    reward.setdefault("bimanual_tool_proximity_contact_clearance", 0.005)
    reward.setdefault("bimanual_tool_proximity_num_points", 128)
    reward["stable_success_dwell_steps"] = max(
        1,
        int(reward.get("stable_success_dwell_steps", 10)),
    )

    env = dict(spec["env_params"])
    env.setdefault("object_solver_position_iteration_count", 16)
    env.setdefault("object_solver_velocity_iteration_count", 1)
    env.setdefault("articulation_solver_position_iteration_count", 8)
    env.setdefault("articulation_solver_velocity_iteration_count", 0)
    env.setdefault("max_depenetration_velocity", 5.0)
    env.setdefault("enable_ccd", False)
    env.setdefault("contact_offset", None)
    env.setdefault("rest_offset", None)
    env.setdefault("visualize_tool_pointcloud", True)
    env.setdefault("generated_parallel_finger_velocity_limit_m_s", 0.05)

    return _to_namespace(
        {
            "action": spec["action_params"],
            "observation": observation,
            "launch": spec["launch_params"],
            "table": spec["table_params"],
            "object_pose_sampling": spec["object_pose_sampling_params"],
            "curriculum": spec.get(
                "curriculum_params",
                {
                    "enabled": False,
                    "start_step": 0,
                    "end_step": 100000,
                    "start_stable_pose_probability": 1.0,
                    "end_stable_pose_probability": 0.0,
                },
            ),
            "asset_assignment": spec["asset_assignment_params"],
            "env": env,
            "reward": reward,
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
