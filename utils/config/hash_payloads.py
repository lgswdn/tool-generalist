"""Small semantic payloads for artifact hashes."""

from __future__ import annotations

from typing import Any

from configs.config_exp import ExpCfg
from utils.io import to_plain_data
from utils.geometry.gripper_cloud_contract import CACHE_SCHEMA_VERSION


CONTACT_ARTIFACT_GENERAL_NAME = "fork_sdf"

_CONTACT_RUNTIME_FIELDS = {
    "enabled",
    "regenerate",
    "skip_existing",
    "shard_count",
    "shard_index",
    "artifact_subdir",
    "visualization",
}
_PRETRAIN_RUNTIME_FIELDS = {
    "enabled",
    "retrain",
    "device",
    "logger",
    "wandb_project",
    "wandb_run_name",
    "wandb_entity",
    "wandb_mode",
}
_RL_RUNTIME_FIELDS = {
    "encoder_checkpoint",
    "init_checkpoint",
    "resume_checkpoint",
    "launch",
}


def experiment_payload(cfg: ExpCfg) -> dict[str, Any]:
    return {
        "paths_yaml": cfg.paths_yaml,
        "general": general_payload(cfg),
        "contact_gen": contact_payload(cfg.contact_gen),
        "pretrain": pretrain_payload(cfg.pretrain),
        "model": model_payload(cfg),
        "rl": rl_payload(cfg.rl),
    }


def contact_general_payload(cfg: ExpCfg) -> dict[str, Any]:
    general = to_plain_data(cfg.general)
    payload = {
        "seed": general.get("seed"),
        "paths_yaml": cfg.paths_yaml,
        "tools_selected_json": general.get("tools_selected_json"),
        "tools_manifest": general.get("tools_manifest"),
        "objects_manifest": (
            general.get("contact_objects_manifest")
            or general.get("objects_manifest")
            or general.get("rl_objects_manifest")
        ),
        "tool_mount": general.get("tool_mount"),
    }
    _add_gripper_cloud_contract(payload, general, cfg)
    return payload


def general_payload(cfg: ExpCfg) -> dict[str, Any]:
    general = to_plain_data(cfg.general)
    payload = {
        "seed": general.get("seed"),
        "num_points": general.get("num_points"),
        "paths_yaml": cfg.paths_yaml,
        "tools_selected_json": general.get("tools_selected_json"),
        "tools_manifest": general.get("tools_manifest"),
        "rl_objects_manifest": (
            general.get("rl_objects_manifest")
            or general.get("objects_manifest")
            or general.get("contact_objects_manifest")
        ),
        "contact_objects_manifest": (
            general.get("contact_objects_manifest")
            or general.get("objects_manifest")
            or general.get("rl_objects_manifest")
        ),
        "randomize_tool_assignment": general.get("randomize_tool_assignment"),
        "randomize_object_assignment": general.get("randomize_object_assignment"),
        "dtype": general.get("dtype"),
        "tool_mount": general.get("tool_mount"),
    }
    _add_gripper_cloud_contract(payload, general, cfg)
    return payload


def pretrain_namespace(cfg: ExpCfg) -> str:
    key = (str(cfg.pretrain.name), str(cfg.model.name))
    generated_gripper_namespaces = {
        ("diff_post_generated_gripper", "generated_gripper_diff_post"): "generated_gripper_diff_post_pretrain",
        ("post_generated_gripper", "generated_gripper_post"): "generated_gripper_post_pretrain",
        ("unicorn_contact_generated_gripper", "unicorn_contact_generated_gripper"): "unicorn_pretrain_generated_gripper",
        ("unicorn_contact_ours_generated_gripper", "unicorn_contact_ours_generated_gripper"): "unicorn_pretrain_ours_generated_gripper",
        (
            "oracle_pointmesh_pointnet_contact_generated_gripper",
            "oracle_pointmesh_pointnet",
        ): "oracle_pointmesh_pointnet_pretrain_generated_gripper",
    }
    return generated_gripper_namespaces.get(key, str(cfg.general.name))


def pretrain_general_payload(cfg: ExpCfg) -> dict[str, Any]:
    general = to_plain_data(cfg.general)
    payload = {
        "seed": general.get("seed"),
        "num_points": general.get("num_points"),
        "paths_yaml": cfg.paths_yaml,
        "tools_selected_json": general.get("tools_selected_json"),
        "tools_manifest": general.get("tools_manifest"),
        "contact_objects_manifest": (
            general.get("contact_objects_manifest")
            or general.get("objects_manifest")
            or general.get("rl_objects_manifest")
        ),
        "randomize_tool_assignment": general.get("randomize_tool_assignment"),
        "randomize_object_assignment": general.get("randomize_object_assignment"),
        "dtype": general.get("dtype"),
        "tool_mount": general.get("tool_mount"),
    }
    _add_gripper_cloud_contract(payload, general, cfg)
    return payload


def _add_gripper_cloud_contract(
    payload: dict[str, Any],
    general: dict[str, Any],
    cfg: ExpCfg,
) -> None:
    paths = " ".join(
        str(general.get(key) or "")
        for key in ("tools_selected_json", "tools_manifest")
    )
    if (
        cfg.rl.env.robot_mode
        in {
            "generated_gripper",
            "one_dof_gripper",
            "cross_embodiment_gripper",
        }
        or "generated_gripper" in paths
    ):
        payload["gripper_cloud_schema"] = CACHE_SCHEMA_VERSION


def pretrain_artifact_payload(cfg: ExpCfg) -> dict[str, Any]:
    return {
        "general": pretrain_general_payload(cfg),
        "contact_gen": contact_payload(cfg.contact_gen),
        "pretrain": pretrain_payload(cfg.pretrain),
        "model": model_payload(cfg),
    }


def contact_payload(contact_cfg: Any) -> dict[str, Any]:
    payload = _drop_keys(to_plain_data(contact_cfg), _CONTACT_RUNTIME_FIELDS)
    payload.pop("name", None)
    if not payload.get("rejection_apply_tangent_gaussian", False):
        payload.pop("rejection_apply_tangent_gaussian", None)
    # Preserve hashes for legacy one-directional contact artifacts. The field
    # is retained when bidirectional rejection is explicitly requested.
    if payload.get("penetration_check_mode") == "tool_into_object":
        payload.pop("penetration_check_mode", None)
    physics = dict(payload.get("physics") or {})
    physics.pop("num_workers", None)
    payload["physics"] = physics
    return payload


def pretrain_payload(pretrain_cfg: Any) -> dict[str, Any]:
    payload = _drop_keys(to_plain_data(pretrain_cfg), _PRETRAIN_RUNTIME_FIELDS)
    payload.pop("name", None)
    batch = dict(payload.get("batch") or {})
    batch.pop("num_workers", None)
    payload["batch"] = batch
    checkpoint_policy = dict(payload.get("checkpoint_policy") or {})
    checkpoint_policy.pop("resume_checkpoint", None)
    payload["checkpoint_policy"] = checkpoint_policy
    return payload


def model_payload(cfg: ExpCfg) -> dict[str, Any]:
    model = to_plain_data(cfg.model)
    model.pop("name", None)
    backend = str(model.get("encoder_backend", "tce")).strip().lower()
    if backend in {"tg"}:
        backend = "tce"
    if backend in {"p2v"}:
        backend = "point2vec"
    if backend in {"corn"}:
        backend = "icp"
    for key, key_backend in (
        ("tce", "tce"),
        ("p2v", "point2vec"),
        ("icp", "icp"),
        ("unicorn", "unicorn"),
        ("patch_distance_pointnet", "patch_distance_pointnet"),
        ("oracle_pointmesh_pointnet", "oracle_pointmesh_pointnet"),
        ("oracle_pointcloud_pointnet", "oracle_pointcloud_pointnet"),
        ("oracle_pointcloud_patch_oracle", "oracle_pointcloud_patch_oracle"),
    ):
        if backend != key_backend:
            model.pop(key, None)
        elif isinstance(model.get(key), dict):
            model[key].pop("name", None)
            if key == "tce":
                kinematic = dict(
                    model[key].get("kinematic_conditioning") or {}
                )
                if not kinematic.get("enabled", False):
                    model[key].pop("kinematic_conditioning", None)
    pretrained = dict(model.get("pretrained_encoder") or {})
    pretrained.pop("name", None)
    pretrained.pop("checkpoint_path", None)
    model["pretrained_encoder"] = pretrained
    return model


def rl_payload(rl_cfg: Any) -> dict[str, Any]:
    payload = _drop_keys(to_plain_data(rl_cfg), _RL_RUNTIME_FIELDS)
    payload.pop("name", None)
    if payload.get("actor_critic_class") != "ActorCriticTGHAMNet":
        payload.pop("hamnet_num_modules", None)
        payload.pop("hamnet_hidden_dims", None)
        payload.pop("hamnet_router_hidden_dims", None)
    env = dict(payload.get("env") or {})
    env.pop("num_envs", None)
    env.pop("visualize_tool_pointcloud", None)
    if env.get("generated_parallel_finger_velocity_limit_m_s") == 0.05:
        env.pop("generated_parallel_finger_velocity_limit_m_s")
    payload["env"] = env
    ppo = dict(payload.get("ppo") or {})
    ppo.pop("save_interval", None)
    payload["ppo"] = ppo
    observation = dict(payload.get("observation") or {})
    if not observation.get("include_kinematic_gripper_clouds", False):
        observation.pop("include_kinematic_gripper_clouds", None)
    if observation.get("point_cloud_noise_enabled", True):
        observation.pop("point_cloud_noise_enabled", None)
    payload["observation"] = observation
    return payload


def _drop_keys(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in keys}
