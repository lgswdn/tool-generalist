"""Payload shaping helpers for config and artifact hashes."""

from __future__ import annotations

from typing import Any

from configs.config_contact_gen import strip_contact_gen_hash_defaults
from configs.config_exp import ExpCfg
from utils.io import to_plain_data


CONTACT_ARTIFACT_GENERAL_NAME = "fork_sdf"
_PRETRAIN_LOGGING_FIELDS = (
    "logger",
    "wandb_project",
    "wandb_run_name",
    "wandb_entity",
    "wandb_mode",
)


def location_stable_general_payload(cfg: ExpCfg) -> dict:
    payload = to_plain_data(cfg.general)
    payload["artifact_root"] = "artifacts"
    return payload


def contact_stable_general_payload(cfg: ExpCfg) -> dict:
    payload = location_stable_general_payload(cfg)
    payload["name"] = CONTACT_ARTIFACT_GENERAL_NAME
    return payload


def location_stable_exp_payload(cfg: ExpCfg) -> dict:
    payload = to_plain_data(cfg)
    payload.pop("pretrain_reuse", None)
    if isinstance(payload.get("general"), dict):
        payload["general"]["artifact_root"] = "artifacts"
    if isinstance(payload.get("pretrain"), dict):
        payload["pretrain"] = strip_pretrain_logging_fields(payload["pretrain"])
    if isinstance(payload.get("contact_gen"), dict):
        payload["contact_gen"]["enabled"] = True
        payload["contact_gen"] = strip_contact_gen_hash_defaults(payload["contact_gen"])
    return payload


def pretrain_stable_payload(cfg: ExpCfg) -> dict:
    return strip_pretrain_logging_fields(to_plain_data(cfg.pretrain))


def contact_hash_payload(contact_cfg: Any) -> dict:
    payload = to_plain_data(contact_cfg)
    payload["enabled"] = True
    return strip_contact_gen_hash_defaults(payload)


def pretrain_model_hash_payload(cfg: ExpCfg) -> dict:
    model = to_plain_data(cfg.model)
    strip_inactive_encoder_backend_defaults(model)
    policy_fusion = dict(model.get("policy_fusion") or {})
    if policy_fusion:
        policy_fusion["query_dim"] = 128
        model["policy_fusion"] = policy_fusion
    return model


def planner_exp_hash_payload(cfg: ExpCfg) -> dict[str, Any]:
    payload = to_plain_data(cfg)
    payload.pop("pretrain_reuse", None)
    if isinstance(payload.get("contact_gen"), dict):
        payload["contact_gen"] = strip_contact_gen_hash_defaults(payload["contact_gen"])
    return payload


def semantic_exp_payload(cfg: ExpCfg) -> dict:
    payload = to_plain_data(cfg)
    payload.pop("pretrain_reuse", None)
    payload.pop("num_gpus", None)
    normalize_artifact_root(payload)
    if "contact_gen" in payload:
        payload["contact_gen"] = strip_contact_runtime_fields(payload["contact_gen"])
    if "pretrain" in payload:
        payload["pretrain"] = strip_pretrain_runtime_fields(payload["pretrain"])
    return payload


def normalize_artifact_root(payload: dict) -> None:
    general = payload.get("general")
    if isinstance(general, dict):
        general["artifact_root"] = "artifacts"


def semantic_contact_general_payload(cfg: ExpCfg) -> dict:
    return contact_stable_general_payload(cfg)


def semantic_contact_gen_payload(contact_cfg: Any) -> dict:
    return strip_contact_runtime_fields(to_plain_data(contact_cfg))


def semantic_pretrain_payload(pretrain_cfg: Any) -> dict:
    return strip_pretrain_runtime_fields(to_plain_data(pretrain_cfg))


def semantic_pretrain_model_payload(cfg: ExpCfg) -> dict:
    return pretrain_model_hash_payload(cfg)


def strip_contact_runtime_fields(payload: dict) -> dict:
    cleaned = dict(payload)
    cleaned["enabled"] = True
    physics = dict(cleaned.get("physics") or {})
    physics.pop("num_workers", None)
    cleaned["physics"] = physics
    return strip_contact_gen_hash_defaults(cleaned)


def strip_pretrain_runtime_fields(payload: dict) -> dict:
    cleaned = dict(payload)
    batch = dict(cleaned.get("batch") or {})
    batch.pop("num_workers", None)
    cleaned["batch"] = batch
    if isinstance(cleaned.get("loss"), dict):
        cleaned["loss"] = dict(cleaned["loss"])
    for key in _PRETRAIN_LOGGING_FIELDS:
        cleaned.pop(key, None)
    strip_inactive_unicorn_pretrain_defaults(cleaned)
    strip_default_sdf_relative_loss(cleaned)
    strip_default_condition_normalization(cleaned)
    return cleaned


def strip_pretrain_logging_fields(payload: dict) -> dict:
    cleaned = dict(payload)
    if isinstance(cleaned.get("loss"), dict):
        cleaned["loss"] = dict(cleaned["loss"])
    for key in _PRETRAIN_LOGGING_FIELDS:
        cleaned.pop(key, None)
    strip_inactive_unicorn_pretrain_defaults(cleaned)
    strip_default_sdf_relative_loss(cleaned)
    strip_default_condition_normalization(cleaned)
    return cleaned


def strip_inactive_encoder_backend_defaults(model: dict) -> None:
    backend = str(model.get("encoder_backend", "tce")).strip().lower()
    if backend in {"tg"}:
        backend = "tce"
    if backend in {"p2v"}:
        backend = "point2vec"
    if backend in {"corn"}:
        backend = "icp"
    if backend != "icp":
        model.pop("icp", None)
    if backend != "unicorn":
        model.pop("unicorn", None)


def strip_inactive_unicorn_pretrain_defaults(pretrain: dict) -> None:
    if pretrain.get("mode") == "unicorn_contact":
        return
    pretrain.pop("mode", None)
    pretrain.pop("device", None)
    pretrain.pop("unicorn", None)
    optimizer = pretrain.get("optimizer")
    if isinstance(optimizer, dict):
        optimizer.pop("sam_rho", None)
    tasks = pretrain.get("tasks")
    if isinstance(tasks, dict):
        tasks.pop("contact", None)


def strip_default_sdf_relative_loss(pretrain: dict) -> None:
    loss = pretrain.get("loss")
    if not isinstance(loss, dict):
        return
    if bool(loss.get("sdf_relative_loss", False)):
        return
    if float(loss.get("sdf_relative_eps", 0.005)) != 0.005:
        return
    loss.pop("sdf_relative_loss", None)
    loss.pop("sdf_relative_eps", None)


def strip_default_condition_normalization(pretrain: dict) -> None:
    if pretrain.get("condition_normalization") is not None:
        return
    pretrain.pop("condition_normalization", None)
    pretrain.pop("condition_norm_sample_files", None)
    pretrain.pop("condition_norm_eps", None)
