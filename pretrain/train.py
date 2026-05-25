"""Canonical pretrain implementation and experiment-stage entrypoint."""

from __future__ import annotations

import hashlib
import os
import argparse
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from configs.config_exp import ExpCfg
from configs.config_contact_gen import strip_contact_gen_hash_defaults
from utils.config.paths import ProjectPaths
from utils.artifacts.resolver import resolve_artifacts
from utils.experiment.runtime import git_metadata, runtime_metadata
from utils.io import hash_json, read_json, to_plain_data, write_json

from pretrain.dataset import make_split
from pretrain.model import ContactDiffusionModel


@dataclass
class PretrainRuntimeConfig:
    data_dir: str
    max_files: int
    val_ratio: float
    augment: bool
    allow_mock_physics: bool
    validation_seed: int
    task: str
    enabled_heads: tuple[str, ...]
    loss_weights: dict[str, float]
    head_mode: str
    patch_agg: str
    head_hidden: tuple[int, ...]
    num_pts: int
    patch_size: int
    encoder_channel: int
    vit_depth: int
    vit_heads: int
    freeze_encoder: bool
    cross_attn_heads: int
    cross_attn_layers: int
    condition_mlp_hidden_dims: tuple[int, ...]
    num_query_A: int
    num_query_B: int
    num_query_C: int
    num_query_D: int
    condition_normalization: bool | None
    condition_norm_sample_files: int
    condition_norm_eps: float
    condition_mean: tuple[float, ...] | None
    condition_std: tuple[float, ...] | None
    pose_dim: int
    movement_cond_dim: int
    denoise_hidden: tuple[int, ...]
    postcontact_hidden: tuple[int, ...]
    num_diffusion_steps: int
    num_precontact_steps: int
    noise_max_trans: float
    noise_max_rot_deg: float
    noise_max_retries: int
    floor_eps: float
    denoise_target_mode: str
    sdf_weight: float
    sdf_backend: str
    sdf_mode: str
    sdf_query: str
    sdf_chunk_size: int
    sdf_fail_without_backend: bool
    sdf_relative_loss: bool
    sdf_relative_eps: float
    denoise_weight: float
    postcontact_weight: float
    denoise_rot_weight: float
    chamfer_weight: float
    quat_norm_beta: float
    batch_size: int
    lr: float
    weight_decay: float
    scheduler: str
    min_lr: float
    epochs: int
    log_interval: int
    num_workers: int
    resume: str | None
    ckpt_dir: str
    checkpoint_schema_version: str
    checkpoint_write_manifest: bool
    dataset_hash_algo: str
    full_config: dict[str, Any]
    full_config_hash: str
    paths_yaml: str
    artifact_dir: str
    wandb: bool
    wandb_project: str
    wandb_run_name: str
    wandb_entity: str | None
    wandb_mode: str
    seed: int


# ============================================================================ #
# Distributed helpers
# ============================================================================ #

def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp() -> tuple[int, int]:
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank


# ============================================================================ #
# Checkpoint helpers
# ============================================================================ #

def _config_dump(cfg: PretrainRuntimeConfig | None) -> dict:
    if cfg is None:
        return {}
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(vars(cfg))


def _hash_dataset_paths(paths: list[str] | tuple[str, ...], algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    for raw_path in sorted(str(p) for p in paths):
        path = Path(raw_path)
        h.update(str(path).encode("utf-8"))
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()


def build_checkpoint_metadata(
    *,
    cfg: PretrainRuntimeConfig | None,
    dataset=None,
    model: torch.nn.Module | None,
    epoch: int,
    best_val: float,
) -> dict:
    cfg_dump = _config_dump(cfg)
    dataset_paths = list(getattr(dataset, "source_paths", []))
    raw_model = model.module if isinstance(model, DDP) else model
    enabled_heads = list(getattr(raw_model, "enabled_heads", cfg_dump.get("enabled_heads", [])))
    loss_weights = dict(getattr(raw_model, "loss_weights", cfg_dump.get("loss_weights", {})))
    model_dims = {
        "num_pts": cfg_dump.get("num_pts"),
        "patch_size": cfg_dump.get("patch_size"),
        "encoder_channel": cfg_dump.get("encoder_channel"),
        "num_patches": getattr(raw_model, "num_patches", None),
        "feature_dim": getattr(getattr(raw_model, "encoder", None), "feature_dim", None),
        "head_mode": getattr(raw_model, "head_mode", cfg_dump.get("head_mode")),
        "cross_attn_layers": cfg_dump.get("cross_attn_layers"),
        "cross_attn_heads": cfg_dump.get("cross_attn_heads"),
        "condition_mlp_hidden_dims": cfg_dump.get("condition_mlp_hidden_dims"),
        "num_query_A": cfg_dump.get("num_query_A"),
        "num_query_B": cfg_dump.get("num_query_B"),
        "num_query_C": cfg_dump.get("num_query_C"),
        "num_query_D": cfg_dump.get("num_query_D"),
        "condition_normalization": cfg_dump.get("condition_normalization"),
        "condition_norm_sample_files": cfg_dump.get("condition_norm_sample_files"),
        "condition_norm_eps": cfg_dump.get("condition_norm_eps"),
        "condition_mean": cfg_dump.get("condition_mean"),
        "condition_std": cfg_dump.get("condition_std"),
    }
    hash_algo = cfg_dump.get("dataset_hash_algo", "sha256")
    dataset_hash = _hash_dataset_paths(dataset_paths, hash_algo) if dataset_paths else ""
    return {
        "schema_version": cfg_dump.get("checkpoint_schema_version", "pretrain_checkpoint_v1"),
        "contact_schema_version": getattr(dataset, "schema_version", "contact_pt_env_v1"),
        "full_config": cfg_dump.get("full_config", {}),
        "full_config_hash": cfg_dump.get("full_config_hash", ""),
        "pretrain_config": cfg_dump,
        "paths": {
            "paths_yaml": cfg_dump.get("paths_yaml", ""),
            "artifact_dir": cfg_dump.get("artifact_dir", ""),
            "checkpoint_dir": cfg_dump.get("ckpt_dir", ""),
        },
        "dataset": {
            "path": cfg_dump.get("data_dir", ""),
            "paths": dataset_paths,
            "hash": dataset_hash,
            "hash_algo": hash_algo,
            "num_files": len(dataset_paths),
            "num_items": len(dataset) if dataset is not None else None,
            "schema_version": getattr(dataset, "schema_version", "contact_pt_env_v1"),
        },
        "dataset_path": cfg_dump.get("data_dir", ""),
        "dataset_paths": dataset_paths,
        "dataset_hash": dataset_hash,
        "dataset_hash_algo": hash_algo,
        "enabled_heads": enabled_heads,
        "loss_weights": loss_weights,
        "best_metric": float(best_val),
        "epoch": int(epoch),
        "model": {
            "dims": model_dims,
            "enabled_heads": enabled_heads,
            "loss_weights": loss_weights,
            "head_hidden": cfg_dump.get("head_hidden"),
            "denoise_hidden": cfg_dump.get("denoise_hidden"),
            "postcontact_hidden": cfg_dump.get("postcontact_hidden"),
        },
        "model_dims": model_dims,
        "git": git_metadata(Path.cwd()),
        "runtime": runtime_metadata(cwd=Path.cwd(), argv=[]),
    }


def save_ckpt(
    path: Path,
    model: torch.nn.Module,
    optimizer,
    epoch: int,
    best_val: float,
    *,
    cfg: PretrainRuntimeConfig | None = None,
    dataset=None,
):
    metadata = build_checkpoint_metadata(
        cfg=cfg,
        dataset=dataset,
        model=model,
        epoch=epoch,
        best_val=best_val,
    )
    payload = {
        "epoch": epoch,
        "best_val": best_val,
        "model": (model.module if isinstance(model, DDP) else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)
    if cfg is None or getattr(cfg, "checkpoint_write_manifest", True):
        manifest_path = path.with_suffix(".manifest.json")
        write_json(manifest_path, metadata)


def load_ckpt(path: str, model: torch.nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = model.module if isinstance(model, DDP) else model
    m.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


# ============================================================================ #
# Collate: skip None-valued optional fields
# ============================================================================ #

# Keys that should be passed through as lists (not stacked into tensors)
# Strings / variable-length fields: pass through as lists, do not torch.stack
_LIST_KEYS = {
    "tool_mesh_path",
    "obj_mesh_path",
    "pt_path",
    "schema_version",
    "object_id",
    "tool_id",
    "target_tool_denoise_mode",
    "object_mesh_vertices",
    "object_mesh_faces",
    "tool_mesh_vertices",
    "tool_mesh_faces",
}


def collate_fn(batch):
    """Stack tensors from batch dicts; pass string/list fields through as lists."""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if vals[0] is None:
            out[key] = None
        elif key in _LIST_KEYS or not isinstance(vals[0], torch.Tensor):
            # strings, lists, variable-length tensors → keep as list
            out[key] = vals
        else:
            out[key] = torch.stack(vals)
    return out


# ============================================================================ #
# Training step
# ============================================================================ #

def train_step(
    model: ContactDiffusionModel,
    batch: dict,
    cfg: PretrainRuntimeConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """One contact_pt_env_v1 CPU/GPU training step."""

    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    return model(
        tool_points_E_k=tensor_batch["tool_points_E_k"],
        object_points_E_k=tensor_batch["object_points_E_k"],
        rel_tool_object_t_k=tensor_batch["rel_tool_object_t_k"],
        cond_tool_post_delta9d=tensor_batch.get("cond_tool_post_delta9d"),
        cond_object_post_delta9d=tensor_batch.get("cond_object_post_delta9d"),
        physics=tensor_batch.get("physics"),
        object_mesh_vertices=tensor_batch.get("object_mesh_vertices"),
        object_mesh_faces=tensor_batch.get("object_mesh_faces"),
        tool_mesh_vertices=tensor_batch.get("tool_mesh_vertices"),
        tool_mesh_faces=tensor_batch.get("tool_mesh_faces"),
        object_rotation_E=tensor_batch.get("object_rotation_E"),
        object_bbox_center_E=tensor_batch.get("object_bbox_center_E"),
        tool_rotation_E_k=tensor_batch.get("tool_rotation_E_k"),
        tool_translation_E_k=tensor_batch.get("tool_translation_E_k"),
        target_tool_denoise_pose9d_k=tensor_batch["target_tool_denoise_pose9d_k"],
        target_object_post_delta9d=tensor_batch.get("target_object_post_delta9d"),
    )


def _format_metric_subset(metrics: dict[str, float], keys: tuple[str, ...]) -> str:
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={float(metrics[key]):.6g}")
    return " ".join(parts)


def run_pretrain(
    exp_cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    runtime_cfg = build_runtime_config(exp_cfg, paths, artifact_dir)
    if _should_spawn_pretrain(exp_cfg.num_gpus):
        return _spawn_distributed_pretrain(runtime_cfg, exp_cfg.num_gpus)
    return _run_training_loop(runtime_cfg)


def _should_spawn_pretrain(num_gpus: int) -> bool:
    return int(num_gpus) > 1 and "LOCAL_RANK" not in os.environ


def _spawn_distributed_pretrain(
    runtime_cfg: PretrainRuntimeConfig,
    num_gpus: int,
) -> dict[str, Any]:
    nproc = _active_pretrain_gpu_count(num_gpus)
    runtime_path = Path(runtime_cfg.artifact_dir) / "pretrain_runtime_config.json"
    write_json(runtime_path, asdict(runtime_cfg))
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node",
        str(nproc),
        "-m",
        "pretrain.train",
        "--runtime-config",
        str(runtime_path),
    ]
    print(
        "[pretrain] launching distributed pretrain "
        f"num_gpus={num_gpus} nproc_per_node={nproc} runtime_config={runtime_path}",
        flush=True,
    )
    completed = subprocess.run(cmd, check=True)
    best_checkpoint_path = Path(runtime_cfg.ckpt_dir) / "best.pt"
    return {
        "status": "complete",
        "distributed": True,
        "num_gpus": int(num_gpus),
        "nproc_per_node": nproc,
        "returncode": completed.returncode,
        "checkpoint_dir": str(runtime_cfg.ckpt_dir),
        "best_checkpoint_path": str(best_checkpoint_path),
        "runtime_config_path": str(runtime_path),
    }


def _active_pretrain_gpu_count(num_gpus: int) -> int:
    requested = int(num_gpus)
    if requested <= 0:
        return 1
    visible = _visible_gpu_count()
    if visible is not None and visible > 0:
        requested = min(requested, visible)
    return max(1, requested)


def build_runtime_config(
    exp_cfg: ExpCfg,
    paths: ProjectPaths,
    artifact_dir: str | Path,
) -> PretrainRuntimeConfig:
    pretrain_cfg = exp_cfg.pretrain
    model_cfg = exp_cfg.model
    general_cfg = exp_cfg.general
    enabled_heads = tuple(pretrain_cfg.enabled_heads)
    if "sdf" in enabled_heads:
        if pretrain_cfg.sdf_target.mode != "signed":
            raise ValueError("PretrainCfg.sdf_target.mode must be 'signed' for SDF supervision")
        if pretrain_cfg.sdf_target.query != "surface_points":
            raise ValueError("PretrainCfg.sdf_target.query must be 'surface_points' for SDF supervision")
    has_pose_head = bool({"diff", "postcontact"}.intersection(enabled_heads))
    if pretrain_cfg.dataset_manifest:
        data_dir = _resolve_config_path(pretrain_cfg.dataset_manifest, paths)
    else:
        data_dir = _contact_artifact_dir(exp_cfg)
    noise_trans_high = float(pretrain_cfg.translation_noise_range[1])
    noise_rot_high = float(pretrain_cfg.rotation_noise_range_deg[1])
    full_config = to_plain_data(exp_cfg)
    full_config_for_hash = _strip_pretrain_logging_from_exp_payload(full_config)
    resume_checkpoint = pretrain_cfg.checkpoint_policy.resume_checkpoint
    if not resume_checkpoint:
        existing_best = Path(artifact_dir) / pretrain_cfg.checkpoint_policy.best_filename
        if existing_best.exists():
            resume_checkpoint = str(existing_best)

    return PretrainRuntimeConfig(
        data_dir=str(data_dir),
        max_files=pretrain_cfg.max_files,
        val_ratio=pretrain_cfg.val_ratio,
        augment=pretrain_cfg.augment,
        allow_mock_physics=pretrain_cfg.allow_mock_physics,
        validation_seed=pretrain_cfg.validation_noising_seed,
        task="sdf-diff" if has_pose_head else "sdf",
        enabled_heads=enabled_heads,
        loss_weights={
            "sdf": float(pretrain_cfg.loss.w_sdf),
            "diff": float(pretrain_cfg.loss.w_diff),
            "postcontact": float(pretrain_cfg.loss.w_post),
        },
        head_mode=pretrain_cfg.sdf_head_mode,
        patch_agg=pretrain_cfg.decoder_pooling,
        head_hidden=tuple(pretrain_cfg.sdf_head_hidden_dims),
        num_pts=model_cfg.num_points,
        patch_size=model_cfg.patch_size,
        encoder_channel=model_cfg.encoder_channel,
        vit_depth=model_cfg.vit_depth,
        vit_heads=model_cfg.vit_heads,
        freeze_encoder=not model_cfg.encoder.trainable,
        cross_attn_heads=pretrain_cfg.cross_attn_heads,
        cross_attn_layers=pretrain_cfg.cross_attn_layers,
        condition_mlp_hidden_dims=tuple(pretrain_cfg.condition_mlp_hidden_dims),
        num_query_A=pretrain_cfg.num_query_A,
        num_query_B=pretrain_cfg.num_query_B,
        num_query_C=pretrain_cfg.num_query_C,
        num_query_D=pretrain_cfg.num_query_D,
        condition_normalization=pretrain_cfg.condition_normalization,
        condition_norm_sample_files=int(pretrain_cfg.condition_norm_sample_files),
        condition_norm_eps=float(pretrain_cfg.condition_norm_eps),
        condition_mean=None,
        condition_std=None,
        pose_dim=pretrain_cfg.pose_dim,
        movement_cond_dim=pretrain_cfg.movement_cond_dim,
        denoise_hidden=tuple(pretrain_cfg.denoise_head_hidden_dims),
        postcontact_hidden=tuple(pretrain_cfg.postcontact_head_hidden_dims),
        num_diffusion_steps=pretrain_cfg.num_precontact_steps,
        num_precontact_steps=pretrain_cfg.num_precontact_steps,
        noise_max_trans=noise_trans_high,
        noise_max_rot_deg=noise_rot_high,
        noise_max_retries=pretrain_cfg.legal_pose_max_tries,
        floor_eps=pretrain_cfg.floor_eps,
        denoise_target_mode=pretrain_cfg.denoise_target_mode,
        sdf_weight=float(pretrain_cfg.loss.w_sdf),
        sdf_backend=pretrain_cfg.sdf_target.backend,
        sdf_mode=pretrain_cfg.sdf_target.mode,
        sdf_query=pretrain_cfg.sdf_target.query,
        sdf_chunk_size=int(pretrain_cfg.sdf_target.chunk_size),
        sdf_fail_without_backend=bool(pretrain_cfg.sdf_target.fail_without_backend),
        sdf_relative_loss=bool(pretrain_cfg.loss.sdf_relative_loss),
        sdf_relative_eps=float(pretrain_cfg.loss.sdf_relative_eps),
        denoise_weight=float(pretrain_cfg.loss.w_diff),
        postcontact_weight=float(pretrain_cfg.loss.w_post),
        denoise_rot_weight=pretrain_cfg.loss.denoise_rot_weight,
        chamfer_weight=pretrain_cfg.loss.chamfer_weight,
        quat_norm_beta=pretrain_cfg.loss.quat_norm_beta,
        batch_size=pretrain_cfg.batch.batch_size,
        lr=pretrain_cfg.optimizer.learning_rate,
        weight_decay=pretrain_cfg.optimizer.weight_decay,
        scheduler=pretrain_cfg.optimizer.scheduler,
        min_lr=pretrain_cfg.optimizer.min_learning_rate,
        epochs=pretrain_cfg.epochs,
        log_interval=pretrain_cfg.log_interval,
        num_workers=_active_pretrain_worker_count(exp_cfg.num_gpus, pretrain_cfg.batch.num_workers),
        resume=resume_checkpoint,
        ckpt_dir=str(Path(artifact_dir)),
        checkpoint_schema_version=pretrain_cfg.checkpoint_policy.schema_version,
        checkpoint_write_manifest=pretrain_cfg.checkpoint_policy.write_manifest,
        dataset_hash_algo=pretrain_cfg.checkpoint_policy.dataset_hash_algo,
        full_config=full_config,
        full_config_hash=hash_json(full_config_for_hash, pretrain_cfg.checkpoint_policy.dataset_hash_algo),
        paths_yaml=str(paths.source_yaml),
        artifact_dir=str(Path(artifact_dir)),
        wandb=(pretrain_cfg.logger == "wandb") or bool(general_cfg.wandb.enabled),
        wandb_project=pretrain_cfg.wandb_project or general_cfg.wandb.project or general_cfg.name,
        wandb_run_name=pretrain_cfg.wandb_run_name or pretrain_cfg.name,
        wandb_entity=pretrain_cfg.wandb_entity or general_cfg.wandb.entity,
        wandb_mode=pretrain_cfg.wandb_mode or general_cfg.wandb.mode,
        seed=general_cfg.seed,
    )


def _strip_pretrain_logging_from_exp_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    cleaned.pop("pretrain_reuse", None)
    contact_gen = dict(cleaned.get("contact_gen") or {})
    if contact_gen:
        cleaned["contact_gen"] = strip_contact_gen_hash_defaults(contact_gen)
    pretrain = dict(cleaned.get("pretrain") or {})
    if isinstance(pretrain.get("loss"), dict):
        pretrain["loss"] = dict(pretrain["loss"])
    for key in ("logger", "wandb_project", "wandb_run_name", "wandb_entity", "wandb_mode"):
        pretrain.pop(key, None)
    _strip_default_sdf_relative_loss(pretrain)
    _strip_default_condition_normalization(pretrain)
    if pretrain:
        cleaned["pretrain"] = pretrain
    model = dict(cleaned.get("model") or {})
    policy_fusion = dict(model.get("policy_fusion") or {})
    if policy_fusion:
        policy_fusion["query_dim"] = 128
        model["policy_fusion"] = policy_fusion
    if model:
        cleaned["model"] = model
    return cleaned


def _strip_default_sdf_relative_loss(pretrain: dict[str, Any]) -> None:
    loss = pretrain.get("loss")
    if not isinstance(loss, dict):
        return
    if bool(loss.get("sdf_relative_loss", False)):
        return
    if float(loss.get("sdf_relative_eps", 0.005)) != 0.005:
        return
    loss.pop("sdf_relative_loss", None)
    loss.pop("sdf_relative_eps", None)


def _strip_default_condition_normalization(pretrain: dict[str, Any]) -> None:
    if pretrain.get("condition_normalization") is not None:
        return
    pretrain.pop("condition_normalization", None)
    pretrain.pop("condition_norm_sample_files", None)
    pretrain.pop("condition_norm_eps", None)


def _contact_artifact_dir(exp_cfg: ExpCfg) -> Path:
    contact_ref_dir: Path | None = None
    for ref in resolve_artifacts(exp_cfg).stages:
        if ref.stage == "contact_gen":
            contact_ref_dir = ref.directory
            break
    if contact_ref_dir is None:
        raise ValueError("PretrainCfg.dataset_manifest is required when no contact stage is configured")
    if contact_ref_dir.exists():
        return contact_ref_dir
    inferred = _latest_existing_contact_artifact(contact_ref_dir.parent)
    if inferred is not None:
        if is_main():
            print(
                "[pretrain] inferred existing contact dataset "
                f"path={inferred} expected_missing={contact_ref_dir}",
                flush=True,
            )
        return inferred
    raise FileNotFoundError(
        "No contact dataset is available for pretrain. "
        f"Expected {contact_ref_dir}, and found no existing sibling artifacts under {contact_ref_dir.parent}. "
        "Either enable contact generation or set PretrainCfg.dataset_manifest."
    )


def _latest_existing_contact_artifact(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    candidates = [
        path
        for path in parent.iterdir()
        if path.is_dir() and _looks_like_contact_artifact(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _looks_like_contact_artifact(path: Path) -> bool:
    manifest = path / "manifest.json"
    if manifest.exists():
        try:
            payload = read_json(manifest)
        except Exception:
            payload = {}
        if payload.get("artifact_type") == "contact" and payload.get("status") == "complete":
            return True
    return any(path.rglob("*.pt.manifest.json"))


def _active_pretrain_worker_count(num_gpus: int, configured_workers: int) -> int:
    requested = int(num_gpus)
    if requested <= 0:
        return int(configured_workers)
    visible = _visible_gpu_count()
    if visible is not None:
        requested = min(requested, visible)
    return max(1, requested)


def _visible_gpu_count() -> int | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    stripped = value.strip()
    if stripped in {"", "-1"}:
        return 0
    return len([item for item in stripped.split(",") if item.strip()])


def _estimate_condition_normalization_stats(
    dataset,
    *,
    enabled_heads: tuple[str, ...],
    movement_cond_dim: int,
    sample_files: int,
    eps: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if "postcontact" not in enabled_heads:
        return (
            tuple(0.0 for _ in range(int(movement_cond_dim))),
            tuple(1.0 for _ in range(int(movement_cond_dim))),
        )
    source_paths = list(getattr(dataset, "source_paths", []))
    if not source_paths:
        raise RuntimeError("condition normalization requires dataset.source_paths")
    selected_paths = source_paths[: max(1, int(sample_files))]
    rows: list[torch.Tensor] = []
    pt_cache = getattr(dataset, "_pt_cache", {})
    for path in selected_paths:
        data = pt_cache[path]
        n = int(data["num_contacts"])
        if n <= 0:
            continue
        valid = torch.ones(n, dtype=torch.bool)
        movement_valid = data.get("movement_delta_valid")
        if movement_valid is not None:
            valid = torch.as_tensor(movement_valid, dtype=torch.bool)
        if not bool(valid.any()):
            continue
        tool_delta = torch.as_tensor(data["post_tool_delta_pose9d_E"], dtype=torch.float32)[valid]
        object_delta = torch.as_tensor(data["post_object_delta_pose9d_E"], dtype=torch.float32)[valid]
        physics = torch.stack(
            [
                torch.as_tensor(data["object_mass"], dtype=torch.float32)[valid],
                torch.as_tensor(data["tool_mass"], dtype=torch.float32)[valid],
                torch.as_tensor(data["object_friction"], dtype=torch.float32)[valid],
                torch.as_tensor(data["tool_friction"], dtype=torch.float32)[valid],
                torch.as_tensor(data["ground_friction"], dtype=torch.float32)[valid],
            ],
            dim=-1,
        )
        if "postcontact" in enabled_heads:
            rows.append(
                _pad_or_truncate_condition(
                    torch.cat((tool_delta, torch.zeros_like(object_delta), physics), dim=-1),
                    movement_cond_dim,
                )
            )
    if not rows:
        return (
            tuple(0.0 for _ in range(int(movement_cond_dim))),
            tuple(1.0 for _ in range(int(movement_cond_dim))),
        )
    cond = torch.cat(rows, dim=0)
    mean = cond.mean(dim=0)
    std = cond.std(dim=0, unbiased=False).clamp_min(float(eps))
    return tuple(float(v) for v in mean.tolist()), tuple(float(v) for v in std.tolist())


def _pad_or_truncate_condition(cond: torch.Tensor, movement_cond_dim: int) -> torch.Tensor:
    if cond.shape[-1] < int(movement_cond_dim):
        return torch.nn.functional.pad(cond, (0, int(movement_cond_dim) - cond.shape[-1]))
    if cond.shape[-1] > int(movement_cond_dim):
        return cond[..., : int(movement_cond_dim)]
    return cond


def _resolve_config_path(raw_path: str, paths: ProjectPaths) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (paths.source_yaml.parent / path).resolve()


def _runtime_config_from_json(path: str | Path) -> PretrainRuntimeConfig:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Pretrain runtime config must be a JSON object: {path}")
    tuple_keys = {
        "enabled_heads",
        "head_hidden",
        "condition_mlp_hidden_dims",
        "denoise_hidden",
        "postcontact_hidden",
        "condition_mean",
        "condition_std",
    }
    for key in tuple_keys:
        if key in payload and payload[key] is not None:
            payload[key] = tuple(payload[key])
    return PretrainRuntimeConfig(**payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Internal pretrain distributed worker entrypoint."
    )
    parser.add_argument("--runtime-config", required=True)
    args = parser.parse_args(argv)
    _run_training_loop(_runtime_config_from_json(args.runtime_config))
    return 0


def _run_training_loop(cfg: PretrainRuntimeConfig) -> dict[str, Any]:
    require_movement = "postcontact" in cfg.enabled_heads

    # ── Setup ────────────────────────────────────────────────────────────
    rank, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed + rank)

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds, val_ds = make_split(
        data_dir=cfg.data_dir,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        augment=cfg.augment,
        max_files=cfg.max_files,
        require_movement=require_movement,
        num_points=cfg.num_pts,
        num_precontact_steps=cfg.num_precontact_steps,
        allow_mock_physics=cfg.allow_mock_physics,
        noise_max_trans=cfg.noise_max_trans,
        noise_max_rot_deg=cfg.noise_max_rot_deg,
        noise_max_retries=cfg.noise_max_retries,
        floor_eps=cfg.floor_eps,
        validation_seed=cfg.validation_seed,
        denoise_target_mode=cfg.denoise_target_mode,
    )
    if cfg.condition_normalization:
        condition_mean, condition_std = _estimate_condition_normalization_stats(
            train_ds,
            enabled_heads=cfg.enabled_heads,
            movement_cond_dim=cfg.movement_cond_dim,
            sample_files=cfg.condition_norm_sample_files,
            eps=cfg.condition_norm_eps,
        )
        cfg.condition_mean = condition_mean
        cfg.condition_std = condition_std

    if is_main():
        print(f"Train: {len(train_ds)} configs, Val: {len(val_ds)} configs")
        print(
            f"Task: {cfg.task}, heads={list(cfg.enabled_heads)}, head_mode={cfg.head_mode}, "
            f"diffusion_steps={cfg.num_diffusion_steps}, batch_size={cfg.batch_size}, "
            f"num_workers={cfg.num_workers}, lr={cfg.lr:.6g}, weight_decay={cfg.weight_decay:.6g}, "
            f"scheduler={cfg.scheduler}, min_lr={cfg.min_lr:.6g}"
        )
        if "sdf" in cfg.enabled_heads:
            print(
                f"SDF target: backend={cfg.sdf_backend}, mode={cfg.sdf_mode}, "
                f"query={cfg.sdf_query}, chunk_size={cfg.sdf_chunk_size}"
            )
        if "diff" in cfg.enabled_heads:
            print("Diffusion conditioning: noised pose/timestep only; post deltas and physics are ignored")
        if cfg.condition_normalization:
            print(
                "[pretrain] condition normalization enabled "
                f"sample_files={cfg.condition_norm_sample_files} eps={cfg.condition_norm_eps:g}",
                flush=True,
            )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_sampler = DistributedSampler(train_ds) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = ContactDiffusionModel(
        head_mode=cfg.head_mode,
        patch_agg=cfg.patch_agg,
        head_hidden=cfg.head_hidden,
        num_pts=cfg.num_pts,
        patch_size=cfg.patch_size,
        encoder_channel=cfg.encoder_channel,
        vit_depth=cfg.vit_depth,
        vit_heads=cfg.vit_heads,
        freeze_encoder=cfg.freeze_encoder,
        cross_attn_heads=cfg.cross_attn_heads,
        cross_attn_layers=cfg.cross_attn_layers,
        condition_mlp_hidden_dims=cfg.condition_mlp_hidden_dims,
        num_query_A=cfg.num_query_A,
        num_query_B=cfg.num_query_B,
        num_query_C=cfg.num_query_C,
        num_query_D=cfg.num_query_D,
        condition_mean=cfg.condition_mean,
        condition_std=cfg.condition_std,
        condition_norm_eps=cfg.condition_norm_eps,
        pose_dim=cfg.pose_dim,
        movement_cond_dim=cfg.movement_cond_dim,
        denoise_hidden=cfg.denoise_hidden,
        postcontact_hidden=cfg.postcontact_hidden,
        sdf_weight=cfg.sdf_weight,
        denoise_weight=cfg.denoise_weight,
        postcontact_weight=cfg.postcontact_weight,
        loss_weights=cfg.loss_weights,
        denoise_rot_weight=cfg.denoise_rot_weight,
        chamfer_weight=cfg.chamfer_weight,
        quat_norm_beta=cfg.quat_norm_beta,
        num_diffusion_steps=cfg.num_diffusion_steps,
        task=cfg.task,
        enabled_heads=cfg.enabled_heads,
        sdf_backend=cfg.sdf_backend,
        sdf_chunk_size=cfg.sdf_chunk_size,
        sdf_relative_loss=cfg.sdf_relative_loss,
        sdf_relative_eps=cfg.sdf_relative_eps,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = _build_lr_scheduler(optimizer, cfg)

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch = 0
    best_val = float("inf")
    if cfg.resume:
        start_epoch, best_val = load_ckpt(cfg.resume, model, optimizer)
        if is_main():
            print(f"Resumed from {cfg.resume} at epoch {start_epoch}, best_val={best_val:.6f}")

    # ── Wandb ────────────────────────────────────────────────────────────
    if cfg.wandb and HAS_WANDB and is_main():
        run_name = cfg.wandb_run_name or f"{cfg.task}_{cfg.head_mode}_T{cfg.num_diffusion_steps}"
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            mode=cfg.wandb_mode,
            config=vars(cfg),
        )
    elif cfg.wandb and not HAS_WANDB and is_main():
        print("[pretrain] wandb requested but wandb is not installed; continuing without wandb", flush=True)

    # ── Checkpoint directory ─────────────────────────────────────────────
    ckpt_dir = Path(cfg.ckpt_dir)
    if is_main():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_dl):
            loss, metrics = train_step(model, batch, cfg, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            n_batches += 1

            if is_main() and (batch_idx + 1) % cfg.log_interval == 0:
                avg = {k: v / n_batches for k, v in epoch_metrics.items()}
                tracked = _format_metric_subset(
                    avg,
                    (
                        "total_loss",
                        "sdf_loss",
                        "tool_sdf_loss",
                        "obj_sdf_loss",
                        "denoise_loss",
                        "denoise_translation_loss",
                        "denoise_pose_trans_loss",
                        "postcontact_loss",
                        "postcontact_gt_translation_abs_mean",
                        "postcontact_gt_rotation_abs_deg_mean",
                        "postcontact_pred_translation_abs_mean",
                        "postcontact_pred_rotation_abs_deg_mean",
                        "postcontact_pose_trans_loss",
                        "postcontact_pose_rot_geodesic_loss",
                        "postcontact_pose_chamfer_loss",
                    ),
                )
                print(
                    f"  [{epoch+1}/{cfg.epochs}] batch {batch_idx+1}/{len(train_dl)} "
                    f"loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.6g} "
                    f"{tracked}"
                )

        # ── Epoch summary ────────────────────────────────────────────────
        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}
        avg_train["epoch_loss"] = epoch_loss / max(n_batches, 1)
        avg_train["epoch_time"] = time.time() - t0
        avg_train["lr"] = optimizer.param_groups[0]["lr"]

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_metrics = {}
        n_val = 0

        with torch.no_grad():
            for batch in val_dl:
                loss, metrics = train_step(model, batch, cfg, device)
                val_loss += loss.item()
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v
                n_val += 1

        avg_val = {f"val_{k}": v / max(n_val, 1) for k, v in val_metrics.items()}
        avg_val["val_loss"] = val_loss / max(n_val, 1)

        if is_main():
            train_detail = _format_metric_subset(
                avg_train,
                (
                    "sdf_loss",
                    "tool_sdf_loss",
                    "obj_sdf_loss",
                    "denoise_loss",
                    "denoise_translation_loss",
                    "denoise_pose_trans_loss",
                    "postcontact_loss",
                    "postcontact_gt_translation_abs_mean",
                    "postcontact_gt_rotation_abs_deg_mean",
                    "postcontact_pred_translation_abs_mean",
                    "postcontact_pred_rotation_abs_deg_mean",
                    "postcontact_pose_trans_loss",
                    "postcontact_pose_rot_geodesic_loss",
                    "postcontact_pose_chamfer_loss",
                ),
            )
            val_detail = _format_metric_subset(
                avg_val,
                (
                    "val_sdf_loss",
                    "val_tool_sdf_loss",
                    "val_obj_sdf_loss",
                    "val_denoise_loss",
                    "val_denoise_translation_loss",
                    "val_denoise_pose_trans_loss",
                    "val_postcontact_loss",
                    "val_postcontact_gt_translation_abs_mean",
                    "val_postcontact_gt_rotation_abs_deg_mean",
                    "val_postcontact_pred_translation_abs_mean",
                    "val_postcontact_pred_rotation_abs_deg_mean",
                    "val_postcontact_pose_trans_loss",
                    "val_postcontact_pose_rot_geodesic_loss",
                    "val_postcontact_pose_chamfer_loss",
                ),
            )
            print(
                f"Epoch {epoch+1}/{cfg.epochs} — "
                f"train_loss={avg_train['epoch_loss']:.6f} "
                f"val_loss={avg_val['val_loss']:.6f} "
                f"time={avg_train['epoch_time']:.1f}s "
                f"{train_detail} {val_detail}"
            )

            # Log to wandb
            if cfg.wandb and HAS_WANDB:
                log_dict = {**avg_train, **avg_val, "epoch": epoch + 1}
                wandb.log(log_dict)

            if avg_val["val_loss"] < best_val:
                best_val = avg_val["val_loss"]
                save_ckpt(
                    ckpt_dir / "best.pt",
                    model,
                    optimizer,
                    epoch + 1,
                    best_val,
                    cfg=cfg,
                    dataset=train_ds,
                )
                print(f"  → New best val_loss: {best_val:.6f}")
        if scheduler is not None:
            scheduler.step()

    # ── Cleanup ──────────────────────────────────────────────────────────
    if dist.is_initialized():
        dist.destroy_process_group()
    if is_main():
        print("Training complete.")
        if cfg.wandb and HAS_WANDB:
            wandb.finish()
    best_checkpoint_path = ckpt_dir / "best.pt"
    return {
        "status": "complete",
        "best_val": best_val,
        "checkpoint_dir": str(ckpt_dir),
        "best_checkpoint_path": str(best_checkpoint_path),
    }


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: PretrainRuntimeConfig,
):
    name = str(cfg.scheduler).lower()
    if name in {"", "none", "fixed", "constant"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(cfg.epochs)),
            eta_min=float(cfg.min_lr),
        )
    raise ValueError(f"Unsupported pretrain optimizer scheduler {cfg.scheduler!r}")


if __name__ == "__main__":
    raise SystemExit(main())
