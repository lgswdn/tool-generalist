#!/usr/bin/env python3
"""Visualize patchwise SDF ground truth and prediction for a pretrain config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment config exposing EXP_CFG.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to pretrain artifact best.pt.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--index", type=int, default=0, help="Dataset item index within the selected split.")
    parser.add_argument("--timestep", type=int, default=0, help="Precontact timestep to visualize.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    from utils.artifacts.resolver import resolve_artifacts
    from utils.config.loader import load_exp_cfg
    from utils.config.paths import load_project_paths
    from utils.experiment.effective_paths import apply_experiment_path_overrides
    from pretrain.dataset import make_split
    from pretrain.model import ContactDiffusionModel
    from pretrain.train import build_runtime_config, collate_fn, load_ckpt

    cfg = load_exp_cfg(args.config)
    paths = apply_experiment_path_overrides(cfg, load_project_paths(cfg.paths_yaml))
    artifacts = resolve_artifacts(cfg)
    pretrain_ref = _stage_ref(artifacts, "pretrain")
    runtime = build_runtime_config(cfg, paths, pretrain_ref.directory)
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else Path(runtime.ckpt_dir) / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_ds, val_ds = make_split(
        data_dir=runtime.data_dir,
        val_ratio=runtime.val_ratio,
        seed=runtime.seed,
        augment=False,
        max_files=runtime.max_files,
        require_movement=(runtime.task == "sdf-diff"),
        num_points=runtime.num_pts,
        num_precontact_steps=runtime.num_precontact_steps,
        allow_mock_physics=runtime.allow_mock_physics,
        noise_max_trans=runtime.noise_max_trans,
        noise_max_rot_deg=runtime.noise_max_rot_deg,
        noise_max_retries=runtime.noise_max_retries,
        floor_eps=runtime.floor_eps,
        validation_seed=runtime.validation_seed,
        denoise_target_mode=runtime.denoise_target_mode,
    )
    dataset = train_ds if args.split == "train" else val_ds
    if len(dataset) == 0:
        raise RuntimeError(f"{args.split} split is empty")
    item = dataset[int(args.index) % len(dataset)]
    batch = collate_fn([item])
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    model = ContactDiffusionModel(
        head_mode=runtime.head_mode,
        patch_agg=runtime.patch_agg,
        head_hidden=runtime.head_hidden,
        num_pts=runtime.num_pts,
        patch_size=runtime.patch_size,
        encoder_channel=runtime.encoder_channel,
        vit_depth=runtime.vit_depth,
        vit_heads=runtime.vit_heads,
        freeze_encoder=runtime.freeze_encoder,
        cross_attn_heads=runtime.cross_attn_heads,
        cross_attn_layers=runtime.cross_attn_layers,
        condition_mlp_hidden_dims=runtime.condition_mlp_hidden_dims,
        num_query_A=runtime.num_query_A,
        num_query_B=runtime.num_query_B,
        num_query_C=runtime.num_query_C,
        num_query_D=runtime.num_query_D,
        pose_dim=runtime.pose_dim,
        movement_cond_dim=runtime.movement_cond_dim,
        denoise_hidden=runtime.denoise_hidden,
        postcontact_hidden=runtime.postcontact_hidden,
        sdf_weight=runtime.sdf_weight,
        denoise_weight=runtime.denoise_weight,
        postcontact_weight=runtime.postcontact_weight,
        loss_weights=runtime.loss_weights,
        denoise_rot_weight=runtime.denoise_rot_weight,
        chamfer_weight=runtime.chamfer_weight,
        quat_norm_beta=runtime.quat_norm_beta,
        num_diffusion_steps=runtime.num_diffusion_steps,
        task=runtime.task,
        enabled_heads=runtime.enabled_heads,
        sdf_backend=runtime.sdf_backend,
        sdf_chunk_size=runtime.sdf_chunk_size,
    ).to(device)
    load_ckpt(str(checkpoint), model)
    model.eval()

    with torch.no_grad():
        viz = _compute_patch_sdf_viz(model, tensor_batch, timestep=int(args.timestep))

    output = Path(args.output).expanduser() if args.output else _default_output_path(checkpoint, item, args.timestep)
    output.parent.mkdir(parents=True, exist_ok=True)
    _plot_patch_sdf(viz, output)
    print(f"[visualize_sdf_patch] wrote {output}", flush=True)
    return 0


def _compute_patch_sdf_viz(model, batch: dict[str, Any], *, timestep: int) -> dict[str, Any]:
    import torch
    from pretrain.model import _aggregate_sdf

    tool_points_E_k = batch["tool_points_E_k"]
    object_points_E_k = batch["object_points_E_k"]
    rel_tool_object_t_k = batch["rel_tool_object_t_k"]
    B, T, N, _ = tool_points_E_k.shape
    if B != 1:
        raise RuntimeError("visualize_sdf_patch expects a single collated sample")
    t = max(0, min(int(timestep), T - 1))

    tool_flat = tool_points_E_k.reshape(B * T, N, 3)
    obj_flat = object_points_E_k.reshape(B * T, N, 3)
    rel_flat = rel_tool_object_t_k.reshape(B * T, 3)
    encoder_result = model.encoder.encode(tool_flat, obj_flat)
    zero_cond = torch.zeros(B * T, model.movement_cond_dim, device=tool_flat.device, dtype=tool_flat.dtype)
    fused_sdf = model.pose_cross_attn(encoder_result.fused_tokens, rel_flat, zero_cond)
    P = model.num_patches
    tool_tokens = fused_sdf[:, :P, :]
    obj_tokens = fused_sdf[:, P:, :]
    tool_gt_full, obj_gt_full = model._signed_mesh_sdf_labels(
        tool_points_E_k=tool_points_E_k,
        object_points_E_k=object_points_E_k,
        object_mesh_vertices=batch.get("object_mesh_vertices"),
        object_mesh_faces=batch.get("object_mesh_faces"),
        tool_mesh_vertices=batch.get("tool_mesh_vertices"),
        tool_mesh_faces=batch.get("tool_mesh_faces"),
        object_rotation_E=batch.get("object_rotation_E"),
        object_bbox_center_E=batch.get("object_bbox_center_E"),
        tool_rotation_E_k=batch.get("tool_rotation_E_k"),
        tool_translation_E_k=batch.get("tool_translation_E_k"),
    )
    tool_gt = _aggregate_sdf(tool_gt_full.reshape(B * T, N), encoder_result.tool_patch_idx, model.patch_agg)
    obj_gt = _aggregate_sdf(obj_gt_full.reshape(B * T, N), encoder_result.obj_patch_idx, model.patch_agg)
    if model.head_mode == "point":
        tool_point_pred = model._predict_point_sdf(
            tool_flat,
            tool_tokens,
            encoder_result.tool_patch_idx,
            encoder_result.tool_patch_centers,
            model.tool_sdf_head,
        )
        obj_point_pred = model._predict_point_sdf(
            obj_flat,
            obj_tokens,
            encoder_result.obj_patch_idx,
            encoder_result.obj_patch_centers,
            model.obj_sdf_head,
        )
        tool_pred = _aggregate_sdf(tool_point_pred, encoder_result.tool_patch_idx, model.patch_agg)
        obj_pred = _aggregate_sdf(obj_point_pred, encoder_result.obj_patch_idx, model.patch_agg)
    else:
        tool_pred = model._predict_patch_sdf(tool_tokens, model.tool_sdf_head)
        obj_pred = model._predict_patch_sdf(obj_tokens, model.obj_sdf_head)

    flat_i = t
    tool_centers_E = encoder_result.tool_patch_centers[flat_i] + batch["tool_translation_E_k"][0, t].reshape(1, 3)
    obj_centers_E = encoder_result.obj_patch_centers[flat_i] + batch["object_bbox_center_E"][0].reshape(1, 3)
    return {
        "timestep": t,
        "tool_centers": _cpu(tool_centers_E),
        "obj_centers": _cpu(obj_centers_E),
        "tool_gt": _cpu(tool_gt[flat_i]),
        "tool_pred": _cpu(tool_pred[flat_i]),
        "obj_gt": _cpu(obj_gt[flat_i]),
        "obj_pred": _cpu(obj_pred[flat_i]),
        "object_id": batch.get("object_id", [""])[0],
        "tool_id": batch.get("tool_id", [""])[0],
        "pt_path": batch.get("pt_path", [""])[0],
    }


def _plot_patch_sdf(viz: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.concatenate([viz["tool_gt"], viz["tool_pred"], viz["obj_gt"], viz["obj_pred"]])
    finite = values[np.isfinite(values)]
    vmax = float(np.quantile(np.abs(finite), 0.95)) if finite.size else 1.0
    vmax = max(vmax, 1e-4)

    fig = plt.figure(figsize=(13, 10))
    title = (
        f"Patch SDF timestep={viz['timestep']} tool={viz['tool_id']} object={viz['object_id']}\n"
        f"{viz['pt_path']}"
    )
    fig.suptitle(title, fontsize=10)
    panels = [
        ("tool GT SDF", viz["tool_centers"], viz["tool_gt"]),
        ("tool PRED SDF", viz["tool_centers"], viz["tool_pred"]),
        ("object GT SDF", viz["obj_centers"], viz["obj_gt"]),
        ("object PRED SDF", viz["obj_centers"], viz["obj_pred"]),
    ]
    all_xyz = np.concatenate([viz["tool_centers"], viz["obj_centers"]], axis=0)
    mins = all_xyz.min(axis=0)
    maxs = all_xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.55, 1e-3)
    for i, (name, xyz, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        sc = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=color,
            s=28,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            depthshade=False,
        )
        ax.set_title(name)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, label="signed distance (m)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _default_output_path(checkpoint: Path, item: dict[str, Any], timestep: int) -> Path:
    stem = f"sdf_patch_{Path(str(item.get('pt_path', 'sample'))).stem}_t{int(timestep)}.png"
    return checkpoint.parent / "sdf_patch_viz" / stem


def _stage_ref(artifacts, stage: str):
    for ref in artifacts.stages:
        if ref.stage == stage:
            return ref
    raise RuntimeError(f"Experiment has no {stage!r} stage")


def _cpu(value):
    return value.detach().cpu().numpy()


if __name__ == "__main__":
    raise SystemExit(main())
