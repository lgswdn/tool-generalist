#!/usr/bin/env python3
"""Render diffusion-head denoising rollouts as one MP4 per pretrain sample."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment config exposing EXP_CFG.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to pretrain artifact best.pt.")
    parser.add_argument("--output-dir", default=None, help="Directory for MP4 files.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--index", type=int, default=0, help="First dataset item index in the selected split.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of consecutive samples to render.")
    parser.add_argument("--indices", default=None, help="Comma-separated explicit dataset indices; overrides index/num-samples.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-faces", type=int, default=8000, help="Max faces per mesh to render. Use 0 for all.")
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    parser.add_argument(
        "--contact-grid-png",
        default=None,
        help="PNG summarizing final predicted contacts. Defaults to <output-dir>/pred_contacts_grid.png.",
    )
    parser.add_argument("--grid-cols", type=int, default=6, help="Predicted states per sample row in the summary PNG.")
    parser.add_argument("--grid-num-files", type=int, default=6, help="Number of source .pt files to show in the PNG.")
    parser.add_argument("--no-videos", action="store_true", help="Only write the optional grid PNG.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    from pretrain.dataset import make_split
    from pretrain.model import ContactDiffusionModel
    from pretrain.train import build_runtime_config, collate_fn, load_ckpt
    from utils.artifacts.resolver import resolve_artifacts
    from utils.config.loader import load_exp_cfg
    from utils.config.paths import load_project_paths
    from utils.experiment.effective_paths import apply_experiment_path_overrides

    cfg = load_exp_cfg(args.config)
    paths = apply_experiment_path_overrides(cfg, load_project_paths(cfg.paths_yaml))
    artifacts = resolve_artifacts(cfg)
    pretrain_ref = _stage_ref(artifacts, "pretrain")
    runtime = build_runtime_config(cfg, paths, pretrain_ref.directory)
    if "diff" not in runtime.enabled_heads:
        raise RuntimeError(
            f"Pretrain config does not enable the diffusion head: enabled_heads={list(runtime.enabled_heads)}"
        )
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else Path(runtime.ckpt_dir) / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    checkpoint_condition_mean, checkpoint_condition_std, checkpoint_condition_eps = _condition_stats_from_checkpoint(
        checkpoint,
        torch,
    )

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_ds, val_ds = make_split(
        data_dir=runtime.data_dir,
        val_ratio=runtime.val_ratio,
        seed=runtime.seed,
        augment=False,
        max_files=runtime.max_files,
        require_movement=False,
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
        condition_mean=checkpoint_condition_mean,
        condition_std=checkpoint_condition_std,
        condition_norm_eps=checkpoint_condition_eps or runtime.condition_norm_eps,
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
        sdf_relative_loss=runtime.sdf_relative_loss,
        sdf_relative_eps=runtime.sdf_relative_eps,
    ).to(device)
    load_ckpt(str(checkpoint), model)
    model.eval()

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else checkpoint.parent / "diffusion_process_viz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_index in ([] if args.no_videos else _sample_indices(args, len(dataset))):
        item = dataset[sample_index]
        viz = _compute_viz_for_index(dataset, sample_index, collate_fn, torch, device, model)
        output = output_dir / f"diffusion_{sample_index:06d}_{_safe_stem(item)}.mp4"
        _write_diffusion_video(
            viz,
            output,
            fps=int(args.fps),
            dpi=int(args.dpi),
            max_faces=int(args.max_faces),
            elev=float(args.elev),
            azim=float(args.azim),
        )
        print(f"[visualize_diffusion_process] wrote {output}", flush=True)

    grid_rows = _collect_contact_grid_rows(
        dataset,
        model,
        collate_fn,
        torch,
        device,
        num_files=max(1, int(args.grid_num_files)),
        contacts_per_file=max(1, int(args.grid_cols)),
    )
    if args.contact_grid_png or grid_rows:
        grid_path = (
            Path(args.contact_grid_png).expanduser()
            if args.contact_grid_png
            else output_dir / "pred_contacts_grid.png"
        )
        _write_contact_grid_png(
            grid_rows,
            grid_path,
            cols=max(1, int(args.grid_cols)),
            dpi=int(args.dpi),
            max_faces=int(args.max_faces),
            elev=float(args.elev),
            azim=float(args.azim),
        )
        print(f"[visualize_diffusion_process] wrote {grid_path}", flush=True)

    return 0


def _condition_stats_from_checkpoint(checkpoint: Path, torch_module) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None, float | None]:
    payload = torch_module.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    pretrain_cfg = metadata.get("pretrain_config", {}) if isinstance(metadata, dict) else {}
    mean = pretrain_cfg.get("condition_mean") if isinstance(pretrain_cfg, dict) else None
    std = pretrain_cfg.get("condition_std") if isinstance(pretrain_cfg, dict) else None
    eps = pretrain_cfg.get("condition_norm_eps") if isinstance(pretrain_cfg, dict) else None
    if mean is None or std is None:
        return None, None, float(eps) if eps is not None else None
    return tuple(float(v) for v in mean), tuple(float(v) for v in std), float(eps) if eps is not None else None


def _sample_indices(args: argparse.Namespace, dataset_len: int) -> list[int]:
    if args.indices:
        values = [int(part.strip()) for part in str(args.indices).split(",") if part.strip()]
        return [value % dataset_len for value in values]
    count = max(1, int(args.num_samples))
    start = int(args.index)
    return [(start + offset) % dataset_len for offset in range(count)]


def _collect_contact_grid_rows(
    dataset,
    model,
    collate_fn,
    torch_module,
    device,
    *,
    num_files: int,
    contacts_per_file: int,
) -> list[list[dict[str, Any]]]:
    if not hasattr(dataset, "_index"):
        raise RuntimeError("Predicted contact grid requires NewPretrainDataset._index for source .pt grouping")
    index = list(dataset._index)
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for dataset_i, (pt_path, _contact_i) in enumerate(index):
        if pt_path not in groups:
            if len(order) >= int(num_files):
                continue
            order.append(pt_path)
            groups[pt_path] = []
        if len(groups[pt_path]) < int(contacts_per_file):
            groups[pt_path].append(dataset_i)
        if len(order) >= int(num_files) and all(len(groups[path]) >= int(contacts_per_file) for path in order):
            break

    rows: list[list[dict[str, Any]]] = []
    for pt_path in order:
        row = [
            _compute_viz_for_index(dataset, dataset_i, collate_fn, torch_module, device, model)
            for dataset_i in groups[pt_path]
        ]
        if row:
            rows.append(row)
    return rows


def _compute_viz_for_index(dataset, sample_index: int, collate_fn, torch_module, device, model) -> dict[str, Any]:
    item = dataset[sample_index]
    batch = collate_fn([item])
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch_module.Tensor) else value
        for key, value in batch.items()
    }
    with torch_module.no_grad():
        viz = _compute_diffusion_rollout(model, tensor_batch)
    viz["sample_index"] = int(sample_index)
    return viz


def _compute_diffusion_rollout(model, batch: dict[str, Any]) -> dict[str, Any]:
    import torch
    from utils.geometry.pose import pose9d_from_rt

    if "diff" not in model.enabled_heads:
        raise RuntimeError("Checkpoint/model was not built with the diffusion head enabled")
    if not hasattr(model, "diff_translation_head"):
        raise RuntimeError(
            "This visualization script expects the current translation-only diffusion head "
            "(model.diff_translation_head)."
        )

    tool_points_T = batch["tool_points_T"][0]
    object_points_O = batch["object_points_O"][0]
    object_R_E = batch["object_rotation_E"][0]
    object_t_E = batch["object_bbox_center_E"][0]
    gt_tool_R = batch["tool_rotation_E_k"][0]
    gt_tool_t = batch["tool_translation_E_k"][0]
    B, T, N, _ = batch["tool_points_E_k"].shape
    if B != 1:
        raise RuntimeError("visualize_diffusion_process expects a single collated sample")
    if T <= 1:
        raise RuntimeError("Diffusion visualization requires num_precontact_steps > 0")

    object_points_model = (object_points_O @ object_R_E.T).unsqueeze(0)

    current_R = gt_tool_R[-1].clone()
    current_t = gt_tool_t[-1].clone()
    pred_R_seq = [current_R.detach().clone()]
    pred_t_seq = [current_t.detach().clone()]
    pred_delta_seq = []
    identity_R = torch.eye(3, device=current_R.device, dtype=current_R.dtype)

    for remaining_step in range(T - 1, 0, -1):
        tool_points_model = (tool_points_T @ current_R.T).unsqueeze(0)
        rel_t = (current_t - object_t_E).reshape(1, 3)
        encoder_result = model.encoder.encode(tool_points_model, object_points_model)
        cond = torch.zeros(1, model.movement_cond_dim, device=tool_points_model.device, dtype=tool_points_model.dtype)
        diff_tokens = model.pose_cross_attn(encoder_result.fused_tokens, rel_t, cond)
        time_token = torch.tensor([remaining_step], dtype=tool_points_model.dtype, device=tool_points_model.device)
        head_input = model._pool_conditioned_tokens(diff_tokens) + model.diff_time_emb(time_token)
        pred_translation_delta = model.diff_translation_head(head_input)[0]
        pred_delta = pose9d_from_rt(
            pred_translation_delta.reshape(1, 3),
            identity_R.reshape(1, 3, 3),
        )[0]
        pred_delta_seq.append(pred_delta.detach().clone())
        current_t = current_t + pred_translation_delta
        pred_R_seq.append(current_R.detach().clone())
        pred_t_seq.append(current_t.detach().clone())

    object_vertices = _first_tensor_list_item(batch["object_mesh_vertices"], "object_mesh_vertices").to(
        device=object_R_E.device,
        dtype=object_R_E.dtype,
    )
    tool_vertices = _first_tensor_list_item(batch["tool_mesh_vertices"], "tool_mesh_vertices").to(
        device=object_R_E.device,
        dtype=object_R_E.dtype,
    )
    object_faces = _first_tensor_list_item(batch["object_mesh_faces"], "object_mesh_faces").long()
    tool_faces = _first_tensor_list_item(batch["tool_mesh_faces"], "tool_mesh_faces").long()

    object_vertices_E = object_vertices @ object_R_E.T + object_t_E.reshape(1, 3)
    pred_vertices = [
        tool_vertices @ rotation.T + translation.reshape(1, 3)
        for rotation, translation in zip(pred_R_seq, pred_t_seq)
    ]
    gt_indices = list(range(T - 1, -1, -1))
    gt_vertices = [
        tool_vertices @ gt_tool_R[index].T + gt_tool_t[index].reshape(1, 3)
        for index in gt_indices
    ]

    return {
        "object_vertices": _cpu(object_vertices_E),
        "object_faces": _cpu(object_faces),
        "pred_vertices": [_cpu(value) for value in pred_vertices],
        "gt_vertices": [_cpu(value) for value in gt_vertices],
        "tool_faces": _cpu(tool_faces),
        "remaining_steps": gt_indices,
        "pred_deltas": [_cpu(value) for value in pred_delta_seq],
        "object_id": batch.get("object_id", [""])[0],
        "tool_id": batch.get("tool_id", [""])[0],
        "pt_path": batch.get("pt_path", [""])[0],
    }


def _write_contact_grid_png(
    rows: list[list[dict[str, Any]]],
    output: Path,
    *,
    cols: int,
    dpi: int,
    max_faces: int,
    elev: float,
    azim: float,
) -> None:
    if not rows:
        raise ValueError("No samples available for contact grid PNG")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = int(cols)
    n_rows = len(rows)
    fig_w = max(0.92 * n_cols, 3.8)
    fig_h = max(0.86 * n_rows, 1.1)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    grid = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=0.99,
        wspace=-0.18,
        hspace=-0.10,
    )
    try:
        for row_i, row_items in enumerate(rows):
            row_xyz = []
            for item in row_items:
                row_xyz.extend([item["object_vertices"], item["pred_vertices"][-1]])
            center, radius = _axis_center_radius(_concat_xyz(row_xyz), margin=0.49)
            row_label = _short_pt_label(row_items[0]["pt_path"])
            for col_i in range(n_cols):
                ax = fig.add_subplot(grid[row_i, col_i], projection="3d")
                ax.set_axis_off()
                if col_i >= len(row_items):
                    continue
                item = row_items[col_i]
                object_faces = _subsample_faces(item["object_faces"], max_faces)
                tool_faces = _subsample_faces(item["tool_faces"], max_faces)
                _add_mesh(ax, item["object_vertices"], object_faces, color=(0.65, 0.67, 0.70), alpha=0.34)
                _add_mesh(ax, item["pred_vertices"][-1], tool_faces, color=(0.95, 0.28, 0.08), alpha=0.68)
                ax.view_init(elev=elev, azim=azim)
                try:
                    ax.set_proj_type("ortho")
                except Exception:
                    pass
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_zlim(center[2] - radius, center[2] + radius)
                if col_i == 0:
                    ax.set_title(row_label, fontsize=5, pad=-4.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", pad_inches=0.0)
    finally:
        plt.close(fig)

def _short_pt_label(path: str) -> str:
    source = Path(str(path))
    parent = source.parent.name
    label = f"{parent}/{source.name}" if parent else source.name
    return label[:120]


def _write_diffusion_video(
    viz: dict[str, Any],
    output: Path,
    *,
    fps: int,
    dpi: int,
    max_faces: int,
    elev: float,
    azim: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    object_faces = _subsample_faces(viz["object_faces"], max_faces)
    tool_faces = _subsample_faces(viz["tool_faces"], max_faces)
    all_xyz = [viz["object_vertices"], *viz["pred_vertices"], *viz["gt_vertices"]]
    center, radius = _axis_center_radius(_concat_xyz(all_xyz))

    fig = plt.figure(figsize=(7, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    metadata = {"title": "pretrain diffusion denoising process"}
    writer = FFMpegWriter(fps=max(1, int(fps)), metadata=metadata)
    try:
        with writer.saving(fig, str(output), dpi=dpi):
            for frame_i, remaining_step in enumerate(viz["remaining_steps"]):
                ax.clear()
                _draw_frame(
                    ax,
                    viz=viz,
                    frame_i=frame_i,
                    remaining_step=remaining_step,
                    object_faces=object_faces,
                    tool_faces=tool_faces,
                    center=center,
                    radius=radius,
                    elev=elev,
                    azim=azim,
                )
                writer.grab_frame()
    finally:
        plt.close(fig)


def _draw_frame(
    ax: Any,
    *,
    viz: dict[str, Any],
    frame_i: int,
    remaining_step: int,
    object_faces,
    tool_faces,
    center,
    radius: float,
    elev: float,
    azim: float,
) -> None:
    from matplotlib.lines import Line2D

    _add_mesh(ax, viz["object_vertices"], object_faces, color=(0.65, 0.67, 0.70), alpha=0.28)
    _add_mesh(ax, viz["gt_vertices"][frame_i], tool_faces, color=(0.10, 0.34, 0.92), alpha=0.30)
    _add_mesh(ax, viz["pred_vertices"][frame_i], tool_faces, color=(0.95, 0.28, 0.08), alpha=0.58)
    ax.set_title(
        f"Diffusion denoise remaining_step={remaining_step} frame={frame_i + 1}/{len(viz['remaining_steps'])}\n"
        f"tool={viz['tool_id']} object={viz['object_id']}",
        fontsize=9,
    )
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_axis_off()
    ax.legend(
        handles=[
            Line2D([0], [0], color=(0.65, 0.67, 0.70), lw=4, label="object"),
            Line2D([0], [0], color=(0.10, 0.34, 0.92), lw=4, label="GT tool"),
            Line2D([0], [0], color=(0.95, 0.28, 0.08), lw=4, label="pred tool"),
        ],
        loc="upper right",
        fontsize=8,
    )


def _add_mesh(ax: Any, vertices, faces, *, color: tuple[float, float, float], alpha: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    collection = Poly3DCollection(
        vertices[faces],
        facecolors=[(*color, alpha)],
        edgecolors=[(0.04, 0.04, 0.04, 0.04)],
        linewidths=0.03,
    )
    ax.add_collection3d(collection)


def _subsample_faces(faces, max_faces: int):
    import numpy as np

    faces = np.asarray(faces, dtype=np.int64)
    if int(max_faces) <= 0 or faces.shape[0] <= int(max_faces):
        return faces
    idx = np.linspace(0, faces.shape[0] - 1, int(max_faces), dtype=np.int64)
    return faces[idx]


def _axis_center_radius(xyz, *, margin: float = 0.58):
    import numpy as np

    xyz = np.asarray(xyz, dtype=float)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * float(margin), 1e-3)
    return center, radius


def _concat_xyz(values: Iterable[Any]):
    import numpy as np

    return np.concatenate([np.asarray(value).reshape(-1, 3) for value in values], axis=0)


def _first_tensor_list_item(value: Any, name: str):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list from collate_fn")
    return value[0]


def _stage_ref(artifacts, stage: str):
    for ref in artifacts.stages:
        if ref.stage == stage:
            return ref
    raise RuntimeError(f"Experiment has no {stage!r} stage")


def _safe_stem(item: dict[str, Any]) -> str:
    stem = Path(str(item.get("pt_path", "sample"))).stem
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)[:80]


def _cpu(value):
    return value.detach().cpu().numpy()


if __name__ == "__main__":
    raise SystemExit(main())
