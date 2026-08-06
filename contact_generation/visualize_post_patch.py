#!/usr/bin/env python3
"""Visualize postcontact GT and prediction as a compact mesh video grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment config exposing EXP_CFG.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to pretrain artifact best.pt.")
    parser.add_argument("--output", default=None, help="Output MP4 path, or output directory when --image is set.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--index", type=int, default=0, help="First dataset item index for optional single-sample debugging.")
    parser.add_argument("--num-files", type=int, default=6, help="Number of source .pt files to show as rows.")
    parser.add_argument("--contacts-per-file", type=int, default=6, help="Number of contact cases per source .pt row.")
    parser.add_argument("--frames-per-phase", type=int, default=12, help="m in the 2m-frame video.")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--max-faces", type=int, default=10000, help="Max object faces to render. Use 0 for all.")
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    parser.add_argument("--image", action="store_true", help="Write two PNG image grids for the before and postcontact states.")
    parser.add_argument(
        "--images-per-page",
        type=int,
        default=36,
        help="Max contact cases per PNG page in --image mode.",
    )
    parser.add_argument("--no-pred", action="store_true", help="Skip checkpoint/model loading and render GT postcontact only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _log(
        "start "
        f"config={args.config} split={args.split} image={bool(args.image)} "
        f"predict={not args.no_pred} output={args.output or '<default>'}"
    )

    _log("importing torch and project modules")
    import torch

    from pretrain.dataset import NewPretrainDataset, collect_pt_files
    from pretrain.train import build_runtime_config, collate_fn, load_ckpt
    from utils.artifacts.resolver import resolve_artifacts
    from utils.config.loader import load_exp_cfg
    from utils.config.paths import load_project_paths
    from utils.experiment.effective_paths import apply_experiment_path_overrides

    _log("loading experiment config")
    cfg = load_exp_cfg(args.config)
    _log("loading project paths and resolving artifacts")
    paths = apply_experiment_path_overrides(
        cfg,
        load_project_paths(cfg.paths_yaml),
        stage="contact_gen",
    )
    artifacts = resolve_artifacts(cfg)
    pretrain_ref = _stage_ref(artifacts, "pretrain")
    _log(f"building runtime config pretrain_dir={pretrain_ref.directory}")
    runtime = build_runtime_config(cfg, paths, pretrain_ref.directory)
    checkpoint = None
    model = None
    checkpoint_condition_mean = None
    checkpoint_condition_std = None
    checkpoint_condition_eps = None
    if not args.no_pred:
        _log("loading model class and checkpoint metadata")
        from pretrain.model import ContactDiffusionModel

        checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else Path(runtime.ckpt_dir) / "best.pt"
        _log(f"using checkpoint={checkpoint}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        checkpoint_condition_mean, checkpoint_condition_std, checkpoint_condition_eps = _condition_stats_from_checkpoint(
            checkpoint,
            torch,
        )
    else:
        _log("no-pred enabled: skipping checkpoint and model loading")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    _log(f"using device={device}")
    dataset = _make_limited_split_dataset(
        NewPretrainDataset,
        collect_pt_files,
        data_dir=runtime.data_dir,
        split=args.split,
        num_files=max(1, int(args.num_files)),
        val_ratio=runtime.val_ratio,
        seed=runtime.seed,
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
        tool_mesh_contract=runtime.tool_mesh_contract,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"{args.split} split is empty")
    _log(f"dataset ready items={len(dataset)}")
    if not args.no_pred:
        _log("constructing model")
        model = ContactDiffusionModel(
            head_mode=runtime.head_mode,
            patch_agg=runtime.patch_agg,
            head_hidden=runtime.head_hidden,
            num_pts=runtime.num_pts,
            patch_size=runtime.patch_size,
            encoder_channel=runtime.encoder_channel,
            vit_depth=runtime.vit_depth,
            vit_heads=runtime.vit_heads,
            vit_attention_mode=runtime.vit_attention_mode,
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
        _log("loading checkpoint weights")
        load_ckpt(
            str(checkpoint),
            model,
            expected_vit_attention_mode=runtime.vit_attention_mode,
        )
        model.eval()
        _log("model ready")

    _log("collecting visualization rows")
    rows = _collect_post_grid_rows(
        dataset,
        model,
        collate_fn,
        torch,
        device,
        predict=not args.no_pred,
        num_files=max(1, int(args.num_files)),
        contacts_per_file=max(1, int(args.contacts_per_file)),
    )
    total_samples = sum(len(row) for row in rows)
    _log(f"collected rows={len(rows)} samples={total_samples}")

    output = (
        Path(args.output).expanduser()
        if args.output
        else _default_output_path(checkpoint, runtime, image=bool(args.image), predict=not args.no_pred)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.image:
        image_output_dir = _image_output_dir(output)
        image_output_dir.mkdir(parents=True, exist_ok=True)
        _log(f"writing PNG image pages output_dir={image_output_dir}")
        _write_post_grid_image(
            rows,
            image_output_dir,
            images_per_page=max(1, int(args.images_per_page)),
            max_faces=int(args.max_faces),
            elev=float(args.elev),
            azim=float(args.azim),
        )
    else:
        _log(f"writing MP4 video output={output}")
        _write_post_grid_video(
            rows,
            output,
            frames_per_phase=max(1, int(args.frames_per_phase)),
            fps=max(1, int(args.fps)),
            max_faces=int(args.max_faces),
            elev=float(args.elev),
            azim=float(args.azim),
        )
    print(f"[visualize_post_mesh] wrote {output}", flush=True)
    return 0


def _make_limited_split_dataset(
    dataset_cls,
    collect_pt_files_fn,
    *,
    data_dir: str,
    split: str,
    num_files: int,
    val_ratio: float,
    seed: int,
    max_files: int,
    require_movement: bool,
    num_points: int,
    num_precontact_steps: int,
    allow_mock_physics: bool,
    noise_max_trans: float,
    noise_max_rot_deg: float,
    noise_max_retries: int,
    floor_eps: float,
    validation_seed: int,
    denoise_target_mode: str,
    tool_mesh_contract: str,
):
    import random

    _log(f"scanning .pt files data_dir={data_dir}")
    files = collect_pt_files_fn(data_dir)
    _log(f"found .pt files count={len(files)}")
    if not files:
        raise RuntimeError(f"No .pt files found under {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_files > 0:
        files = files[:max_files]
    n_val = max(1, int(len(files) * val_ratio))
    val_files = files[:n_val]
    train_files = files[n_val:] or val_files
    split_files = train_files if split == "train" else val_files
    selected_files = split_files[: max(1, int(num_files))]
    if not selected_files:
        raise RuntimeError(f"{split} split has no files under {data_dir}")
    print(
        "[visualize_post_mesh] loading limited dataset "
        f"split={split} files={len(selected_files)}/{len(split_files)}",
        flush=True,
    )
    for i, path in enumerate(selected_files, start=1):
        _log(f"selected file {i}/{len(selected_files)}: {path}")
    _log("constructing NewPretrainDataset")
    return dataset_cls(
        selected_files,
        augment=False,
        require_movement=require_movement,
        num_points=num_points,
        num_precontact_steps=num_precontact_steps,
        allow_mock_physics=allow_mock_physics,
        noise_max_trans=noise_max_trans,
        noise_max_rot_deg=noise_max_rot_deg,
        noise_max_retries=noise_max_retries,
        floor_eps=floor_eps,
        validation_seed=validation_seed,
        denoise_target_mode=denoise_target_mode,
        tool_mesh_contract=tool_mesh_contract,
    )


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


def _collect_post_grid_rows(
    dataset,
    model,
    collate_fn,
    torch_module,
    device,
    *,
    predict: bool,
    num_files: int,
    contacts_per_file: int,
) -> list[list[dict[str, Any]]]:
    if not hasattr(dataset, "_index"):
        raise RuntimeError("Postcontact grid requires NewPretrainDataset._index for source .pt grouping")
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for dataset_i, (pt_path, _contact_i) in enumerate(list(dataset._index)):
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
    for row_i, pt_path in enumerate(order, start=1):
        indices = groups[pt_path]
        _log(f"processing row {row_i}/{len(order)} pt={pt_path} contacts={len(indices)}")
        row = []
        for col_i, dataset_i in enumerate(indices, start=1):
            _log(f"computing sample row={row_i} contact={col_i}/{len(indices)} dataset_index={dataset_i}")
            row.append(
                _compute_post_viz_for_index(
                    dataset,
                    dataset_i,
                    collate_fn,
                    torch_module,
                    device,
                    model,
                    predict=predict,
                )
            )
        if row:
            rows.append(row)
    return rows


def _log(message: str) -> None:
    print(f"[visualize_post_mesh] {message}", flush=True)


def _compute_post_viz_for_index(
    dataset,
    sample_index: int,
    collate_fn,
    torch_module,
    device,
    model,
    *,
    predict: bool,
) -> dict[str, Any]:
    item = dataset[sample_index]
    _log(f"loaded dataset sample index={sample_index}")
    batch = collate_fn([item])
    _log(f"collated sample index={sample_index}")
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch_module.Tensor) else value
        for key, value in batch.items()
    }
    with torch_module.no_grad():
        viz = _compute_post_viz(model, tensor_batch, predict=predict)
    _log(f"computed visualization sample index={sample_index}")
    viz["sample_index"] = int(sample_index)
    return viz


def _compute_post_viz(model, batch: dict[str, Any], *, predict: bool) -> dict[str, Any]:
    import math
    import torch
    from utils.geometry.pose import rotation_from_pose9d

    gt_pose9d = batch["target_object_post_delta9d"]
    if int(gt_pose9d.shape[0]) != 1:
        raise RuntimeError("visualize_post_patch expects a single collated sample")

    pred_pose9d = None
    if predict:
        if model is None:
            raise RuntimeError("Prediction rendering requires a loaded model")
        if "postcontact" not in model.enabled_heads:
            raise RuntimeError("Checkpoint/model was not built with the postcontact head enabled")
        tool_points_E_k = batch["tool_points_E_k"]
        object_points_E_k = batch["object_points_E_k"]
        rel_tool_object_t_k = batch["rel_tool_object_t_k"]
        B, T, N, _ = tool_points_E_k.shape

        tool_flat = tool_points_E_k.reshape(B * T, N, 3)
        obj_flat = object_points_E_k.reshape(B * T, N, 3)
        fused = model.encoder.encode(tool_flat, obj_flat).fused_tokens
        post_fused = fused.index_select(0, torch.zeros(B, dtype=torch.long, device=fused.device))
        post_rel = rel_tool_object_t_k[:, 0, :]
        post_cond = model._compose_condition(
            batch["cond_tool_post_delta9d"],
            batch["cond_object_post_delta9d"],
            batch["physics"],
            include_object_delta=False,
        )
        post_tokens = model.pose_cross_attn(post_fused, post_rel, post_cond)
        pred_pose9d = model.postcontact_head(model._pool_conditioned_tokens(post_tokens))

    obj_vertices = _first_tensor_list_item(batch["object_mesh_vertices"], "object_mesh_vertices").to(
        device=gt_pose9d.device, dtype=gt_pose9d.dtype
    )
    obj_faces = _first_tensor_list_item(batch["object_mesh_faces"], "object_mesh_faces").long()

    gt_R = rotation_from_pose9d(gt_pose9d)[0]
    pred_R = rotation_from_pose9d(pred_pose9d)[0] if pred_pose9d is not None else None
    tool_delta_R = rotation_from_pose9d(batch["cond_tool_post_delta9d"])[0]
    gt_t = gt_pose9d[0, :3]
    pred_t = pred_pose9d[0, :3] if pred_pose9d is not None else None
    tool_delta_t = batch["cond_tool_post_delta9d"][0, :3]
    gt_translation_abs = float(gt_t.norm().detach().cpu())
    gt_rotation_trace = gt_R[0, 0] + gt_R[1, 1] + gt_R[2, 2]
    gt_rotation_cos = ((gt_rotation_trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    gt_rotation_deg = float((torch.acos(gt_rotation_cos) * (180.0 / math.pi)).detach().cpu())

    object_R_E = batch["object_rotation_E"][0]
    object_t_E = batch["object_bbox_center_E"][0]
    tool_vertices = _first_tensor_list_item(batch["tool_mesh_vertices"], "tool_mesh_vertices").to(
        device=gt_pose9d.device, dtype=gt_pose9d.dtype
    )
    tool_faces = _first_tensor_list_item(batch["tool_mesh_faces"], "tool_mesh_faces").long()
    tool_R_E = batch["contact_tool_rotation_E"][0]
    tool_t_E = batch["contact_tool_translation_E"][0]

    initial_R = object_R_E
    initial_t = object_t_E
    final_gt_R = gt_R @ initial_R
    final_gt_t = initial_t + gt_t

    before_vertices = obj_vertices @ initial_R.transpose(-1, -2) + initial_t.reshape(1, 3)
    gt_vertices = obj_vertices @ final_gt_R.transpose(-1, -2) + final_gt_t.reshape(1, 3)
    tool_vertices_E = tool_vertices @ tool_R_E.transpose(-1, -2) + tool_t_E.reshape(1, 3)
    tool_post_R_E = tool_delta_R @ tool_R_E
    tool_post_t_E = tool_t_E + tool_delta_t
    tool_post_vertices_E = tool_vertices @ tool_post_R_E.transpose(-1, -2) + tool_post_t_E.reshape(1, 3)

    viz = {
        "before_vertices": _cpu(before_vertices),
        "gt_vertices": _cpu(gt_vertices),
        "faces": _cpu(obj_faces),
        "tool_vertices": _cpu(tool_vertices_E),
        "tool_post_vertices": _cpu(tool_post_vertices_E),
        "tool_faces": _cpu(tool_faces),
        "gt_pose9d": _cpu(gt_pose9d[0]),
        "gt_translation_abs": gt_translation_abs,
        "gt_rotation_deg": gt_rotation_deg,
        "object_id": batch.get("object_id", [""])[0],
        "tool_id": batch.get("tool_id", [""])[0],
        "pt_path": batch.get("pt_path", [""])[0],
    }
    if pred_R is not None and pred_t is not None and pred_pose9d is not None:
        final_pred_R = pred_R @ initial_R
        final_pred_t = initial_t + pred_t
        pred_vertices = obj_vertices @ final_pred_R.transpose(-1, -2) + final_pred_t.reshape(1, 3)
        vertex_error = (pred_vertices - gt_vertices).norm(dim=-1)
        viz.update(
            {
                "pred_vertices": _cpu(pred_vertices),
                "vertex_error": _cpu(vertex_error),
                "pred_pose9d": _cpu(pred_pose9d[0]),
            }
        )
    return viz


def _write_post_grid_video(
    rows: list[list[dict[str, Any]]],
    output: Path,
    *,
    frames_per_phase: int,
    fps: int,
    max_faces: int,
    elev: float,
    azim: float,
) -> None:
    if not rows:
        raise ValueError("No samples available for postcontact grid video")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    n_rows = len(rows)
    contacts_per_row = max(len(row) for row in rows)
    include_pred = _rows_include_pred(rows)
    panels_per_contact = 2 if include_pred else 1
    n_axis_cols = contacts_per_row * panels_per_contact
    fig_w = max(0.68 * n_axis_cols, 4.4)
    fig_h = max(0.68 * n_rows, 1.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)
    writer_title = "postcontact GT/PRED mesh grid" if include_pred else "postcontact GT mesh grid"
    writer = FFMpegWriter(fps=max(1, int(fps)), metadata={"title": writer_title})
    try:
        with writer.saving(fig, str(output), dpi=160):
            total_frames = int(frames_per_phase) * 2
            for frame_i in range(total_frames):
                show_final = frame_i >= int(frames_per_phase)
                _draw_post_grid_frame(
                    fig,
                    rows,
                    n_axis_cols=n_axis_cols,
                    include_pred=include_pred,
                    show_final=show_final,
                    max_faces=max_faces,
                    elev=elev,
                    azim=azim,
                    frame_i=frame_i,
                    total_frames=total_frames,
                )
                writer.grab_frame()
    finally:
        plt.close(fig)


def _write_post_grid_image(
    rows: list[list[dict[str, Any]]],
    output_dir: Path,
    *,
    images_per_page: int,
    max_faces: int,
    elev: float,
    azim: float,
) -> None:
    if not rows:
        raise ValueError("No samples available for postcontact grid image")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [viz for row in rows for viz in row]
    if not items:
        raise ValueError("No samples available for postcontact grid image")
    include_pred = _rows_include_pred(rows)
    pages = _paginate_items(items, images_per_page)
    total_pages = len(pages)
    for page_i, page_items in enumerate(pages, start=1):
        page_rows = _grid_rows_from_items(page_items)
        n_rows = len(page_rows)
        contacts_per_row = max(len(row) for row in page_rows)
        panels_per_contact = 2 if include_pred else 1
        n_axis_cols = contacts_per_row * panels_per_contact
        fig_w = max(0.68 * n_axis_cols, 4.4)
        fig_h = max(0.68 * n_rows, 1.0)
        for label, show_final in (("before", False), ("postcontact", True)):
            image_output = output_dir / f"page_{page_i:03d}_{label}.png"
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)
            try:
                _log(
                    f"drawing PNG page={page_i}/{total_pages} state={label} rows={n_rows} "
                    f"columns={n_axis_cols} cases={len(page_items)} include_pred={include_pred}"
                )
                _draw_post_grid_frame(
                    fig,
                    page_rows,
                    n_axis_cols=n_axis_cols,
                    include_pred=include_pred,
                    show_final=show_final,
                    max_faces=max_faces,
                    elev=elev,
                    azim=azim,
                    frame_i=0,
                    total_frames=1,
                )
                _log(f"saving PNG page={page_i}/{total_pages} state={label} output={image_output}")
                fig.savefig(image_output, bbox_inches="tight", pad_inches=0.02)
            finally:
                plt.close(fig)


def _draw_post_grid_frame(
    fig,
    rows: list[list[dict[str, Any]]],
    *,
    n_axis_cols: int,
    include_pred: bool,
    show_final: bool,
    max_faces: int,
    elev: float,
    azim: float,
    frame_i: int,
    total_frames: int,
    title: str = "",
) -> None:
    fig.clear()
    top = 0.92 if title else 0.985
    grid = fig.add_gridspec(
        len(rows),
        n_axis_cols,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=top,
        wspace=-0.36,
        hspace=-0.24,
    )
    if title:
        fig.suptitle(title, fontsize=10)
    for row_i, row in enumerate(rows):
        row_xyz = []
        for viz in row:
            row_xyz.extend([
                viz["before_vertices"],
                viz["gt_vertices"],
                viz["tool_vertices"],
                viz["tool_post_vertices"],
            ])
            if include_pred:
                row_xyz.append(viz["pred_vertices"])
        center, radius = _axis_center_radius(_concat_xyz(row_xyz), margin=0.46)
        for contact_i, viz in enumerate(row):
            gt_vertices = viz["gt_vertices"] if show_final else viz["before_vertices"]
            tool_vertices = viz["tool_post_vertices"] if show_final else viz["tool_vertices"]
            panels = [("GT", gt_vertices, (0.10, 0.34, 0.92))]
            if include_pred:
                pred_vertices = viz["pred_vertices"] if show_final else viz["before_vertices"]
                panels.append(("PRED", pred_vertices, (0.95, 0.28, 0.08)))
            for sub_i, (tag, vertices, color) in enumerate(panels):
                ax = fig.add_subplot(grid[row_i, contact_i * len(panels) + sub_i], projection="3d")
                ax.set_axis_off()
                _add_mesh(ax, tool_vertices, _subsample_faces(viz["tool_faces"], max_faces), color=(0.18, 0.18, 0.18), alpha=0.34)
                _add_mesh(ax, vertices, _subsample_faces(viz["faces"], max_faces), color=color, alpha=0.62)
                ax.view_init(elev=elev, azim=azim)
                try:
                    ax.set_proj_type("ortho")
                except Exception:
                    pass
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_zlim(center[2] - radius, center[2] + radius)

def _plot_post_mesh(viz: dict[str, Any], output: Path, *, max_faces: int, elev: float, azim: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    faces = _subsample_faces(viz["faces"], max_faces)
    all_xyz_values = [viz["before_vertices"], viz["gt_vertices"]]
    if "pred_vertices" in viz:
        all_xyz_values.append(viz["pred_vertices"])
    all_xyz = np.concatenate(all_xyz_values, axis=0)
    center, radius = _axis_center_radius(all_xyz)
    vertex_error = viz.get("vertex_error")
    if vertex_error is not None:
        finite = vertex_error[np.isfinite(vertex_error)]
        vmax = float(np.quantile(finite, 0.95)) if finite.size else 1.0
        vmax = max(vmax, 1e-5)
    else:
        vmax = 1.0

    panel_count = 3 if "pred_vertices" in viz else 2
    fig = plt.figure(figsize=(5 * panel_count, 5), dpi=180)
    panels = [
        (viz["before_vertices"], None),
        (viz["gt_vertices"], vertex_error),
    ]
    if "pred_vertices" in viz:
        panels.append((viz["pred_vertices"], vertex_error))
    for i, (vertices, error) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, panel_count, i, projection="3d")
        if error is None:
            _add_mesh(ax, vertices, faces, color=(0.62, 0.70, 0.82), alpha=0.72)
        else:
            _add_error_mesh(ax, vertices, faces, error, vmax=vmax)
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_axis_off()
    fig.tight_layout(pad=0.4)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _add_mesh(ax: Any, vertices, faces, *, color: tuple[float, float, float], alpha: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    collection = Poly3DCollection(
        vertices[faces],
        facecolors=[(*color, alpha)],
        edgecolors=[(0.08, 0.08, 0.08, 0.08)],
        linewidths=0.05,
    )
    ax.add_collection3d(collection)


def _add_error_mesh(ax: Any, vertices, faces, error, *, vmax: float) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    face_error = error[faces].mean(axis=1)
    colors = plt.get_cmap("magma")(face_error / vmax)
    colors[:, 3] = 0.82
    collection = Poly3DCollection(
        vertices[faces],
        facecolors=colors,
        edgecolors=[(0.05, 0.05, 0.05, 0.04)],
        linewidths=0.04,
    )
    ax.add_collection3d(collection)


def _first_tensor_list_item(value: Any, name: str):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list from collate_fn")
    return value[0]


def _subsample_faces(faces, max_faces: int):
    import numpy as np

    faces = np.asarray(faces, dtype=np.int64)
    if max_faces <= 0 or faces.shape[0] <= max_faces:
        return faces
    keep = np.linspace(0, faces.shape[0] - 1, num=max_faces, dtype=np.int64)
    return faces[keep]


def _axis_center_radius(points, *, margin: float = 0.58):
    import numpy as np

    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        raise ValueError("cannot render non-finite vertices")
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * float(margin), 1e-3)
    return center, radius


def _concat_xyz(values):
    import numpy as np

    return np.concatenate([np.asarray(value).reshape(-1, 3) for value in values], axis=0)


def _rows_include_pred(rows: list[list[dict[str, Any]]]) -> bool:
    flat = [viz for row in rows for viz in row]
    return bool(flat) and all("pred_vertices" in viz for viz in flat)


def _paginate_items(items: list[dict[str, Any]], images_per_page: int) -> list[list[dict[str, Any]]]:
    limit = max(1, int(images_per_page))
    return [items[start : start + limit] for start in range(0, len(items), limit)]


def _grid_rows_from_items(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    import math

    if not items:
        return []
    cols = max(1, int(math.ceil(math.sqrt(len(items)))))
    return [items[start : start + cols] for start in range(0, len(items), cols)]


def _image_output_dir(output: Path) -> Path:
    return output.with_suffix("") if output.suffix else output


def _format_gt_motion_label(viz: dict[str, Any]) -> str:
    translation = float(viz.get("gt_translation_abs", 0.0))
    rotation = float(viz.get("gt_rotation_deg", 0.0))
    return f"GT |dt|={translation:.4g} rot={rotation:.3g} deg"


def _short_pt_label(path: str) -> str:
    source = Path(str(path))
    parent = source.parent.name
    label = f"{parent}/{source.name}" if parent else source.name
    return label[:120]


def _default_output_path(checkpoint: Path | None, runtime, *, image: bool, predict: bool) -> Path:
    output_root = checkpoint.parent if checkpoint is not None else Path(runtime.ckpt_dir)
    stem = "postcontact_gt_pred_grid" if predict else "postcontact_gt_grid"
    if image:
        return output_root / "post_mesh_viz" / stem
    return output_root / "post_mesh_viz" / f"{stem}.mp4"


def _stage_ref(artifacts, stage: str):
    for ref in artifacts.stages:
        if ref.stage == stage:
            return ref
    raise RuntimeError(f"Experiment has no {stage!r} stage")


def _cpu(value):
    return value.detach().cpu().numpy()


if __name__ == "__main__":
    raise SystemExit(main())
