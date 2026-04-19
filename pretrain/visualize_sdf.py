#!/usr/bin/env python3
"""visualize_sdf.py — Visualize SDF predictions and encoder features (matplotlib-based).

Usage:
    # Visualize a single sample
    python visualize_sdf.py --checkpoint checkpoints/best.pt --data-dir tmp_data/

    # Compare prediction vs GT
    python visualize_sdf.py --checkpoint checkpoints/best.pt --data-dir tmp_data/ --compare-gt

    # Show encoder features (PCA)
    python visualize_sdf.py --checkpoint checkpoints/best.pt --data-dir tmp_data/ --show-features

    # Plot histograms
    python visualize_sdf.py --checkpoint checkpoints/best.pt --data-dir tmp_data/ --histogram
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_PRETRAIN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PRETRAIN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import ContactDataset, collect_pt_files
from model import SDFSegmentor


# --------------------------------------------------------------------------- #
# Color mapping utilities
# --------------------------------------------------------------------------- #

def sdf_to_color(sdf: np.ndarray, vmin: float = -0.1, vmax: float = 0.1) -> np.ndarray:
    """Map SDF values to colors: blue (inside) → white (surface) → red (outside)."""
    normalized = (sdf - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    colors = np.zeros((len(sdf), 3))
    colors[:, 0] = normalized  # Red
    colors[:, 2] = 1 - normalized  # Blue

    # White at surface
    surface_mask = np.abs(sdf) < 0.005
    colors[surface_mask] = [1, 1, 1]

    return colors


def features_to_color_pca(features: torch.Tensor) -> np.ndarray:
    """Colorize features via 3-component PCA."""
    zz = features.reshape(-1, features.shape[-1])
    u, s, v = torch.pca_lowrank(zz, 3)
    vec = (zz - zz.mean(dim=0, keepdim=True)) @ v

    values = vec.cpu().numpy()
    colors = np.zeros_like(values)

    for i in range(3):
        v_min, v_max = values[:, i].min(), values[:, i].max()
        if v_max > v_min:
            colors[:, i] = (values[:, i] - v_min) / (v_max - v_min)
        else:
            colors[:, i] = 0.5

    return colors


# --------------------------------------------------------------------------- #
# Matplotlib visualization
# --------------------------------------------------------------------------- #

def plot_point_cloud_3d(
    ax: Axes3D,
    points: np.ndarray,
    colors: np.ndarray,
    title: str = "",
    offset: tuple = (0, 0, 0),
    s: float = 1.0,
):
    """Plot 3D point cloud with colors on matplotlib axes."""
    pts = points + np.array(offset)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=s, alpha=0.8)
    ax.set_title(title)


def visualize_sample(
    model: SDFSegmentor,
    sample: dict,
    device: torch.device,
    show_features: bool = False,
    compare_gt: bool = False,
    sample_idx: int = 0,
    output_dir: str = "vis_outputs",
):
    """Visualize a single sample with matplotlib."""
    model.eval()

    with torch.no_grad():
        tool_pc = sample["tool_pc"].unsqueeze(0).to(device)
        obj_pc = sample["obj_pc"].unsqueeze(0).to(device)

        res = model.encoder.encode(tool_pc, obj_pc)

        if model.head_mode == "point":
            tool_pred = model._predict_point(
                tool_pc, res.tool_tokens, res.global_feat,
                res.tool_patch_idx, model.tool_head,
            )
            obj_pred = model._predict_point(
                obj_pc, res.obj_tokens, res.global_feat,
                res.obj_patch_idx, model.obj_head,
            )
        else:
            tool_pred = model._predict_patch(res.tool_tokens, res.global_feat, model.tool_head)
            obj_pred = model._predict_patch(res.obj_tokens, res.global_feat, model.obj_head)

    # Convert to numpy
    tool_pc_np = tool_pc.squeeze(0).cpu().numpy()
    obj_pc_np = obj_pc.squeeze(0).cpu().numpy()
    tool_pred_np = tool_pred.squeeze(0).cpu().numpy()
    obj_pred_np = obj_pred.squeeze(0).cpu().numpy()
    tool_gt_np = sample["tool_pts_sdf"].numpy()
    obj_gt_np = sample["obj_pts_sdf"].numpy()

    # Determine SDF range
    sdf_min = min(tool_pred_np.min(), obj_pred_np.min(), tool_gt_np.min(), obj_gt_np.min())
    sdf_max = max(tool_pred_np.max(), obj_pred_np.max(), tool_gt_np.max(), obj_gt_np.max())
    sdf_min = max(sdf_min, -0.1)
    sdf_max = min(sdf_max, 0.1)

    # Create figure with subplots
    n_rows = 2 if compare_gt else 1
    n_cols = 2 if show_features else 2

    fig = plt.figure(figsize=(12, 4 * n_rows))

    # Row 1: Predictions (and GT if compare_gt)
    ax1 = fig.add_subplot(n_rows, n_cols, 1, projection="3d")
    ax2 = fig.add_subplot(n_rows, n_cols, 2, projection="3d")

    # Tool prediction
    colors_tool_pred = sdf_to_color(tool_pred_np, sdf_min, sdf_max)
    plot_point_cloud_3d(ax1, tool_pc_np, colors_tool_pred, "Tool SDF (pred)", s=2)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Object prediction
    colors_obj_pred = sdf_to_color(obj_pred_np, sdf_min, sdf_max)
    plot_point_cloud_3d(ax2, obj_pc_np, colors_obj_pred, "Object SDF (pred)", s=2)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    if compare_gt:
        ax3 = fig.add_subplot(n_rows, n_cols, 3, projection="3d")
        ax4 = fig.add_subplot(n_rows, n_cols, 4, projection="3d")

        # Tool GT
        colors_tool_gt = sdf_to_color(tool_gt_np, sdf_min, sdf_max)
        plot_point_cloud_3d(ax3, tool_pc_np, colors_tool_gt, "Tool SDF (GT)", s=2)

        # Object GT
        colors_obj_gt = sdf_to_color(obj_gt_np, sdf_min, sdf_max)
        plot_point_cloud_3d(ax4, obj_pc_np, colors_obj_gt, "Object SDF (GT)", s=2)

    if show_features:
        # Add feature visualization subplot
        ax5 = fig.add_subplot(n_rows, n_cols + 2, n_cols + 1, projection="3d")
        ax6 = fig.add_subplot(n_rows, n_cols + 2, n_cols + 2, projection="3d")

        tool_tokens = res.tool_tokens.squeeze(0).cpu()
        obj_tokens = res.obj_tokens.squeeze(0).cpu()

        colors_tool_feat = features_to_color_pca(tool_tokens)
        colors_obj_feat = features_to_color_pca(obj_tokens)

        # Patch centers
        tool_patch_idx = res.tool_patch_idx.squeeze(0).cpu().numpy()
        obj_patch_idx = res.obj_patch_idx.squeeze(0).cpu().numpy()

        tool_centers = tool_pc_np[tool_patch_idx[:, 0]]
        obj_centers = obj_pc_np[obj_patch_idx[:, 0]]

        plot_point_cloud_3d(ax5, tool_centers, colors_tool_feat, "Tool Features (PCA)", s=20)
        plot_point_cloud_3d(ax6, obj_centers, colors_obj_feat, "Object Features (PCA)", s=20)

    plt.tight_layout()

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"sdf_sample_{sample_idx}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")

    # Print statistics
    print(f"\n--- Sample {sample_idx} ---")
    print(f"Tool SDF pred: min={tool_pred_np.min():.4f}, max={tool_pred_np.max():.4f}, mean={tool_pred_np.mean():.4f}")
    print(f"Object SDF pred: min={obj_pred_np.min():.4f}, max={obj_pred_np.max():.4f}, mean={obj_pred_np.mean():.4f}")
    if compare_gt:
        print(f"Tool SDF GT: min={tool_gt_np.min():.4f}, max={tool_gt_np.max():.4f}, mean={tool_gt_np.mean():.4f}")
        print(f"Object SDF GT: min={obj_gt_np.min():.4f}, max={obj_gt_np.max():.4f}, mean={obj_gt_np.mean():.4f}")
        tool_err = np.abs(tool_pred_np - tool_gt_np).mean()
        obj_err = np.abs(obj_pred_np - obj_gt_np).mean()
        print(f"MAE: tool={tool_err:.4f}, object={obj_err:.4f}")


def visualize_histogram(
    model: SDFSegmentor,
    sample: dict,
    device: torch.device,
    sample_idx: int = 0,
    output_dir: str = "vis_outputs",
):
    """Plot histogram of predicted vs GT SDF values."""
    model.eval()

    with torch.no_grad():
        tool_pc = sample["tool_pc"].unsqueeze(0).to(device)
        obj_pc = sample["obj_pc"].unsqueeze(0).to(device)

        tool_pred, obj_pred = model(tool_pc, obj_pc)

    tool_pred_np = tool_pred.squeeze(0).cpu().numpy()
    obj_pred_np = obj_pred.squeeze(0).cpu().numpy()
    tool_gt_np = sample["tool_pts_sdf"].numpy()
    obj_gt_np = sample["obj_pts_sdf"].numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Tool histogram
    axes[0, 0].hist(tool_pred_np, bins=50, alpha=0.7, label="Pred", color="blue")
    axes[0, 0].hist(tool_gt_np, bins=50, alpha=0.7, label="GT", color="orange")
    axes[0, 0].set_title("Tool SDF Distribution")
    axes[0, 0].legend()

    # Object histogram
    axes[0, 1].hist(obj_pred_np, bins=50, alpha=0.7, label="Pred", color="blue")
    axes[0, 1].hist(obj_gt_np, bins=50, alpha=0.7, label="GT", color="orange")
    axes[0, 1].set_title("Object SDF Distribution")
    axes[0, 1].legend()

    # Tool scatter
    axes[1, 0].scatter(tool_gt_np, tool_pred_np, alpha=0.5, s=1)
    axes[1, 0].set_xlabel("GT SDF")
    axes[1, 0].set_ylabel("Pred SDF")
    axes[1, 0].set_title("Tool: Pred vs GT")
    lims = [min(tool_gt_np.min(), tool_pred_np.min()), max(tool_gt_np.max(), tool_pred_np.max())]
    axes[1, 0].plot(lims, lims, "r--", label="Ideal")
    axes[1, 0].legend()

    # Object scatter
    axes[1, 1].scatter(obj_gt_np, obj_pred_np, alpha=0.5, s=1)
    axes[1, 1].set_xlabel("GT SDF")
    axes[1, 1].set_ylabel("Pred SDF")
    axes[1, 1].set_title("Object: Pred vs GT")
    lims = [min(obj_gt_np.min(), obj_pred_np.min()), max(obj_gt_np.max(), obj_pred_np.max())]
    axes[1, 1].plot(lims, lims, "r--", label="Ideal")
    axes[1, 1].legend()

    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"sdf_histogram_{sample_idx}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histogram to {output_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", default="tmp_data", help="Data directory")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to visualize")
    parser.add_argument("--show-features", action="store_true", help="Visualize encoder features (PCA)")
    parser.add_argument("--compare-gt", action="store_true", help="Show GT SDF alongside prediction")
    parser.add_argument("--histogram", action="store_true", help="Plot SDF histograms")
    parser.add_argument("--head-mode", default="point", choices=["point", "patch"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="vis_outputs", help="Output directory for images")

    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    model_state = ckpt.get("model", ckpt)
    epoch = ckpt.get("epoch", "unknown")
    print(f"Loaded checkpoint from epoch {epoch}")

    # Create dataset
    pt_files = collect_pt_files(args.data_dir)
    if not pt_files:
        raise RuntimeError(f"No .pt files found under {args.data_dir}")
    dataset = ContactDataset(pt_files)
    sample = dataset[0]
    num_pts = sample["tool_pc"].shape[0]

    # Create model
    model = SDFSegmentor(
        head_mode=args.head_mode,
        num_pts=num_pts,
    ).to(args.device)

    model.load_state_dict(model_state, strict=False)
    print(f"Model loaded: {args.head_mode} mode, {num_pts} pts")

    # Visualize
    for i in range(args.num_samples):
        sample = dataset[i]

        if args.histogram:
            visualize_histogram(model, sample, args.device, sample_idx=i, output_dir=args.output_dir)
        else:
            visualize_sample(
                model, sample, args.device,
                show_features=args.show_features,
                compare_gt=args.compare_gt,
                sample_idx=i,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()