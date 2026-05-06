"""config.py — Hyperparameters for RPDiff-style joint SDF + denoising pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class NewPretrainConfig:
    """All hyper-parameters for new_pretrain, settable via CLI."""

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = ""
    max_files: int = 0           # 0 = use all
    val_ratio: float = 0.1
    augment: bool = True

    # ── Task selector ────────────────────────────────────────────────────
    # "sdf"      → SDF-only (contact pose)
    # "sdf-diff" → SDF + diffusion joint training
    # "corn"     → CORN object-patch contact prediction
    task: str = "sdf-diff"

    # ── SDF head ─────────────────────────────────────────────────────────
    head_mode: str = "point"       # "point" or "patch"
    patch_agg: str = "mean"        # "mean", "min", "max" (patch mode only)
    head_hidden: Tuple[int, ...] = (256, 128)

    # ── CORN contact head ────────────────────────────────────────────────
    corn_tool_root: str = "/home/galbot/tool/eef/meshdata_adjusted"
    corn_head_hidden: Tuple[int, ...] = (256, 128)
    corn_pos_weight: float = 0.0  # 0 = no explicit BCE pos_weight

    # ── Shared ViT encoder ───────────────────────────────────────────────
    num_pts: int = 512
    patch_size: int = 32
    encoder_channel: int = 128
    vit_depth: int = 12
    vit_heads: int = 4
    freeze_encoder: bool = False

    # ── Cross-attention (pose conditioning) ──────────────────────────────
    cross_attn_heads: int = 4
    cross_attn_layers: int = 4
    # tool_t - obj_centroid: relative translation only (3D); rotation baked into encoder input
    pose_dim: int = 3
    # [delta_tool_t(3), delta_tool_quat(4), delta_obj_t(3), delta_obj_quat(4)]
    movement_cond_dim: int = 14

    # ── Denoising head ───────────────────────────────────────────────────
    denoise_hidden: Tuple[int, ...] = (512, 256, 128)

    # ── Diffusion noising ────────────────────────────────────────────────
    num_diffusion_steps: int = 10
    noise_max_trans: float = 0.1   # metres
    noise_max_rot_deg: float = 30.0
    interp_trajectory: bool = True  # True=SLERP interp, False=random walk
    precise_diff_prob: bool = False  # bias toward smaller steps

    # ── Loss weights ─────────────────────────────────────────────────────
    sdf_weight: float = 1.0
    denoise_weight: float = 1.0
    denoise_rot_weight: float = 50.0
    chamfer_weight: float = 1.0    # RPDiff chamfer term weight
    quat_norm_beta: float = 0.1   # RPDiff quaternion norm regularization

    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    log_interval: int = 10
    save_interval: int = 50
    num_workers: int = 4

    # ── Checkpoint ───────────────────────────────────────────────────────
    resume: str = ""
    ckpt_dir: str = "/mnt/project/world_model/tool_generalist/model/encoder/fork_sdf_patch"

    # ── Logging ──────────────────────────────────────────────────────────
    wandb: bool = False
    wandb_project: str = "new_pretrain"
    wandb_run_name: str = ""

    seed: int = 42
