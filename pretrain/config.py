"""config.py — Training configuration for SDF + Diffusion encoder pretraining."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainConfig:
    """All training parameters in one place."""

    # Data
    data_dir: str = "/mnt/project/world_model/tool_generalist/teardrop_contact/000_asym_teardrop_contour_scraper_var_000/"
    out_dir: str = "checkpoints_diffusion_sdf"
    val_ratio: float = 0.1
    num_workers: int = 4

    # Training
    epochs: int = 200
    batch_size: int = 256
    lr: float = 3e-4
    resume: str = ""
    amp: bool = False  # Disabled: float16 corrupts DDPM noise prediction

    # Logging
    wandb: bool = True
    wandb_project: str = "sdf-diffusion"
    wandb_name: str = ""

    # Shared encoder
    num_pts: int = 512
    patch_size: int = 32
    encoder_channel: int = 128
    vit_depth: int = 4
    vit_heads: int = 4
    freeze_encoder: bool = False

    # SDF head
    head_mode: str = "point"  # "point" or "patch"
    patch_agg: str = "mean"   # "mean", "min", "max"
    head_hidden: Tuple[int, ...] = (128, 64)  # SDF prediction MLP hidden dims

    # Diffusion head (using TransformerForDiffusion + DDPMScheduler)
    diffusion: bool = True
    n_layer: int = 4
    n_head: int = 4
    n_emb: int = 256
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.0

    # Loss weights
    sdf_weight: float = 1.0
    diffusion_weight: float = 1.0


# Default config instance
DEFAULT_CONFIG = TrainConfig()