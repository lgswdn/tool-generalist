"""config.py — Training configuration for SDF + Diffusion encoder pretraining."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainConfig:
    """All training parameters in one place."""

    # Data
    data_dir: str = "/mnt/project/world_model/tool_generalist/teardrop_contact/000_asym_teardrop_contour_scraper_var_000/"
    out_dir: str = "checkpoints_diffusion_sdf_2"
    val_ratio: float = 0.1
    num_workers: int = 4
    max_files: int = 0  # 0 = use all .pt files, >0 = limit

    # Training
    epochs: int = 1000
    total_steps: int = 0
    batch_size: int = 256
    lr: float = 5e-4
    resume: str = ""
    amp: bool = False  # Disabled: float16 corrupts DDPM noise prediction

    # Warmup: SDF+aux only phase before flow matching starts (0 = disabled).
    warmup_epochs: int = 0

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
    head_mode: str = "patch"  # "point" or "patch"
    patch_agg: str = "min"    # "mean", "min", "max" (used in patch mode)
    head_hidden: Tuple[int, ...] = (128, 64)  # SDF prediction MLP hidden dims

    # Diffusion head
    diffusion: bool = True
    use_mlp_head: bool = True   # MLP noise predictor (proven faster for horizon=1)
    n_layer: int = 5
    n_head: int = 4
    n_emb: int = 256
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.0

    # Auxiliary regression (prevents encoder posterior collapse)
    aux_reg: bool = True
    aux_weight: float = 1.0

    # Loss weights
    sdf_weight: float = 1.0
    diffusion_weight: float = 1.0
    movement_weight: float = 1.0

    # Task selector
    task: str = "joint"  # "joint" (SDF+flow) or "movement" (SDF+movement prediction)


# Default config instance
DEFAULT_CONFIG = TrainConfig()