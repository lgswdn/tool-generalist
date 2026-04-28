"""config.py — Training configuration for SDF + Flow Matching encoder pretraining."""

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
    total_steps: int = 0      # Fixed step budget (overrides epochs; 0 = use epochs)
    batch_size: int = 256     # GLOBAL batch size, split evenly across GPUs
    lr: float = 5e-4
    resume: str = ""
    amp: bool = False  # Disabled: float16 corrupts flow matching prediction

    # Warmup: SDF+aux only phase before flow matching starts (0 = disabled).
    warmup_epochs: int = 0

    # Task selector: "joint" (SDF+flow), "movement" (SDF+movement), "sdf" (SDF-only)
    task: str = "joint"

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

    # Flow matching / diffusion head
    diffusion: bool = True
    use_mlp_head: bool = True   # MLP velocity net (proven faster for horizon=1)
    n_layer: int = 5
    n_head: int = 4
    n_emb: int = 256
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.0

    # Auxiliary regression (prevents encoder posterior collapse)
    aux_reg: bool = True
    aux_weight: float = 1.0

    # Movement prediction
    movement_pred: bool = True
    movement_n_heads: int = 2

    # Loss weights
    sdf_weight: float = 1.0
    diffusion_weight: float = 1.0
    movement_weight: float = 10.0


# Default config instance
DEFAULT_CONFIG = TrainConfig()