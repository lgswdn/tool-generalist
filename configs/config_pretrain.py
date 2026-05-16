"""Pretrain-stage config for experiment planning.

This module intentionally does not import model, torch, or dataset code.  It
only records semantic parameters needed to name and validate artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config_utils import clone_cfg


@dataclass
class PretrainTaskCfg:
    sdf: bool = False
    diffusion: bool = True
    postcontact: bool = True


@dataclass
class PretrainBatchCfg:
    batch_size: int = 128
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False


@dataclass
class PretrainOptimizerCfg:
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    scheduler: str = "cosine"
    min_learning_rate: float = 1e-6


@dataclass
class PretrainLossCfg:
    w_sdf: float = 1.0
    w_diff: float = 1.0
    w_post: float = 1.0
    sdf_loss: str = "soft_l1"
    sdf_relative_loss: bool = False
    sdf_relative_eps: float = 0.005
    denoise_rot_weight: float = 1.0
    chamfer_weight: float = 1.0
    quat_norm_beta: float = 0.1


@dataclass
class SDFTargetCfg:
    backend: str = "kaolin"
    mode: str = "signed"
    query: str = "surface_points"
    chunk_size: int = 8192
    fail_without_backend: bool = True


@dataclass
class CheckpointPolicyCfg:
    save_best: bool = True
    save_last: bool = True
    save_every_epochs: int = 1
    keep_last: int = 3
    monitor: str = "val_total_loss"
    mode: str = "min"
    best_filename: str = "best.pt"
    last_filename: str = "last.pt"
    save_optimizer: bool = True
    write_manifest: bool = True
    schema_version: str = "pretrain_checkpoint_v1"
    dataset_hash_algo: str = "sha256"
    resume_checkpoint: Optional[str] = None


@dataclass
class PretrainCfg:
    name: str = "pretrain_default"
    enabled: bool = False
    retrain: bool = False
    enabled_heads: list[str] = field(default_factory=lambda: ["diff", "postcontact"])
    dataset_manifest: Optional[str] = None
    max_files: int = 0
    val_ratio: float = 0.1
    augment: bool = True
    allow_mock_physics: bool = False
    num_precontact_steps: int = 10
    translation_noise_range: tuple[float, float] = (0.0, 0.1)
    rotation_noise_range_deg: tuple[float, float] = (0.0, 30.0)
    noise_schedule_mode: str = "interpolation"
    legal_pose_max_tries: int = 10
    floor_eps: float = 0.0
    num_query_A: int = 2
    num_query_B: int = 2
    num_query_C: int = 2
    num_query_D: int = 2
    condition_normalization: Optional[bool] = None
    condition_norm_sample_files: int = 64
    condition_norm_eps: float = 1e-4
    condition_mlp_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    cross_attn_layers: int = 2
    cross_attn_heads: int = 2
    decoder_pooling: str = "min"
    sdf_head_mode: str = "patch"
    pose_dim: int = 3
    movement_cond_dim: int = 25
    sdf_head_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    denoise_head_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    postcontact_head_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    denoise_target_mode: str = "one_step"
    validation_noising_seed: int = 12345
    fixed_validation_sampling: bool = True
    batch: PretrainBatchCfg = field(default_factory=PretrainBatchCfg)
    epochs: int = 20
    optimizer: PretrainOptimizerCfg = field(default_factory=PretrainOptimizerCfg)
    log_interval: int = 10
    logger: str = "none"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"
    loss: PretrainLossCfg = field(default_factory=PretrainLossCfg)
    sdf_target: SDFTargetCfg = field(default_factory=SDFTargetCfg)
    checkpoint_policy: CheckpointPolicyCfg = field(default_factory=CheckpointPolicyCfg)
    tasks: PretrainTaskCfg = field(default_factory=PretrainTaskCfg)

    @property
    def checkpoint_path(self) -> Optional[str]:
        return self.checkpoint_policy.resume_checkpoint

    @checkpoint_path.setter
    def checkpoint_path(self, value: Optional[str]) -> None:
        self.checkpoint_policy.resume_checkpoint = value

    @property
    def K(self) -> int:
        return self.num_precontact_steps

    @K.setter
    def K(self, value: int) -> None:
        self.num_precontact_steps = value

    @property
    def batch_size(self) -> int:
        return self.batch.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.batch.batch_size = value

    @property
    def num_workers(self) -> int:
        return self.batch.num_workers

    @num_workers.setter
    def num_workers(self, value: int) -> None:
        self.batch.num_workers = value

    @property
    def num_epochs(self) -> int:
        return self.epochs

    @num_epochs.setter
    def num_epochs(self, value: int) -> None:
        self.epochs = value

    @property
    def learning_rate(self) -> float:
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.optimizer.learning_rate = value

    @property
    def denoise_rot_weight(self) -> float:
        return self.loss.denoise_rot_weight

    @denoise_rot_weight.setter
    def denoise_rot_weight(self, value: float) -> None:
        self.loss.denoise_rot_weight = value

    @property
    def chamfer_weight(self) -> float:
        return self.loss.chamfer_weight

    @chamfer_weight.setter
    def chamfer_weight(self, value: float) -> None:
        self.loss.chamfer_weight = value

    @property
    def quat_norm_beta(self) -> float:
        return self.loss.quat_norm_beta

    @quat_norm_beta.setter
    def quat_norm_beta(self, value: float) -> None:
        self.loss.quat_norm_beta = value


DEFAULT_PRETRAIN_CFG: PretrainCfg = PretrainCfg()

SDF_CFG: PretrainCfg = PretrainCfg(
    name="sdf_only",
    enabled=True,
    enabled_heads=["sdf"],
    num_precontact_steps=1,
    tasks=PretrainTaskCfg(
        sdf=True,
        diffusion=False,
        postcontact=False,
    ),
)

SDF_DIFF_CFG: PretrainCfg = PretrainCfg(
    name="sdf_diff",
    enabled=True,
    enabled_heads=["sdf", "diff"],
    translation_noise_range=(0.0, 0.2),
    rotation_noise_range_deg=(0.0, 0.0),
    tasks=PretrainTaskCfg(
        sdf=True,
        diffusion=True,
        postcontact=False,
    ),
)

SDF_POST_CFG: PretrainCfg = PretrainCfg(
    name="sdf_post",
    enabled=True,
    enabled_heads=["sdf", "postcontact"],
    num_precontact_steps=1,
    tasks=PretrainTaskCfg(
        sdf=True,
        diffusion=False,
        postcontact=True,
    ),
)

POST_CFG: PretrainCfg = PretrainCfg(
    name="sdf_post",
    enabled=True,
    enabled_heads=["postcontact"],
    num_precontact_steps=0,
    tasks=PretrainTaskCfg(
        sdf=False,
        diffusion=False,
        postcontact=True,
    ),
)

SDF_DIFF_POST_CFG: PretrainCfg = PretrainCfg(
    name="sdf_diff_post",
    enabled=True,
    enabled_heads=["sdf", "diff", "postcontact"],
    translation_noise_range=(0.0, 0.2),
    rotation_noise_range_deg=(0.0, 0.0),
    tasks=PretrainTaskCfg(
        sdf=True,
        diffusion=True,
        postcontact=True,
    ),
)

DIFF_CFG: PretrainCfg = PretrainCfg(
    name="diff_only",
    enabled=True,
    enabled_heads=["diff"],
    translation_noise_range=(0.0, 0.2),
    rotation_noise_range_deg=(0.0, 0.0),
    tasks=PretrainTaskCfg(
        sdf=False,
        diffusion=True,
        postcontact=False,
    ),
)
