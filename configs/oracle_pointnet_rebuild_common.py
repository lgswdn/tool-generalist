"""Strict artifact locations for the end-to-end new-200 PointNet rebuild."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "artifacts/oracle_pointnet_rebuild_new200_d12"
SOURCE_EXPERIMENT_CONFIG = (
    REPO_ROOT
    / "configs/experiments/ce_prl_oracle_rebuild_d12_pca_source.py"
)
SOURCE_WANDB_RUN_ID = "sxy80qhc"
SOURCE_DGN_RUN_DIR = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/RL/"
    "ce_prl_unicorn_d4_full_paper_head_dgn_5k/"
    "contact_gen_prl_paper_head_500k/TCE/"
    "ce_prl_unicorn_d4_full_paper_head_dgn_5k/20260730T101638Z"
)
SOURCE_DGN_PRETRAIN_CHECKPOINT = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "ce_prl_unicorn_d4_full_paper_head_dgn_5k/"
    "contact_gen_prl_paper_head_500k/"
    "ce_prl_unicorn_d4_full_paper_head_"
    "ce_prl_unicorn_d4_full_paper_head/"
    "6dc609d2a79f945aa79a9e56974e596296da26651adeb5fb43f41468c8ff3a07/"
    "best.pt"
)
LEGACY_DEPTH12_PRETRAIN_CHECKPOINT = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "unicorn_pretrain_ours_generated_gripper/contact_gen_generated_gripper/"
    "unicorn_contact_ours_generated_gripper_"
    "unicorn_contact_ours_generated_gripper/"
    "14fba2398c961a4fc6446b54914910f92471837326a0768ff674a423175b66f0/"
    "best.pt"
)
SOURCE_PRETRAIN_CHECKPOINT = (
    PIPELINE_ROOT / "depth12_joint_self_explicit_v1.pt"
)
SOURCE_CONTACT_DIR = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_prl_paper_head_500k/"
    "f3f1461f5b99774f4b6a298ce1f030634466435b6f5fb273ef215eedc2d493e7"
)
BOTTLENECK_CONFIG = (
    REPO_ROOT
    / "configs/experiments/ce_prl_oracle_rebuild_bottleneck_dgn_5k.py"
)
POINTNET_DGN_CONFIG = (
    REPO_ROOT
    / "configs/experiments/ce_prl_oracle_rebuild_pointnet_dgn_5k.py"
)
POINTNET_GG_CONFIG = (
    REPO_ROOT
    / "configs/experiments/ce_prl_oracle_rebuild_pointnet_gg_15k.py"
)
CANONICAL_CONTACT_DIR = (
    REPO_ROOT
    / "artifacts/oracle_pointnet_rebuild_new200/"
    "canonical_paper_head_candidates"
)
PCA_CHECKPOINT = PIPELINE_ROOT / "encoder_token_pca.pt"
PROBE_DIR = PIPELINE_ROOT / "rank10_pointnet_source"
POINTNET_DIR = PIPELINE_ROOT / "fast_pointcloud11"
POINTNET_DATA_DIR = POINTNET_DIR / "data"
POINTNET_CHECKPOINT = POINTNET_DIR / "fast_pointcloud11_best.pt"
LINEAGE_PATH = PIPELINE_ROOT / "lineage.json"


def require_lineage_value(key: str) -> str:
    if not LINEAGE_PATH.is_file():
        raise RuntimeError(f"Pipeline lineage does not exist: {LINEAGE_PATH}")
    payload = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Pipeline lineage lacks a non-empty {key!r}")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline lineage {key!r} does not exist: {path}")
    return str(path)
