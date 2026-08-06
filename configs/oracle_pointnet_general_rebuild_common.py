"""Strict artifact locations for the 200-parallel + 200-revolute rebuild."""

from __future__ import annotations

import json
from pathlib import Path

from configs.oracle_pointnet_rebuild_common import SOURCE_PRETRAIN_CHECKPOINT


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENT_CONFIG = (
    REPO_ROOT
    / "configs/experiments/ce_general_oracle_rebuild_d12_pca_source.py"
)
SOURCE_CONTACT_DIR = Path(
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_general_paper_400tools_128bin_500k/"
    "d07d6c6dc2f771d531e8ff44a0c8693e201889ef691cf5864a6aeb2eff1b5562"
)
TOOLS_ADJUSTED_JSON = (
    REPO_ROOT
    / "configs/generated_gripper_contact_assets_general_128/tools_adjusted.json"
)
TOOLS_SELECTED_JSON = (
    REPO_ROOT
    / "configs/generated_gripper_contact_assets_general_128/tools_selected.json"
)
BOTTLENECK_CONFIG = (
    REPO_ROOT
    / "configs/experiments/ce_general_oracle_rebuild_bottleneck_dgn_5k.py"
)
POINTNET_DGN_CONFIG = (
    REPO_ROOT
    / "configs/experiments/ce_general_oracle_rebuild_pointnet_dgn_5k.py"
)
POINTNET_GG_CONFIG = (
    REPO_ROOT
    / "configs/experiments/ce_general_oracle_rebuild_pointnet_gg_15k.py"
)
PIPELINE_ROOT = REPO_ROOT / "artifacts/oracle_pointnet_general_rebuild_d12"
PCA_CHECKPOINT = PIPELINE_ROOT / "encoder_token_pca.pt"
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
