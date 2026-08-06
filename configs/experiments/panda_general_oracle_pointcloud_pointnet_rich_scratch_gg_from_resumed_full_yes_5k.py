"""GG 15k transfer from the completed rich-scratch full-YES resume run."""

import json
from pathlib import Path

from configs.panda_comparison_common import configure_gg_comparison
from configs.panda_experiment_common import (
    generated_gripper_oracle_pointcloud_pointnet_rl_cfg,
)


PARENT_EXPERIMENT = (
    "panda_general_oracle_pointcloud_pointnet_rich_scratch_resume_to_5k"
)
LEGACY_PARENT_ARTIFACT_EXPERIMENT = (
    "panda_general_oracle_pointcloud_pointnet_rich_scratch_full_yes_5k"
)


def _completed_resumed_parent_checkpoint() -> str:
    artifact_root = Path("/mnt/project/world_model/tool_generalist/artifacts/RL")
    # The first completed resume predates the general.name/rl.name fix and was
    # therefore written below the original full-YES experiment hierarchy.
    artifact_experiments = (PARENT_EXPERIMENT, LEGACY_PARENT_ARTIFACT_EXPERIMENT)
    for artifact_experiment in artifact_experiments:
        runs_root = (
            artifact_root
            / artifact_experiment
            / "no-contact"
            / "oracle_pointcloud_pointnet"
            / artifact_experiment
        )
        for run_dir in sorted(runs_root.glob("*"), reverse=True):
            manifest_path = run_dir / "manifest.json"
            checkpoint_path = run_dir / "model_best.pt"
            if not manifest_path.is_file() or not checkpoint_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                continue
            config = manifest.get("config_dump", {})
            if (
                manifest.get("status") == "complete"
                and config.get("name") == PARENT_EXPERIMENT
                and config.get("rl", {}).get("ppo", {}).get("max_iterations") == 1540
                and config.get("rl", {}).get("action", {}).get("scale") == 0.06
            ):
                return str(checkpoint_path)
    pending_root = artifact_root / PARENT_EXPERIMENT / "no-contact"
    return str(pending_root / "PENDING_COMPLETED_RESUME_PARENT" / "model_best.pt")


EXP_CFG = generated_gripper_oracle_pointcloud_pointnet_rl_cfg(
    "panda_general_oracle_pointcloud_pointnet_rich_scratch_gg_from_resumed_full_yes_5k",
    checkpoint_path=None,
)
pointcloud = EXP_CFG.model.oracle_pointcloud_pointnet
pointcloud.feature_mode = "rich21"
pointcloud.load_fitted_weights = False
pointcloud.use_rank10_bottleneck = False
EXP_CFG.rl.init_checkpoint = _completed_resumed_parent_checkpoint()
EXP_CFG.rl.resume_checkpoint = None
configure_gg_comparison(EXP_CFG)
