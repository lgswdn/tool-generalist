#!/usr/bin/env python3
"""Run the strict depth-12 oracle rebuild over parallel and revolute grippers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.oracle_pointnet_general_rebuild_common import (
    BOTTLENECK_CONFIG,
    LINEAGE_PATH,
    PCA_CHECKPOINT,
    PIPELINE_ROOT,
    POINTNET_CHECKPOINT,
    POINTNET_DATA_DIR,
    POINTNET_DGN_CONFIG,
    POINTNET_DIR,
    POINTNET_GG_CONFIG,
    SOURCE_CONTACT_DIR,
    SOURCE_EXPERIMENT_CONFIG,
    TOOLS_ADJUSTED_JSON,
    TOOLS_SELECTED_JSON,
)
from configs.oracle_pointnet_rebuild_common import SOURCE_PRETRAIN_CHECKPOINT
from configs.panda_experiment_common import (
    CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
)
from scripts.run_ce_prl_oracle_rebuild_pipeline import (
    _find_rl_checkpoint,
    _replace_incomplete,
    _require_explicit_depth12_checkpoint,
    _require_rl_checkpoint,
    _run,
    _run_experiment,
    _sha256,
    _validate_pca,
    _validate_pointnet,
    _validate_pointnet_data,
)
from utils.assets.tool_assets import load_tool_kinematic_cloud


BOTTLENECK_NAME = "ce_general_oracle_rebuild_d12_bottleneck_dgn_5k"
POINTNET_DGN_NAME = "ce_general_oracle_rebuild_d12_pointnet_dgn_5k"
POINTNET_GG_NAME = "ce_general_oracle_rebuild_d12_pointnet_gg_15k"
DISTILLATION_CONTACTS_PER_FILE = 28


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_lineage(updates: dict[str, Any]) -> None:
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = _load_json(LINEAGE_PATH) if LINEAGE_PATH.is_file() else {}
    payload.update(updates)
    temporary = LINEAGE_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(LINEAGE_PATH)


def _require_general_source() -> Path:
    root = SOURCE_CONTACT_DIR.resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"General paper contact artifact is missing: {root}")
    manifest = _load_json(manifest_path)
    config = manifest.get("config_dump")
    contact = config.get("contact_gen") if isinstance(config, dict) else None
    if (
        manifest.get("config_hash") != root.name
        or not isinstance(config, dict)
        or config.get("paths_yaml") != CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML
        or not isinstance(contact, dict)
        or contact.get("name")
        != "contact_gen_general_paper_400tools_128bin_500k"
        or contact.get("num_pairs") != 2_000
        or contact.get("B") != 250
    ):
        raise RuntimeError(
            f"General paper contact manifest violates the rebuild contract: {manifest_path}"
        )

    selected = _load_json(TOOLS_SELECTED_JSON)
    entries = _load_json(TOOLS_ADJUSTED_JSON)
    if not isinstance(selected, list) or not isinstance(entries, list):
        raise RuntimeError("General tool catalogs must both be JSON lists")
    selected_ids = [
        str(item if isinstance(item, str) else item.get("name"))
        for item in selected
    ]
    adjusted = {str(item.get("name")): item for item in entries}
    parallel = [name for name in selected_ids if name.startswith("generated_gripper_")]
    revolute = [
        name
        for name in selected_ids
        if name.startswith("one_dof_gripper_two_finger_revolute_")
    ]
    if (
        len(selected_ids) != 400
        or len(set(selected_ids)) != 400
        or len(parallel) != 200
        or len(revolute) != 200
        or set(adjusted) != set(selected_ids)
    ):
        raise RuntimeError(
            "General tool catalog must contain exactly 200 parallel and "
            "200 revolute grippers"
        )
    for tool_id in selected_ids:
        entry = adjusted[tool_id]
        cache = Path(str(entry.get("kinematic_cloud_cache", ""))).expanduser()
        expected_source_key = (
            "source_generated_gripper_id"
            if tool_id.startswith("generated_gripper_")
            else "source_one_dof_gripper_id"
        )
        if (
            not cache.is_file()
            or "opening_fraction" not in entry
            or expected_source_key not in entry
            or not Path(str(entry.get("source_manifest", ""))).is_file()
        ):
            raise RuntimeError(
                f"{tool_id!r} lacks complete canonical cache provenance"
            )

    candidate_manifests = sorted(root.glob("*/*.candidate.manifest.json"))
    candidate_files = sorted(root.glob("*/*.candidate.pt"))
    per_tool = Counter(path.parent.name for path in candidate_files)
    if (
        len(candidate_manifests) != 2_000
        or len(candidate_files) != 2_000
        or set(per_tool) != set(selected_ids)
        or set(per_tool.values()) != {5}
    ):
        raise RuntimeError(
            "General contact artifact must contain five files for each of "
            "the 400 selected grippers"
        )
    candidate_total = 0
    for path in candidate_manifests:
        payload = _load_json(path)
        if (
            payload.get("status") != "candidate_generated"
            or payload.get("num_candidates") != 250
        ):
            raise RuntimeError(f"Incomplete general candidate manifest: {path}")
        candidate_total += int(payload["num_candidates"])
    if candidate_total != 500_000:
        raise RuntimeError(
            f"General contact candidate count must be 500000, got {candidate_total}"
        )

    # Check canonical equality at both ends and the middle of each family.
    for tool_id in (
        parallel[0],
        parallel[99],
        parallel[-1],
        revolute[0],
        revolute[99],
        revolute[-1],
    ):
        path = next((root / tool_id).glob("*.candidate.pt"))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _, cloud = load_tool_kinematic_cloud(TOOLS_ADJUSTED_JSON, tool_id)
        center = torch.as_tensor(payload["tool_bbox_center_M"], dtype=torch.float32)
        saved = torch.as_tensor(payload["tool_points_T"], dtype=torch.float32)
        expected = (cloud.to(dtype=torch.float32) - center).contiguous()
        if (
            tuple(saved.shape) != (512, 3)
            or not bool(saved.isfinite().all())
            or not torch.equal(saved.contiguous(), expected)
        ):
            raise RuntimeError(
                f"{tool_id!r} candidate does not use its canonical 128-bin cloud"
            )
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--replace-incomplete-offline",
        action="store_true",
        help="delete only incomplete PCA/PointNet outputs before rebuilding",
    )
    args = parser.parse_args()
    if args.num_gpus != 8:
        parser.error("The general pipeline requires exactly eight GPUs for a 4/4 split")
    return args


def main() -> None:
    args = parse_args()
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    pretrain_checkpoint = _require_explicit_depth12_checkpoint()
    if pretrain_checkpoint != SOURCE_PRETRAIN_CHECKPOINT.resolve():
        raise RuntimeError("Depth-12 checkpoint migration path mismatch")
    contact_dir = _require_general_source()
    _write_lineage(
        {
            "contact_dataset": str(contact_dir),
            "contact_dataset_config_hash": contact_dir.name,
            "tool_catalog": str(TOOLS_ADJUSTED_JSON.resolve()),
            "parallel_grippers": 200,
            "revolute_grippers": 200,
            "revolute_actuated_joints": 4,
            "pretrain_checkpoint": str(pretrain_checkpoint),
            "pretrain_checkpoint_sha256": _sha256(pretrain_checkpoint),
        }
    )

    if not _validate_pca(
        PCA_CHECKPOINT,
        pretrain_checkpoint=pretrain_checkpoint,
        contact_dir=contact_dir,
    ):
        if PCA_CHECKPOINT.exists() and args.replace_incomplete_offline:
            PCA_CHECKPOINT.unlink()
        _run(
            [
                sys.executable,
                "scripts/build_unicorn_ours_encoder_token_pca.py",
                "--config",
                str(SOURCE_EXPERIMENT_CONFIG),
                "--checkpoint",
                str(pretrain_checkpoint),
                "--data-dir",
                str(contact_dir),
                "--output",
                str(PCA_CHECKPOINT),
                "--max-files",
                "4096",
                "--batch-size",
                "64",
                "--num-workers",
                "4",
                "--device",
                args.device,
            ]
        )
        if not _validate_pca(
            PCA_CHECKPOINT,
            pretrain_checkpoint=pretrain_checkpoint,
            contact_dir=contact_dir,
        ):
            raise RuntimeError("General PCA stage completed without a valid checkpoint")

    teacher_checkpoint = _find_rl_checkpoint(
        BOTTLENECK_NAME,
        encoder_family="TCE",
        max_iterations=5_000,
        encoder_checkpoint=pretrain_checkpoint,
        init_checkpoint=None,
        expected_paths_yaml=CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
    )
    if teacher_checkpoint is None:
        _run_experiment(BOTTLENECK_CONFIG, num_gpus=args.num_gpus)
        teacher_checkpoint = _require_rl_checkpoint(
            BOTTLENECK_NAME,
            encoder_family="TCE",
            max_iterations=5_000,
            encoder_checkpoint=pretrain_checkpoint,
            init_checkpoint=None,
            expected_paths_yaml=CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
        )
    _write_lineage(
        {
            "pca_checkpoint": str(PCA_CHECKPOINT.resolve()),
            "rank10_teacher_checkpoint": str(teacher_checkpoint),
        }
    )

    if not _validate_pointnet_data(
        POINTNET_DATA_DIR,
        teacher_checkpoint=teacher_checkpoint,
        contact_dir=contact_dir,
        max_contacts_per_file=DISTILLATION_CONTACTS_PER_FILE,
    ):
        if POINTNET_DIR.exists():
            if not args.replace_incomplete_offline:
                raise RuntimeError(
                    f"Incomplete PointNet source exists: {POINTNET_DIR}. "
                    "Rerun with --replace-incomplete-offline."
                )
            _replace_incomplete(POINTNET_DIR, enabled=True)
        _run(
            [
                sys.executable,
                "scripts/extract_rank10_pointcloud_pointnet_source.py",
                "--rl-checkpoint",
                str(teacher_checkpoint),
                "--data-dir",
                str(contact_dir),
                "--output-dir",
                str(POINTNET_DATA_DIR),
                "--device",
                args.device,
                "--vit-attention-mode",
                "joint_self",
                "--max-files",
                "2000",
                "--max-contacts-per-file",
                str(DISTILLATION_CONTACTS_PER_FILE),
                "--validation-ratio",
                "0.1",
                "--seed",
                "0",
                "--batch-size",
                "8",
                "--num-workers",
                "0",
                "--shard-patches",
                "65536",
            ]
        )
        if not _validate_pointnet_data(
            POINTNET_DATA_DIR,
            teacher_checkpoint=teacher_checkpoint,
            contact_dir=contact_dir,
            max_contacts_per_file=DISTILLATION_CONTACTS_PER_FILE,
        ):
            raise RuntimeError(
                "General PointNet extraction completed without a valid manifest"
            )

    if not _validate_pointnet(
        POINTNET_CHECKPOINT,
        teacher_checkpoint=teacher_checkpoint,
    ):
        _run(
            [
                sys.executable,
                "scripts/train_rank10_minimal_pointnet.py",
                "--stage",
                "train",
                "--output-dir",
                str(POINTNET_DIR),
                "--device",
                args.device,
                "--epochs",
                "20",
                "--batch-size",
                "4096",
                "--learning-rate",
                "3e-4",
                "--cosine-weight",
                "0.1",
                "--seed",
                "0",
            ]
        )
        if not _validate_pointnet(
            POINTNET_CHECKPOINT,
            teacher_checkpoint=teacher_checkpoint,
        ):
            raise RuntimeError("General PointNet fitting produced no valid checkpoint")
    _write_lineage(
        {
            "pointnet_checkpoint": str(POINTNET_CHECKPOINT.resolve()),
            "pointnet_checkpoint_sha256": _sha256(POINTNET_CHECKPOINT),
        }
    )

    dgn_checkpoint = _find_rl_checkpoint(
        POINTNET_DGN_NAME,
        encoder_family="oracle_pointcloud_pointnet",
        max_iterations=5_000,
        encoder_checkpoint=POINTNET_CHECKPOINT.resolve(),
        init_checkpoint=None,
        expected_paths_yaml=CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
    )
    if dgn_checkpoint is None:
        _run_experiment(POINTNET_DGN_CONFIG, num_gpus=args.num_gpus)
        dgn_checkpoint = _require_rl_checkpoint(
            POINTNET_DGN_NAME,
            encoder_family="oracle_pointcloud_pointnet",
            max_iterations=5_000,
            encoder_checkpoint=POINTNET_CHECKPOINT.resolve(),
            init_checkpoint=None,
            expected_paths_yaml=CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
        )
    _write_lineage({"pointnet_dgn_checkpoint": str(dgn_checkpoint)})

    gg_checkpoint = _find_rl_checkpoint(
        POINTNET_GG_NAME,
        encoder_family="oracle_pointcloud_pointnet",
        max_iterations=15_000,
        encoder_checkpoint=POINTNET_CHECKPOINT.resolve(),
        init_checkpoint=dgn_checkpoint,
        expected_paths_yaml=CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
    )
    if gg_checkpoint is None:
        _run_experiment(POINTNET_GG_CONFIG, num_gpus=args.num_gpus)
        gg_checkpoint = _require_rl_checkpoint(
            POINTNET_GG_NAME,
            encoder_family="oracle_pointcloud_pointnet",
            max_iterations=15_000,
            encoder_checkpoint=POINTNET_CHECKPOINT.resolve(),
            init_checkpoint=dgn_checkpoint,
            expected_paths_yaml=CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
        )
    _write_lineage({"pointnet_gg_checkpoint": str(gg_checkpoint)})
    print(f"[general-oracle-rebuild] complete: {gg_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
