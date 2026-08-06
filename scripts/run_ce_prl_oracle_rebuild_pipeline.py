#!/usr/bin/env python3
"""Run the strict end-to-end new-200 PointNet rebuild pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.oracle_pointnet_rebuild_common import (
    BOTTLENECK_CONFIG,
    CANONICAL_CONTACT_DIR,
    LINEAGE_PATH,
    LEGACY_DEPTH12_PRETRAIN_CHECKPOINT,
    PCA_CHECKPOINT,
    PIPELINE_ROOT,
    POINTNET_CHECKPOINT,
    POINTNET_DATA_DIR,
    POINTNET_DGN_CONFIG,
    POINTNET_DIR,
    POINTNET_GG_CONFIG,
    SOURCE_CONTACT_DIR,
    SOURCE_DGN_PRETRAIN_CHECKPOINT,
    SOURCE_DGN_RUN_DIR,
    SOURCE_EXPERIMENT_CONFIG,
    SOURCE_PRETRAIN_CHECKPOINT,
    SOURCE_WANDB_RUN_ID,
)
from configs.panda_experiment_common import GENERATED_GRIPPER_NEW_PATHS_YAML
from scripts.train_rank10_minimal_pointnet import (
    FAST_POINT_FEATURE_NAMES,
)


ARTIFACT_ROOT = Path("/mnt/project/world_model/tool_generalist/artifacts")
BOTTLENECK_NAME = "ce_prl_oracle_rebuild_d12_bottleneck_dgn_5k"
POINTNET_DGN_NAME = "ce_prl_oracle_rebuild_d12_pointnet_dgn_5k"
POINTNET_GG_NAME = "ce_prl_oracle_rebuild_d12_pointnet_gg_15k"
LEGACY_DEPTH12_SHA256 = (
    "479295d6ea8b6e7fc9387622b6cdb490c6bf9e750d19acb5b7ed3d22f099d560"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, num_gpus: int | None = None) -> None:
    print(f"[oracle-rebuild] RUN {' '.join(command)}", flush=True)
    env = os.environ.copy()
    if num_gpus is not None:
        env["RUN_NUM_GPUS"] = str(num_gpus)
        env["RUN_TOTAL_ENVS"] = "8192"
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _run_experiment(config_path: Path, *, num_gpus: int) -> None:
    _run(
        [str(REPO_ROOT / "run.bash"), str(config_path)],
        num_gpus=num_gpus,
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


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


def _require_explicit_depth12_checkpoint() -> Path:
    source = LEGACY_DEPTH12_PRETRAIN_CHECKPOINT.resolve()
    destination = SOURCE_PRETRAIN_CHECKPOINT.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Legacy depth-12 checkpoint is missing: {source}")
    source_sha = _sha256(source)
    if source_sha != LEGACY_DEPTH12_SHA256:
        raise RuntimeError(
            f"Legacy depth-12 checkpoint SHA-256 changed: {source_sha}"
        )
    import torch

    if not destination.is_file():
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
        dims = metadata.get("model_dims") if isinstance(metadata, dict) else None
        if (
            not isinstance(dims, dict)
            or metadata.get("schema_version") != "pretrain_checkpoint_v1"
            or dims.get("num_pts") != 512
            or dims.get("patch_size") != 32
            or dims.get("encoder_channel") != 128
            or "vit_attention_mode" in dims
            or "vit_attention_contract" in dims
        ):
            raise RuntimeError(
                f"Legacy depth-12 checkpoint does not match migration contract: {source}"
            )
        migrated = dict(checkpoint)
        migrated_metadata = dict(metadata)
        migrated_dims = dict(dims)
        migrated_dims.update(
            {
                "vit_depth": 12,
                "vit_heads": 4,
                "vit_attention_mode": "joint_self",
                "vit_attention_contract": "explicit_v1",
                "kinematic_conditioning": False,
            }
        )
        migrated_metadata["model_dims"] = migrated_dims
        migrated_metadata["attention_contract_migration"] = {
            "source_checkpoint": str(source),
            "source_sha256": source_sha,
            "weights_modified": False,
            "historical_semantics": "unmasked_joint_self_attention",
        }
        migrated["metadata"] = migrated_metadata
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(migrated, temporary)
        temporary.replace(destination)
    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    dims = metadata.get("model_dims") if isinstance(metadata, dict) else None
    migration = (
        metadata.get("attention_contract_migration")
        if isinstance(metadata, dict)
        else None
    )
    if (
        not isinstance(dims, dict)
        or dims.get("vit_depth") != 12
        or dims.get("vit_attention_mode") != "joint_self"
        or dims.get("vit_attention_contract") != "explicit_v1"
        or not isinstance(migration, dict)
        or migration.get("source_sha256") != source_sha
        or migration.get("weights_modified") is not False
    ):
        raise RuntimeError(
            f"Migrated depth-12 checkpoint is invalid: {destination}"
        )
    return destination


def _require_selected_paper_head_source() -> tuple[Path, Path]:
    run_manifest_path = SOURCE_DGN_RUN_DIR / "manifest.json"
    runtime_spec_path = SOURCE_DGN_RUN_DIR / "rl_runtime_spec.json"
    if not run_manifest_path.is_file() or not runtime_spec_path.is_file():
        raise FileNotFoundError(
            f"Selected W&B run {SOURCE_WANDB_RUN_ID} lacks its local RL artifact"
        )
    run_manifest = _load_json(run_manifest_path)
    runtime_spec = _load_json(runtime_spec_path)
    config = run_manifest.get("config_dump")
    if not isinstance(config, dict):
        raise RuntimeError(f"Selected RL manifest lacks config_dump: {run_manifest_path}")
    model = config.get("model")
    contact = config.get("contact_gen")
    rl = config.get("rl")
    if not isinstance(model, dict) or not isinstance(contact, dict) or not isinstance(rl, dict):
        raise RuntimeError(f"Selected RL config is incomplete: {run_manifest_path}")
    if not (
        run_manifest.get("status") == "complete"
        and config.get("name") == "ce_prl_unicorn_d4_full_paper_head_dgn_5k"
        and config.get("paths_yaml") == GENERATED_GRIPPER_NEW_PATHS_YAML
        and config.get("num_gpus") == 8
        and model.get("tce", {}).get("vit_depth") == 4
        and model.get("tce", {}).get("vit_attention_mode") == "joint_self"
        and contact.get("name") == "contact_gen_prl_paper_head_500k"
        and rl.get("ppo", {}).get("max_iterations") == 5_000
        and runtime_spec.get("freeze_encoder") is True
        and runtime_spec.get("encoder_checkpoint")
        == str(SOURCE_DGN_PRETRAIN_CHECKPOINT)
    ):
        raise RuntimeError(
            f"Selected W&B run {SOURCE_WANDB_RUN_ID} does not match the "
            "strict paper-head D4 source contract"
        )
    if not SOURCE_DGN_PRETRAIN_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Selected paper-head source encoder is missing: "
            f"{SOURCE_DGN_PRETRAIN_CHECKPOINT}"
        )
    pretrain_manifest_path = SOURCE_DGN_PRETRAIN_CHECKPOINT.with_name(
        "manifest.json"
    )
    if not pretrain_manifest_path.is_file():
        raise FileNotFoundError(
            f"Selected paper-head encoder lacks manifest: {pretrain_manifest_path}"
        )
    if _load_json(pretrain_manifest_path).get("status") != "complete":
        raise RuntimeError(
            f"Selected paper-head encoder is not complete: {pretrain_manifest_path}"
        )
    if not SOURCE_PRETRAIN_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Migrated depth-12 bottleneck encoder is missing: "
            f"{SOURCE_PRETRAIN_CHECKPOINT}"
        )
    depth12_manifest_path = LEGACY_DEPTH12_PRETRAIN_CHECKPOINT.with_name(
        "manifest.json"
    )
    if not depth12_manifest_path.is_file():
        raise FileNotFoundError(
            f"Depth-12 bottleneck encoder lacks manifest: {depth12_manifest_path}"
        )
    depth12_manifest = _load_json(depth12_manifest_path)
    depth12_config = depth12_manifest.get("config_dump")
    depth12_tce = (
        depth12_config.get("model", {}).get("tce", {})
        if isinstance(depth12_config, dict)
        else {}
    )
    if (
        not isinstance(depth12_config, dict)
        or depth12_tce.get("vit_depth") != 12
        or depth12_tce.get("vit_attention_mode", "joint_self")
        != "joint_self"
    ):
        raise RuntimeError(
            f"Selected bottleneck checkpoint is not depth-12 joint-self: "
            f"{depth12_manifest_path}"
        )
    if not SOURCE_CONTACT_DIR.is_dir():
        raise FileNotFoundError(
            f"Selected paper-head contact dataset is missing: {SOURCE_CONTACT_DIR}"
        )
    candidate_count = sum(1 for _ in SOURCE_CONTACT_DIR.rglob("*.candidate.pt"))
    if candidate_count != 1_000:
        raise RuntimeError(
            "Selected paper-head dataset must contain exactly 1000 candidate "
            f"files, got {candidate_count}: {SOURCE_CONTACT_DIR}"
        )
    return SOURCE_CONTACT_DIR.resolve(), SOURCE_PRETRAIN_CHECKPOINT.resolve()


def _require_canonical_contact_dir(source_dir: Path) -> Path | None:
    manifest_path = CANONICAL_CONTACT_DIR / "manifest.json"
    if not manifest_path.is_file():
        if CANONICAL_CONTACT_DIR.exists():
            raise RuntimeError(
                "Incomplete canonical candidate directory exists: "
                f"{CANONICAL_CONTACT_DIR}. Move it aside before rerunning."
            )
        return None
    manifest = _load_json(manifest_path)
    expected = {
        "schema_version": "canonical_gripper_candidate_dataset_v1",
        "status": "complete",
        "source_dir": str(source_dir.resolve()),
        "file_count": 1_000,
        "candidate_count": 500_000,
        "canonical_gripper_cloud_contract": (
            "128_bins_512_corresponding_points"
        ),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Canonical candidate manifest mismatch for {key}: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    actual_count = sum(
        1 for _ in CANONICAL_CONTACT_DIR.rglob("*.candidate.pt")
    )
    if actual_count != expected["file_count"]:
        raise RuntimeError(
            "Canonical candidate file count mismatch: "
            f"{actual_count} != {expected['file_count']}"
        )
    return CANONICAL_CONTACT_DIR.resolve()


def _find_rl_checkpoint(
    experiment: str,
    *,
    encoder_family: str,
    max_iterations: int,
    encoder_checkpoint: Path,
    init_checkpoint: Path | None,
    expected_paths_yaml: str = GENERATED_GRIPPER_NEW_PATHS_YAML,
) -> Path | None:
    root = (
        ARTIFACT_ROOT
        / "RL"
        / experiment
        / "no-contact"
        / encoder_family
        / experiment
    )
    for run_dir in sorted(root.glob("*"), reverse=True):
        manifest_path = run_dir / "manifest.json"
        checkpoint = run_dir / "model_best.pt"
        if not manifest_path.is_file() or not checkpoint.is_file():
            continue
        manifest = _load_json(manifest_path)
        config = manifest.get("config_dump")
        if not isinstance(config, dict):
            continue
        rl = config.get("rl")
        model = config.get("model")
        if not isinstance(rl, dict) or not isinstance(model, dict):
            continue
        configured_encoder = (
            model.get("pretrained_encoder", {}).get("checkpoint_path")
            if isinstance(model.get("pretrained_encoder"), dict)
            else None
        )
        env = rl.get("env")
        ppo = rl.get("ppo")
        if (
            manifest.get("status") == "complete"
            and config.get("name") == experiment
            and config.get("paths_yaml") == expected_paths_yaml
            and configured_encoder == str(encoder_checkpoint)
            and isinstance(env, dict)
            and env.get("generated_parallel_finger_velocity_limit_m_s") == 0.05
            and isinstance(ppo, dict)
            and ppo.get("max_iterations") == max_iterations
            and rl.get("init_checkpoint")
            == (str(init_checkpoint) if init_checkpoint is not None else None)
        ):
            return checkpoint.resolve()
    return None


def _require_rl_checkpoint(
    experiment: str,
    *,
    encoder_family: str,
    max_iterations: int,
    encoder_checkpoint: Path,
    init_checkpoint: Path | None,
    expected_paths_yaml: str = GENERATED_GRIPPER_NEW_PATHS_YAML,
) -> Path:
    checkpoint = _find_rl_checkpoint(
        experiment,
        encoder_family=encoder_family,
        max_iterations=max_iterations,
        encoder_checkpoint=encoder_checkpoint,
        init_checkpoint=init_checkpoint,
        expected_paths_yaml=expected_paths_yaml,
    )
    if checkpoint is not None:
        return checkpoint
    root = (
        ARTIFACT_ROOT
        / "RL"
        / experiment
        / "no-contact"
        / encoder_family
        / experiment
    )
    raise RuntimeError(
        f"No completed exact {experiment} model_best.pt under {root}"
    )


def _validate_pca(
    checkpoint: Path,
    *,
    pretrain_checkpoint: Path,
    contact_dir: Path,
) -> bool:
    if not checkpoint.is_file():
        return False
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"PCA checkpoint is not a mapping: {checkpoint}")
    expected = {
        "schema_version": "unicorn_encoder_token_pca_v1",
        "checkpoint": str(pretrain_checkpoint.resolve()),
        "data_dir": str(contact_dir.resolve()),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"Existing PCA lineage mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    return True


def _validate_pointnet_data(
    data_dir: Path,
    *,
    teacher_checkpoint: Path,
    contact_dir: Path,
    max_contacts_per_file: int = 56,
) -> bool:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _load_json(manifest_path)
    expected = {
        "schema_version": "rank10_fast_pointcloud11_dataset_v1",
        "source_rl_checkpoint": str(teacher_checkpoint.resolve()),
        "source_rl_checkpoint_sha256": _sha256(teacher_checkpoint),
        "data_dir": str(contact_dir.resolve()),
        "point_feature_names": list(FAST_POINT_FEATURE_NAMES),
        "distance": "unsigned_nearest_opposite_pointcloud_point",
        "mesh_queries": False,
        "saved_contact_labels_used": False,
        "max_contacts_per_file": max_contacts_per_file,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Existing probe lineage mismatch for {key}: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    return True


def _validate_pointnet(
    checkpoint: Path,
    *,
    teacher_checkpoint: Path,
) -> bool:
    if not checkpoint.is_file():
        return False
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": "rank10_fast_pointcloud11_v2",
        "source_rl_checkpoint": str(teacher_checkpoint.resolve()),
    }
    for key, value in expected.items():
        if not isinstance(payload, dict) or payload.get(key) != value:
            raise RuntimeError(
                f"Existing PointNet lineage mismatch for {key}: "
                f"{payload.get(key) if isinstance(payload, dict) else None!r} "
                f"!= {value!r}"
            )
    return True


def _replace_incomplete(path: Path, *, enabled: bool) -> None:
    if not path.exists():
        return
    if not enabled:
        return
    print(f"[oracle-rebuild] removing incomplete offline stage: {path}", flush=True)
    shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--replace-incomplete-offline",
        action="store_true",
        help="delete only incomplete PCA/probe/PointNet outputs before rebuilding",
    )
    args = parser.parse_args()
    if args.num_gpus != 8:
        parser.error("This controlled reconstruction requires exactly 8 GPUs")
    return args


def main() -> None:
    args = parse_args()
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    explicit_depth12_checkpoint = _require_explicit_depth12_checkpoint()

    source_contact_dir, pretrain_checkpoint = _require_selected_paper_head_source()
    if pretrain_checkpoint != explicit_depth12_checkpoint:
        raise RuntimeError("Depth-12 checkpoint migration path mismatch")
    contact_dir = _require_canonical_contact_dir(source_contact_dir)
    if contact_dir is None:
        _run(
            [
                sys.executable,
                "scripts/materialize_canonical_gripper_candidates.py",
                "--source-dir",
                str(source_contact_dir),
                "--output-dir",
                str(CANONICAL_CONTACT_DIR),
            ]
        )
        contact_dir = _require_canonical_contact_dir(source_contact_dir)
        if contact_dir is None:
            raise RuntimeError(
                "Canonical candidate materialization completed without a valid dataset"
            )
    _write_lineage(
        {
            "source_wandb_run_id": SOURCE_WANDB_RUN_ID,
            "source_dgn_run_dir": str(SOURCE_DGN_RUN_DIR),
            "source_dgn_encoder_checkpoint": str(
                SOURCE_DGN_PRETRAIN_CHECKPOINT
            ),
            "source_contact_dataset": str(source_contact_dir),
            "contact_dataset": str(contact_dir),
            "pretrain_checkpoint": str(pretrain_checkpoint),
            "pretrain_checkpoint_sha256": _sha256(pretrain_checkpoint),
            "pca_source_checkpoint": str(
                LEGACY_DEPTH12_PRETRAIN_CHECKPOINT.resolve()
            ),
        }
    )

    if not _validate_pca(
        PCA_CHECKPOINT,
        pretrain_checkpoint=LEGACY_DEPTH12_PRETRAIN_CHECKPOINT.resolve(),
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
                str(LEGACY_DEPTH12_PRETRAIN_CHECKPOINT.resolve()),
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
            pretrain_checkpoint=LEGACY_DEPTH12_PRETRAIN_CHECKPOINT.resolve(),
            contact_dir=contact_dir,
        ):
            raise RuntimeError("PCA stage completed without a valid checkpoint")
    _write_lineage(
        {
            "pca_checkpoint": str(PCA_CHECKPOINT.resolve()),
            "pca_checkpoint_sha256": _sha256(PCA_CHECKPOINT),
        }
    )

    teacher_checkpoint = _find_rl_checkpoint(
        BOTTLENECK_NAME,
        encoder_family="TCE",
        max_iterations=5_000,
        encoder_checkpoint=pretrain_checkpoint,
        init_checkpoint=None,
    )
    if teacher_checkpoint is None:
        _run_experiment(BOTTLENECK_CONFIG, num_gpus=args.num_gpus)
        teacher_checkpoint = _require_rl_checkpoint(
            BOTTLENECK_NAME,
            encoder_family="TCE",
            max_iterations=5_000,
            encoder_checkpoint=pretrain_checkpoint,
            init_checkpoint=None,
        )
    _write_lineage(
        {
            "rank10_teacher_checkpoint": str(teacher_checkpoint),
            "rank10_teacher_checkpoint_sha256": _sha256(teacher_checkpoint),
        }
    )

    if not _validate_pointnet_data(
        POINTNET_DATA_DIR,
        teacher_checkpoint=teacher_checkpoint,
        contact_dir=contact_dir,
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
                "1000",
                "--max-contacts-per-file",
                "56",
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
        ):
            raise RuntimeError(
                "Point-cloud extraction completed without a valid manifest"
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
            raise RuntimeError("PointNet fitting completed without a valid checkpoint")
    _write_lineage(
        {
            "pointnet_checkpoint": str(POINTNET_CHECKPOINT.resolve()),
            "pointnet_checkpoint_sha256": _sha256(POINTNET_CHECKPOINT),
        }
    )

    pointnet_dgn_checkpoint = _find_rl_checkpoint(
        POINTNET_DGN_NAME,
        encoder_family="oracle_pointcloud_pointnet",
        max_iterations=5_000,
        encoder_checkpoint=POINTNET_CHECKPOINT.resolve(),
        init_checkpoint=None,
    )
    if pointnet_dgn_checkpoint is None:
        _run_experiment(POINTNET_DGN_CONFIG, num_gpus=args.num_gpus)
        pointnet_dgn_checkpoint = _require_rl_checkpoint(
            POINTNET_DGN_NAME,
            encoder_family="oracle_pointcloud_pointnet",
            max_iterations=5_000,
            encoder_checkpoint=POINTNET_CHECKPOINT.resolve(),
            init_checkpoint=None,
        )
    _write_lineage(
        {
            "pointnet_dgn_checkpoint": str(pointnet_dgn_checkpoint),
            "pointnet_dgn_checkpoint_sha256": _sha256(
                pointnet_dgn_checkpoint
            ),
        }
    )

    pointnet_gg_checkpoint = _find_rl_checkpoint(
        POINTNET_GG_NAME,
        encoder_family="oracle_pointcloud_pointnet",
        max_iterations=15_000,
        encoder_checkpoint=POINTNET_CHECKPOINT.resolve(),
        init_checkpoint=pointnet_dgn_checkpoint,
    )
    if pointnet_gg_checkpoint is None:
        _run_experiment(POINTNET_GG_CONFIG, num_gpus=args.num_gpus)
        pointnet_gg_checkpoint = _require_rl_checkpoint(
            POINTNET_GG_NAME,
            encoder_family="oracle_pointcloud_pointnet",
            max_iterations=15_000,
            encoder_checkpoint=POINTNET_CHECKPOINT.resolve(),
            init_checkpoint=pointnet_dgn_checkpoint,
        )
    _write_lineage(
        {
            "pointnet_gg_checkpoint": str(pointnet_gg_checkpoint),
            "pointnet_gg_checkpoint_sha256": _sha256(
                pointnet_gg_checkpoint
            ),
            "status": "complete",
        }
    )
    print(
        f"[oracle-rebuild] COMPLETE final={pointnet_gg_checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
