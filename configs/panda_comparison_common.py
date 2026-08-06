"""Shared settings for the Unicorn-ours versus DPOC RL comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ARTIFACT_ROOT = Path("/mnt/project/world_model/tool_generalist/artifacts")
FULL_YES_MANIFEST = "/mnt/project/world_model/tool_generalist/assets/DGN/full_yes.json"
GG_UNGRASPABLE_MANIFEST = (
    "../object_selections/"
    "panda_general_dpoc_gg_no_high_conf_free_but_high_conf_colliding_"
    "conf_gt_0p9_listed_scales.json"
)
FULL_YES_MAX_ITERATIONS = 5000
DGN_10K_MAX_ITERATIONS = 10000
GG_MAX_ITERATIONS = 15000
ORIGINAL_GRIPPER_MANIFEST_RESTORED_AT_UTC = "2026-07-18T13:05:08+00:00"
OLD_DPOC_ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "generated_gripper_diff_post_pretrain/contact_gen_generated_gripper/"
    "diff_post_generated_gripper_generated_gripper_diff_post/"
    "070c98e77b135e637bdeb857f81886e7d1473df2e9438c782dcce4a79eedd779/"
    "best.pt"
)
ORIGINAL_DPOC_ENCODER_CHECKPOINT = (
    "/mnt/project/world_model/tool_generalist/artifacts/encoder/"
    "single_unstable_object_center_diff_post/contact_gen_full_tool/"
    "diff_post_object_center_single_unstable_object_center_diff_post/"
    "0decde5d969739adeead4cf40baa49161b32e6d706076fb6cf36800663ca9ec0/"
    "best.pt"
)


REFERENCE_DPOC_CONTACT_DATASET = (
    "/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/"
    "contact_gen_generated_gripper/"
    "fdc5885d5d2a55727c19a6d984557275d2a7f5e48e70f6ef32e01a5bbc03daa3"
)


def completed_parent_checkpoint(
    parent_experiment: str,
    *,
    contact_name: str = "contact_gen_generated_gripper",
    encoder_family: str = "TCE",
    expected_bottleneck_rank: int | None = None,
    expected_pretrained_encoder_checkpoint: str | None = None,
    expected_paths_yaml: str | None = None,
    expected_max_iterations: int = FULL_YES_MAX_ITERATIONS,
    expected_num_gpus: int | None = None,
    expected_vit_attention_contract: str | None = None,
    expected_vit_attention_mode: str | None = None,
    created_at_or_after: str | None = None,
    checkpoint_filename: str = "model_best.pt",
) -> str:
    """Return the requested checkpoint from a matching completed parent run.

    RL run directories are timestamped, so a transfer config cannot know its
    parent's concrete directory before the parent has run.  Restricting the
    lookup to complete manifests prevents a child from consuming an early
    ``model_best.pt`` while its parent is still training.
    """

    runs_root = (
        ARTIFACT_ROOT
        / "RL"
        / parent_experiment
        / contact_name
        / encoder_family
        / parent_experiment
    )
    for run_dir in sorted(runs_root.glob("*"), reverse=True):
        manifest_path = run_dir / "manifest.json"
        checkpoint_path = run_dir / checkpoint_filename
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest.get("config_dump", {})
        except (OSError, TypeError, ValueError):
            continue
        if (
            manifest.get("status") == "complete"
            and config.get("name") == parent_experiment
            and _matches_fixed_full_yes_parent_contract(
                config,
                max_iterations=expected_max_iterations,
            )
            and (
                expected_bottleneck_rank is None
                or config.get("model", {})
                .get("tce", {})
                .get("encoder_token_bottleneck_rank")
                == expected_bottleneck_rank
            )
            and (
                expected_pretrained_encoder_checkpoint is None
                or config.get("model", {})
                .get("pretrained_encoder", {})
                .get("checkpoint_path")
                == expected_pretrained_encoder_checkpoint
            )
            and (
                expected_paths_yaml is None
                or config.get("paths_yaml") == expected_paths_yaml
            )
            and (
                expected_num_gpus is None
                or config.get("num_gpus") == expected_num_gpus
            )
            and (
                expected_vit_attention_contract is None
                or _matches_encoder_attention_contract(
                    run_dir,
                    expected_contract=expected_vit_attention_contract,
                    expected_mode=expected_vit_attention_mode,
                )
            )
            and (
                created_at_or_after is None
                or run_dir.name >= _run_id_from_utc_iso(created_at_or_after)
            )
        ):
            return str(checkpoint_path)

    # Keep the config loadable before its parent runs, but make plan/run
    # validation fail loudly instead of silently training from random weights.
    return str(
        runs_root
        / f"PENDING_VALID_COMPLETED_{expected_max_iterations}_ITERATION_PARENT_RUN"
        / checkpoint_filename
    )


def _matches_encoder_attention_contract(
    run_dir: Path,
    *,
    expected_contract: str,
    expected_mode: str | None,
) -> bool:
    """Check the concrete encoder consumed by a completed RL parent."""

    try:
        runtime = json.loads(
            (run_dir / "rl_runtime_spec.json").read_text(encoding="utf-8")
        )
        encoder_path = (
            runtime.get("policy_params", {}).get("encoder_weights_path")
        )
        if not encoder_path:
            return False
        encoder_manifest = Path(encoder_path).with_suffix(".manifest.json")
        metadata = json.loads(encoder_manifest.read_text(encoding="utf-8"))
        dims = metadata.get("model_dims", {})
    except (OSError, TypeError, ValueError):
        return False
    actual_contract = dims.get("vit_attention_contract")
    actual_mode = dims.get("vit_attention_mode")
    contract_matches = actual_contract == expected_contract
    # Before explicit propagation, the multitask constructor defaulted to
    # joint_self. Therefore legacy joint_self metadata is truthful; legacy
    # cross_only metadata is ambiguous and remains rejected.
    legacy_joint_self = (
        actual_contract is None
        and expected_mode == "joint_self"
        and actual_mode == "joint_self"
    )
    return (
        (contract_matches or legacy_joint_self)
        and (expected_mode is None or actual_mode == expected_mode)
    )


def _run_id_from_utc_iso(value: str) -> str:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _matches_fixed_full_yes_parent_contract(
    config: dict,
    *,
    max_iterations: int = FULL_YES_MAX_ITERATIONS,
) -> bool:
    """Reject completed parents produced with the earlier broken 0.01 scale setup."""

    general = config.get("general", {})
    rl = config.get("rl", {})
    scale = rl.get("domain_randomization", {}).get("object", {}).get("scale", {})
    return (
        general.get("rl_objects_manifest") == FULL_YES_MANIFEST
        and rl.get("ppo", {}).get("max_iterations") == int(max_iterations)
        and rl.get("action", {}).get("scale") == 0.06
        and rl.get("observation", {}).get("object_cloud_source") == "mesh_sampled"
        and scale.get("enabled") is True
        and scale.get("range") in ([0.1, 0.2], (0.1, 0.2))
    )


def _configure_comparison(cfg, objects_manifest: str, *, max_iterations: int) -> None:
    """Apply settings shared by full-YES parents and GG transfers."""

    cfg.general.rl_objects_manifest = objects_manifest
    cfg.rl.ppo.max_iterations = int(max_iterations)
    cfg.rl.action.scale = 0.06
    cfg.rl.observation.object_cloud_source = "mesh_sampled"
    cfg.rl.curriculum.enabled = False
    cfg.rl.curriculum.start_step = 0
    cfg.rl.curriculum.end_step = 0
    cfg.rl.curriculum.start_stable_pose_probability = 0.0
    cfg.rl.curriculum.end_stable_pose_probability = 0.0


def configure_full_yes_comparison(cfg) -> None:
    """Use the July-1 full-YES scale regime with current 0.06 actions."""

    _configure_comparison(
        cfg,
        FULL_YES_MANIFEST,
        max_iterations=FULL_YES_MAX_ITERATIONS,
    )
    cfg.rl.launch.wandb_project = "dgn_set"
    cfg.rl.domain_randomization.object.scale.enabled = True
    cfg.rl.domain_randomization.object.scale.range = (0.1, 0.2)


def configure_dgn_10k_comparison(cfg) -> None:
    """Use the same full-DGN object/scale contract for 10,000 RL iterations."""

    _configure_comparison(
        cfg,
        FULL_YES_MANIFEST,
        max_iterations=DGN_10K_MAX_ITERATIONS,
    )
    cfg.rl.launch.wandb_project = "dgn_set"
    cfg.rl.domain_randomization.object.scale.enabled = True
    cfg.rl.domain_randomization.object.scale.range = (0.1, 0.2)


def configure_gg_comparison(cfg) -> None:
    """Train GG for 15k iterations using explicit per-object scales."""

    _configure_comparison(
        cfg,
        GG_UNGRASPABLE_MANIFEST,
        max_iterations=GG_MAX_ITERATIONS,
    )
    cfg.rl.launch.wandb_project = "ungraspable_set"
    cfg.rl.domain_randomization.object.scale.enabled = False

def configure_post_contact_reuse(cfg) -> None:
    """Reuse the contact data consumed by the reference DPOC encoder."""

    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.dataset_manifest = REFERENCE_DPOC_CONTACT_DATASET
