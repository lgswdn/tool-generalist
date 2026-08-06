#!/usr/bin/env python3
"""Wait for a selected UniCORN/DGN parent run, then launch GG 15k."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_GENERATED_MANIFEST = (
    "/mnt/project/world_model/tool_generalist/gripper/generated_grippers.json"
)
OLD_GENERATED_ROOT = (
    "/mnt/project/world_model/tool_generalist/gripper/franka_with_diverse_hands"
)
NEW_GENERATED_MANIFEST = (
    "/mnt/project/world_model/tool_generalist/gripper_new/generated_grippers.json"
)
NEW_GENERATED_ROOT = (
    "/mnt/project/world_model/tool_generalist/gripper_new/franka_with_diverse_hands"
)

# parent, child, manifest, generated root, expected count
VARIANTS = {
    "cross_only_depth1": (
        "panda_general_unicorn_ours_cross_only_depth1_full_yes_5k",
        "panda_general_unicorn_ours_cross_only_depth1_gg_from_full_yes_5k",
        OLD_GENERATED_MANIFEST,
        OLD_GENERATED_ROOT,
        400,
    ),
    "intersecting_geometry": (
        "panda_general_unicorn_ours_intersecting_geometry_full_yes_5k",
        "panda_general_unicorn_ours_intersecting_geometry_gg_from_full_yes_5k",
        OLD_GENERATED_MANIFEST,
        OLD_GENERATED_ROOT,
        400,
    ),
    # Backward-compatible alias used by the original two-variant wrapper.
    "intersecting": (
        "panda_general_unicorn_ours_intersecting_geometry_full_yes_5k",
        "panda_general_unicorn_ours_intersecting_geometry_gg_from_full_yes_5k",
        OLD_GENERATED_MANIFEST,
        OLD_GENERATED_ROOT,
        400,
    ),
    "intersecting_depth1": (
        "panda_general_unicorn_ours_intersecting_depth1_full_yes_5k",
        "panda_general_unicorn_ours_intersecting_depth1_gg_from_full_yes_5k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "ce_unicorn_ours": (
        "ce_unicorn_ours_dgn_10k",
        "ce_unicorn_ours_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "ce_unicorn_ours_raw_contact": (
        "ce_unicorn_ours_raw_contact_dgn_10k",
        "ce_unicorn_ours_raw_contact_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "ce_unicorn_ours_nonpenetrating_contact": (
        "ce_unicorn_ours_nonpenetrating_contact_dgn_10k",
        "ce_unicorn_ours_nonpenetrating_contact_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "ce_rev_unicorn_ours": (
        "ce_rev_unicorn_ours_dgn_10k",
        "ce_rev_unicorn_ours_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "ce_rev_unicorn_ours_raw_contact": (
        "ce_rev_unicorn_ours_raw_contact_dgn_10k",
        "ce_rev_unicorn_ours_raw_contact_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_nonpenetrating_1mm_no_scale": (
        "ce_prl_unicorn_d1_full_nonpenetrating_contact_1mm_no_scale_dgn_5k",
        "ce_prl_unicorn_d1_full_nonpenetrating_contact_1mm_no_scale_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_paper": (
        "ce_prl_unicorn_d1_full_paper_contact_dgn_10k",
        "ce_prl_unicorn_d1_full_paper_contact_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_paper_head": (
        "ce_prl_unicorn_d1_full_paper_head_dgn_10k",
        "ce_prl_unicorn_d1_full_paper_head_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_raw": (
        "ce_prl_unicorn_d1_full_raw_contact_dgn_10k",
        "ce_prl_unicorn_d1_full_raw_contact_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_nonpenetrating": (
        "ce_prl_unicorn_d1_full_nonpenetrating_contact_dgn_10k",
        "ce_prl_unicorn_d1_full_nonpenetrating_contact_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_d4_kinematic_concavity": (
        "ce_prl_unicorn_d4_full_nonpenetrating_contact_concavity_biased_kinematic_dgn_5k",
        "ce_prl_unicorn_d4_full_nonpenetrating_contact_concavity_biased_kinematic_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_d4_kinematic_paper": (
        "ce_prl_unicorn_d4_full_paper_contact_kinematic_dgn_5k",
        "ce_prl_unicorn_d4_full_paper_contact_kinematic_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_d4_kinematic_paper_head": (
        "ce_prl_unicorn_d4_full_paper_head_kinematic_dgn_5k",
        "ce_prl_unicorn_d4_full_paper_head_kinematic_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "general_d4_paper_raw": (
        "ce_general_d4_full_paper_raw_dgn_5k",
        "ce_general_d4_full_paper_raw_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "general_d4_paper_kinematic": (
        "ce_general_d4_full_paper_kinematic_dgn_5k",
        "ce_general_d4_full_paper_kinematic_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "general_d4_concavity_global_raw": (
        "ce_general_d4_full_concavity_global_raw_dgn_5k",
        "ce_general_d4_full_concavity_global_raw_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "general_d4_concavity_global_raw_hamnet": (
        "ce_general_d4_full_concavity_global_raw_hamnet_dgn_5k",
        "ce_general_d4_full_concavity_global_raw_hamnet_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "general_d4_concavity_global_kinematic": (
        "ce_general_d4_full_concavity_global_kinematic_dgn_5k",
        "ce_general_d4_full_concavity_global_kinematic_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "general_oracle_pointnet_ggbest_frozen": (
        "ce_general_oracle_pointcloud_pointnet_ggbest_frozen_dgn_5k",
        "ce_general_oracle_pointcloud_pointnet_ggbest_frozen_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "general_oracle_pointnet_ggbest_unfrozen": (
        "ce_general_oracle_pointcloud_pointnet_ggbest_unfrozen_dgn_5k",
        "ce_general_oracle_pointcloud_pointnet_ggbest_unfrozen_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "general_oracle_pointnet_fitted_unfrozen": (
        "ce_general_oracle_pointcloud_pointnet_fitted_unfrozen_dgn_5k",
        "ce_general_oracle_pointcloud_pointnet_fitted_unfrozen_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_oracle_pointnet_ggbest_frozen": (
        "ce_prl_oracle_pointcloud_pointnet_ggbest_frozen_dgn_5k",
        "ce_prl_oracle_pointcloud_pointnet_ggbest_frozen_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_oracle_pointnet_ggbest_unfrozen": (
        "ce_prl_oracle_pointcloud_pointnet_ggbest_unfrozen_dgn_5k",
        "ce_prl_oracle_pointcloud_pointnet_ggbest_unfrozen_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "prl_oracle_pointnet_fitted_unfrozen": (
        "ce_prl_oracle_pointcloud_pointnet_fitted_unfrozen_dgn_5k",
        "ce_prl_oracle_pointcloud_pointnet_fitted_unfrozen_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "native_pointnet_post_original400_unfrozen": (
        "panda_general_native_pointnet_post_original400_dgn_5k",
        "panda_general_native_pointnet_post_original400_unfrozen_gg_15k",
        OLD_GENERATED_MANIFEST,
        OLD_GENERATED_ROOT,
        400,
    ),
    "native_pointnet_post_original400_frozen": (
        "panda_general_native_pointnet_post_original400_dgn_5k",
        "panda_general_native_pointnet_post_original400_frozen_gg_15k",
        OLD_GENERATED_MANIFEST,
        OLD_GENERATED_ROOT,
        400,
    ),
    "native_pointnet_post_original400_safe_velocity_unfrozen": (
        "panda_general_native_pointnet_post_original400_safe_velocity_unfrozen_dgn_5k",
        "panda_general_native_pointnet_post_original400_safe_velocity_unfrozen_gg_15k",
        OLD_GENERATED_MANIFEST,
        OLD_GENERATED_ROOT,
        400,
    ),
    "native_pointnet_post_original400_safe_velocity_frozen": (
        "panda_general_native_pointnet_post_original400_safe_velocity_frozen_dgn_5k",
        "panda_general_native_pointnet_post_original400_safe_velocity_frozen_gg_15k",
        OLD_GENERATED_MANIFEST,
        OLD_GENERATED_ROOT,
        400,
    ),
    "ce_general_native_pointnet_post_current_velocity_unfrozen": (
        "ce_general_native_pointnet_post_current_velocity_unfrozen_dgn_5k",
        "ce_general_native_pointnet_post_current_velocity_unfrozen_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "ce_prl_native_pointnet_post_current_velocity_unfrozen": (
        "ce_prl_native_pointnet_post_current_velocity_unfrozen_dgn_5k",
        "ce_prl_native_pointnet_post_current_velocity_unfrozen_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "ce_prl_native_pointnet_normalized_post_frozen": (
        "ce_prl_native_pointnet_normalized_post_frozen_dgn_5k",
        "ce_prl_native_pointnet_normalized_post_frozen_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
    "ce_prl_native_pointnet_normalized_post_unfrozen": (
        "ce_prl_native_pointnet_normalized_post_unfrozen_dgn_5k",
        "ce_prl_native_pointnet_normalized_post_unfrozen_gg_15k",
        NEW_GENERATED_MANIFEST,
        NEW_GENERATED_ROOT,
        200,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?")
    parser.add_argument("--child-experiment")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--check-once", action="store_true")
    args = parser.parse_args()
    if bool(args.variant) == bool(args.child_experiment):
        parser.error("provide exactly one variant or --child-experiment")
    variant = args.variant
    if args.child_experiment:
        matches = [
            key
            for key, (_, child, _, _, _) in VARIANTS.items()
            if child == args.child_experiment
        ]
        if not matches:
            return 4
        variant = matches[0]
    if variant not in VARIANTS:
        parser.error(
            f"unknown variant {variant!r}; choose one of {sorted(VARIANTS)}"
        )
    parent, child, manifest, generated_root, count = VARIANTS[variant]
    target = REPO_ROOT / "scripts" / "wait_for_full_yes_then_run_gg.py"
    oracle_pointnet = variant in {
        "general_oracle_pointnet_ggbest_frozen",
        "general_oracle_pointnet_ggbest_unfrozen",
        "general_oracle_pointnet_fitted_unfrozen",
        "prl_oracle_pointnet_ggbest_frozen",
        "prl_oracle_pointnet_ggbest_unfrozen",
        "prl_oracle_pointnet_fitted_unfrozen",
        "native_pointnet_post_original400_unfrozen",
        "native_pointnet_post_original400_frozen",
        "native_pointnet_post_original400_safe_velocity_unfrozen",
        "native_pointnet_post_original400_safe_velocity_frozen",
        "ce_general_native_pointnet_post_current_velocity_unfrozen",
        "ce_prl_native_pointnet_post_current_velocity_unfrozen",
        "ce_prl_native_pointnet_normalized_post_frozen",
        "ce_prl_native_pointnet_normalized_post_unfrozen",
    }
    command = [
        sys.executable,
        str(target),
        "--parent-experiment",
        parent,
        "--child-experiment",
        child,
        "--encoder-family",
        "oracle_pointcloud_pointnet" if oracle_pointnet else "TCE",
        "--encoder-backend",
        "oracle_pointcloud_pointnet" if oracle_pointnet else "tce",
        "--generated-gripper-manifest",
        manifest,
        "--generated-gripper-root",
        generated_root,
        "--generated-gripper-count",
        str(count),
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    if (
        variant.startswith("prl_d4_kinematic_")
        or variant.startswith("general_d4_")
        or oracle_pointnet
    ):
        command.extend(["--parent-checkpoint-filename", "model_last.pt"])
    if args.check_once:
        command.append("--check-once")
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    raise SystemExit(main())
