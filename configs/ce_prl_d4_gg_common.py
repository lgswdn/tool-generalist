"""Strict GG continuations for the four-layer CE-PRL contact suite."""

from __future__ import annotations

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    GENERATED_GRIPPER_NEW_PATHS_YAML,
    parallel_concavity_sdf_regression_rl_cfg,
    parallel_kinematic_conditioning_rl_cfg,
    parallel_paper_contact_quality_rl_cfg,
)


_DGN_SUFFIX = "_dgn_5k"
_GG_SUFFIX = "_gg_15k"
_CONCAVITY_VARIANT = "nonpenetrating_contact_concavity_biased"


def d4_gg_continuation_cfg(
    name: str,
    *,
    contact_variant: str,
    architecture: str = "contact",
):
    """Build a 15k GG child that strictly consumes its matching 5k parent."""

    if not name.endswith(_GG_SUFFIX):
        raise ValueError(f"D4 GG experiment name must end with {_GG_SUFFIX!r}")
    parent = f"{name.removesuffix(_GG_SUFFIX)}{_DGN_SUFFIX}"

    if architecture == "contact":
        cfg = parallel_paper_contact_quality_rl_cfg(
            name,
            contact_variant=contact_variant,
            transformer_depth=4,
            dgn_iterations=5_000,
        )
    elif architecture == "kinematic":
        cfg = parallel_kinematic_conditioning_rl_cfg(
            name,
            contact_variant=contact_variant,
            transformer_depth=4,
            dgn_iterations=5_000,
        )
    elif architecture == "sdf":
        if contact_variant != _CONCAVITY_VARIANT:
            raise ValueError("The D4 SDF architecture requires concavity contacts")
        cfg = parallel_concavity_sdf_regression_rl_cfg(
            name,
            transformer_depth=4,
            dgn_iterations=5_000,
        )
    else:
        raise ValueError(
            "D4 GG architecture must be 'contact', 'kinematic', or 'sdf'"
        )

    contact_name = cfg.contact_gen.name
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = f"{parent}.py"
    cfg.rl.init_checkpoint = completed_parent_checkpoint(
        parent,
        contact_name=contact_name,
        expected_paths_yaml=GENERATED_GRIPPER_NEW_PATHS_YAML,
        expected_max_iterations=5_000,
        expected_num_gpus=8,
        expected_vit_attention_contract="explicit_v1",
        expected_vit_attention_mode="joint_self",
        checkpoint_filename="model_last.pt",
    )
    cfg.rl.resume_checkpoint = None
    configure_gg_comparison(cfg)
    return cfg
