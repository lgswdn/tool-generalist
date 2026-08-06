"""Strict GG continuations for the combined D4 gripper experiments."""

from __future__ import annotations

from configs.panda_comparison_common import (
    completed_parent_checkpoint,
    configure_gg_comparison,
)
from configs.panda_experiment_common import (
    CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
    general_d4_full_rl_cfg,
    general_d4_hamnet_rl_cfg,
)


def general_d4_gg_continuation_cfg(
    name: str,
    *,
    contact_quality: str,
    architecture: str,
):
    """Build a 15k GG child from its exact completed DGN5k parent."""

    suffix = "_gg_15k"
    if not name.endswith(suffix):
        raise ValueError(f"General D4 GG experiment must end with {suffix!r}")
    parent = f"{name.removesuffix(suffix)}_dgn_5k"
    cfg = general_d4_full_rl_cfg(
        name,
        contact_quality=contact_quality,
        architecture=architecture,
    )
    contact_name = cfg.contact_gen.name
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = f"{parent}.py"
    cfg.rl.init_checkpoint = completed_parent_checkpoint(
        parent,
        contact_name=contact_name,
        expected_paths_yaml=CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
        expected_max_iterations=5_000,
        expected_num_gpus=8,
        expected_vit_attention_contract="explicit_v1",
        expected_vit_attention_mode="joint_self",
        checkpoint_filename="model_last.pt",
    )
    cfg.rl.resume_checkpoint = None
    configure_gg_comparison(cfg)
    return cfg


def general_d4_hamnet_gg_continuation_cfg(name: str):
    """Build the HAMNet GG15k child from its exact HAMNet DGN5k parent."""

    suffix = "_gg_15k"
    if not name.endswith(suffix):
        raise ValueError(f"General D4 HAMNet GG experiment must end with {suffix!r}")
    parent = f"{name.removesuffix(suffix)}_dgn_5k"
    cfg = general_d4_hamnet_rl_cfg(name)
    contact_name = cfg.contact_gen.name
    cfg.contact_gen.enabled = False
    cfg.contact_gen.regenerate = False
    cfg.pretrain.retrain = False
    cfg.pretrain_reuse = f"{parent}.py"
    cfg.rl.init_checkpoint = completed_parent_checkpoint(
        parent,
        contact_name=contact_name,
        expected_paths_yaml=CE_GENERAL_CONTACT_PRETRAIN_PATHS_YAML,
        expected_max_iterations=5_000,
        expected_num_gpus=8,
        expected_vit_attention_contract="explicit_v1",
        expected_vit_attention_mode="joint_self",
        checkpoint_filename="model_last.pt",
    )
    cfg.rl.resume_checkpoint = None
    configure_gg_comparison(cfg)
    return cfg
