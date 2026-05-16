"""Stage declarations for the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from configs.config_exp import ExpCfg


@dataclass(frozen=True)
class StageSpec:
    name: str
    artifact_type: str
    enabled: bool
    entrypoint: str | None = None
    requested: bool = False
    required: bool = False
    dependency_reason: str | None = None


def all_stages(cfg: ExpCfg) -> tuple[StageSpec, ...]:
    contact_requested = bool(cfg.contact_gen.enabled)
    contact_required = contact_stage_required(cfg)
    pretrain_required = bool(cfg.pretrain.enabled and not cfg.pretrain_reuse)
    rl_required = bool(cfg.rl.enabled)
    return (
        StageSpec(
            "contact_gen",
            "contact",
            contact_required,
            "utils.experiment.contact_stage:run_contact_stage",
            requested=contact_requested,
            required=contact_required,
            dependency_reason=(
                "pretrain_without_dataset_manifest"
                if contact_required and not contact_requested
                else None
            ),
        ),
        StageSpec(
            "pretrain",
            "encoder",
            pretrain_required,
            "utils.experiment.pretrain_stage:run_pretrain_stage",
            requested=pretrain_required,
            required=pretrain_required,
        ),
        StageSpec(
            "rl",
            "rl",
            rl_required,
            "utils.experiment.rl_stage:run_rl_stage",
            requested=rl_required,
            required=rl_required,
        ),
    )


def enabled_stages(cfg: ExpCfg) -> Iterable[StageSpec]:
    return (stage for stage in all_stages(cfg) if stage.enabled)


def contact_stage_required(cfg: ExpCfg) -> bool:
    if cfg.pretrain_reuse:
        return False
    return bool(cfg.contact_gen.enabled)
