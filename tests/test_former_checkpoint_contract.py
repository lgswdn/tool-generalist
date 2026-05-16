"""Specific contract check for the ActorCriticFormer encoder checkpoint.

Run directly with:
    python -m pytest tests/test_former_checkpoint_contract.py -q

This test intentionally has no config loader and no CLI arguments.  It verifies
that the fixed checkpoint path can strict-load into the SDF encoder shape used
by ActorCriticFormer.  The checkpoint may be an older raw checkpoint without
metadata, so tensor strict-load is the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import torch

from rsl_rl.modules.models.cloud.sdf_encoder import SDFEncoderCfg, SDFPointCloudEncoder
from pretrain.model import TCEPointCloudEncoder, TCEPointCloudEncoderCfg


CHECKPOINT = Path("/mnt/project/world_model/tool_generalist/model/encoder/tool_sdf_patch/best.pt")
EXPECTED_DIMS = {"num_pts": 512, "patch_size": 32, "encoder_channel": 128, "vit_depth": 4, "vit_heads": 4}


def _state_dict_from_checkpoint(ckpt: object) -> dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise AssertionError(f"checkpoint payload must be a dict, got {type(ckpt).__name__}")
    for key in ("model", "state_dict", "encoder"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    if all(isinstance(key, str) for key in ckpt):
        return ckpt
    raise AssertionError("checkpoint has no model/state_dict/encoder payload")


def _extract_encoder_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in ("module.encoder.", "encoder."):
        selected = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if selected:
            return selected

    root_prefixes = ("patch_enc.", "pos_embed.", "type_embed", "cls_token", "vit.", "norm.")
    if any(key.startswith(root_prefixes) for key in state_dict):
        return dict(state_dict)
    raise AssertionError("checkpoint does not contain canonical SDF encoder keys")


def test_former_checkpoint_strict_loads_into_vit4_sdf_encoder() -> None:
    assert CHECKPOINT.is_file(), f"missing checkpoint: {CHECKPOINT}"

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    encoder = SDFPointCloudEncoder(
        SDFEncoderCfg(
            num_pts=EXPECTED_DIMS["num_pts"],
            patch_size=EXPECTED_DIMS["patch_size"],
            encoder_channel=EXPECTED_DIMS["encoder_channel"],
            vit_depth=EXPECTED_DIMS["vit_depth"],
            vit_heads=EXPECTED_DIMS["vit_heads"],
            freeze=False,
            weights_path=None,
        )
    )
    encoder_state = _extract_encoder_state(_state_dict_from_checkpoint(ckpt))
    incompatible = encoder.load_state_dict(encoder_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_former_checkpoint_strict_loads_into_new_tce_encoder() -> None:
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    encoder = TCEPointCloudEncoder(
        TCEPointCloudEncoderCfg(
            num_pts=EXPECTED_DIMS["num_pts"],
            patch_size=EXPECTED_DIMS["patch_size"],
            encoder_channel=EXPECTED_DIMS["encoder_channel"],
            vit_depth=EXPECTED_DIMS["vit_depth"],
            vit_heads=EXPECTED_DIMS["vit_heads"],
            freeze=False,
        )
    )
    encoder_state = _extract_encoder_state(_state_dict_from_checkpoint(ckpt))
    incompatible = encoder.load_state_dict(encoder_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
