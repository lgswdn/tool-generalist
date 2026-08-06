"""Pretraining with the exact direct-128 RL PointNet encoder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from pretrain.model import ContactDiffusionModel


def _encoder_class():
    path = (
        Path(__file__).parents[1]
        / "rsl_rl/modules/oracle_pointcloud_pointnet_encoder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "oracle_pointcloud_pointnet_encoder", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load oracle point-cloud PointNet: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OraclePointCloudPointNetEncoder


class OraclePointCloudPointNetPretrainModel(ContactDiffusionModel):
    """Use the same direct-128 ``fast11`` encoder in pretraining and RL."""

    def __init__(self, **kwargs) -> None:
        if kwargs.get("kinematic_conditioning", False):
            raise ValueError(
                "Direct PointNet pretraining does not support "
                "kinematic conditioning"
            )
        pointcloud_input_normalization = kwargs.pop(
            "pointcloud_input_normalization", "identity"
        )
        super().__init__(**kwargs)
        self.model_family = "oracle_pointcloud_pointnet"
        self.encoder = _encoder_class()(
            num_points=kwargs.get("num_pts", 512),
            num_patches=kwargs.get("num_pts", 512)
            // kwargs.get("patch_size", 32),
            patch_size=kwargs.get("patch_size", 32),
            feature_dim=kwargs.get("encoder_channel", 128),
            feature_mode="fast11",
            use_rank10_bottleneck=False,
            token_mode="patches",
            input_normalization=pointcloud_input_normalization,
        )
