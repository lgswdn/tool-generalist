"""Contact pretraining for the unsigned point-to-mesh patchwise PointNet."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from pretrain.unicorn_model import UnicornPretrainModel
from utils.geometry.sdf import mutual_unsigned_mesh_distance_env_frame


def _encoder_class():
    path = Path(__file__).parents[1] / "rsl_rl/modules/oracle_pointmesh_pointnet_encoder.py"
    spec = importlib.util.spec_from_file_location("oracle_pointmesh_pointnet_encoder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load oracle pointmesh PointNet encoder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OraclePointMeshPointNetEncoder


class OraclePointMeshPointNetPretrainModel(UnicornPretrainModel):
    """Predict patch contact from continuous ``(xyz, unsigned mesh distance)`` tokens."""

    def __init__(
        self,
        *,
        coordinate_scale_m: float = 0.30,
        distance_scale_m: float = 0.10,
        normalization_clip: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_family = "oracle_pointmesh_pointnet"
        self.encoder = _encoder_class()(
            num_points=kwargs.get("num_points", 512),
            num_patches=kwargs.get("num_patches", 16),
            patch_size=kwargs.get("patch_size", 32),
            feature_dim=kwargs.get("encoder_channel", 128),
            coordinate_scale_m=coordinate_scale_m,
            distance_scale_m=distance_scale_m,
            normalization_clip=normalization_clip,
        )

    def forward(
        self,
        *,
        tool_points_E_k: torch.Tensor,
        object_points_E_k: torch.Tensor,
        rel_tool_object_t_k: torch.Tensor,
        object_mesh_vertices,
        object_mesh_faces,
        tool_mesh_vertices,
        tool_mesh_faces,
        object_rotation_E: torch.Tensor,
        object_bbox_center_E: torch.Tensor,
        tool_rotation_E_k: torch.Tensor,
        tool_translation_E_k: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if tool_points_E_k.shape[1] != 1 or object_points_E_k.shape[1] != 1:
            raise RuntimeError("oracle_pointmesh_contact requires exactly one contact timestep")
        tool_points = tool_points_E_k[:, 0]
        object_points = object_points_E_k[:, 0]
        encoder_tool_points = tool_points
        if self.encoder_input_centering == "object_center":
            encoder_tool_points = tool_points + rel_tool_object_t_k[:, 0].unsqueeze(-2)

        tool_points_env = tool_points + tool_translation_E_k[:, 0].unsqueeze(-2)
        object_points_env = object_points + object_bbox_center_E.unsqueeze(-2)
        with torch.no_grad():
            tool_distance, object_distance = mutual_unsigned_mesh_distance_env_frame(
                tool_query_points_E=tool_points_env.unsqueeze(1),
                object_query_points_E=object_points_env.unsqueeze(1),
                object_mesh_vertices=object_mesh_vertices,
                object_mesh_faces=object_mesh_faces,
                tool_mesh_vertices=tool_mesh_vertices,
                tool_mesh_faces=tool_mesh_faces,
                object_rotation_E=object_rotation_E,
                object_bbox_center_E=object_bbox_center_E,
                tool_rotation_E_k=tool_rotation_E_k[:, :1],
                tool_translation_E_k=tool_translation_E_k[:, :1],
                chunk_size=self.label_chunk_size,
                backend=self.label_backend,
            )
            tool_distance = tool_distance[:, 0]
            object_distance = object_distance[:, 0]

        paired = self.encoder.encode(
            encoder_tool_points,
            object_points,
            tool_unsigned_distance=tool_distance,
            obj_unsigned_distance=object_distance,
        )
        tool_tokens = paired.fused_tokens[:, : self.num_patches]
        object_tokens = paired.fused_tokens[:, self.num_patches :]
        tool_labels = self._patch_labels(
            (tool_distance <= self.contact_eps).to(tool_points.dtype),
            paired.tool_patch_idx,
        )
        object_labels = self._patch_labels(
            (object_distance <= self.contact_eps).to(object_points.dtype),
            paired.obj_patch_idx,
        )
        tool_logits = self.tool_contact_head(tool_tokens).squeeze(-1)
        object_logits = self.obj_contact_head(object_tokens).squeeze(-1)
        tool_loss, tool_stats = self._balanced_bce(tool_logits, tool_labels)
        object_loss, object_stats = self._balanced_bce(object_logits, object_labels)
        loss = tool_loss + object_loss

        with torch.no_grad():
            logits = torch.cat((tool_logits, object_logits), dim=1)
            labels = torch.cat((tool_labels, object_labels), dim=1) >= 0.5
            prediction = logits.sigmoid() >= 0.5
            true_positive = (prediction & labels).sum().float()
            false_positive = (prediction & ~labels).sum().float()
            false_negative = (~prediction & labels).sum().float()
            accuracy = (prediction == labels).float().mean()
            precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
            recall = true_positive / (true_positive + false_negative).clamp_min(1.0)

        return loss, {
            "total_loss": float(loss.detach().cpu()),
            "contact_loss": float(loss.detach().cpu()),
            "bce_A": float(tool_loss.detach().cpu()),
            "bce_B": float(object_loss.detach().cpu()),
            "patch_pos_frac_A": float(tool_labels.mean().detach().cpu()),
            "patch_pos_frac_B": float(object_labels.mean().detach().cpu()),
            "empty_positive_patch_count": float(
                tool_stats["empty_pos"] + object_stats["empty_pos"]
            ),
            "contact_acc": float(accuracy.detach().cpu()),
            "contact_precision": float(precision.detach().cpu()),
            "contact_recall": float(recall.detach().cpu()),
        }
