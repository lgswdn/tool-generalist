"""model.py — Dual-encoder contact prediction model.

Architecture:
  shared ICPNet ──► tool_tokens   (B, P, D)  → mean → tool_global  (B, D)
  shared ICPNet ──► obj_tokens    (B, P, D)  → mean → obj_global   (B, D)
                                   concat → (B, 2D)
                                   MLP   → (B, 15)  reshape → (B, 5, 3)

The ICPNet is used from rsl_rl.modules.models.rl.net.icp with headers=[].
No context (keys={}) — the geometry alone drives the prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow running from the pretrain/ directory directly
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rsl_rl.modules.models.rl.net.icp import ICPNet


# --------------------------------------------------------------------------- #
# Chamfer distance (symmetric, permutation-invariant)
# --------------------------------------------------------------------------- #

def chamfer_distance(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    pred_normals: torch.Tensor | None = None,
    gt_normals: torch.Tensor | None = None,
    normal_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Symmetric Chamfer distance on contact points with optional matched normal loss.

    Each predicted point is matched to its nearest GT point (and vice-versa);
    cosine dissimilarity is then computed on the matched normal pairs.

    Args:
        pred_pts:      (B, M, 3)
        gt_pts:        (B, M, 3)
        pred_normals:  (B, M, 3) optional — predicted normals (need not be unit)
        gt_normals:    (B, M, 3) optional — ground-truth unit normals
        normal_weight: weight on the cosine normal loss
    Returns:
        (total_loss, metrics_dict)
    """
    diff = pred_pts.unsqueeze(2) - gt_pts.unsqueeze(1)  # (B, M, M, 3)
    dist2 = diff.pow(2).sum(-1)                          # (B, M, M)

    # pred → gt
    nn_pred2gt = dist2.argmin(dim=2)                     # (B, M)  index into gt
    d_pred2gt  = dist2.min(dim=2).values.mean(dim=1)    # (B,)

    # gt → pred
    nn_gt2pred = dist2.argmin(dim=1)                     # (B, M)  index into pred
    d_gt2pred  = dist2.min(dim=1).values.mean(dim=1)    # (B,)

    chamfer = (d_pred2gt + d_gt2pred).mean()
    metrics = {"chamfer": chamfer.item()}
    total = chamfer

    if pred_normals is not None and gt_normals is not None:
        # Normalize predictions to unit sphere
        pred_n = torch.nn.functional.normalize(pred_normals, dim=-1)  # (B, M, 3)
        gt_n   = gt_normals                                            # already unit

        # pred → gt: for each pred point, gather its matched GT normal
        idx = nn_pred2gt.unsqueeze(-1).expand(-1, -1, 3)   # (B, M, 3)
        gt_n_matched = gt_n.gather(1, idx)                 # (B, M, 3)
        cos_pred2gt  = (1.0 - (pred_n * gt_n_matched).sum(-1)).mean()  # (B,)

        # gt → pred: for each gt point, gather its matched pred normal
        idx2 = nn_gt2pred.unsqueeze(-1).expand(-1, -1, 3)  # (B, M, 3)
        pred_n_matched = pred_n.gather(1, idx2)            # (B, M, 3)
        cos_gt2pred    = (1.0 - (pred_n_matched * gt_n).sum(-1)).mean()

        normal_loss = (cos_pred2gt + cos_gt2pred) * 0.5
        metrics["normal_cos"] = normal_loss.item()
        total = chamfer + normal_weight * normal_loss

    metrics["total"] = total.item()
    return total, metrics


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class ContactPredictor(nn.Module):
    """Predicts 5 contact points (in world/object frame) from posed clouds.

    Args:
        num_contact_pts: Number of contact points to predict (default 5).
        icp_weights_path: Optional path to a pretrained ICPNet checkpoint.
        freeze_icp: Whether to freeze the ICP encoder during training.
        num_pts: Number of points in each input cloud (must match ICPNet config).
        patch_size: Patch size for ICPNet grouping.
        encoder_channel: Hidden dimension of the ICPNet transformer.
        head_hidden: Hidden dims of the MLP decoder head.
    """

    def __init__(
        self,
        num_contact_pts: int = 5,
        icp_weights_path: str | None = None,
        freeze_icp: bool = False,
        num_pts: int = 512,
        patch_size: int = 32,
        encoder_channel: int = 128,
        head_hidden: tuple[int, ...] = (256, 128),
    ):
        super().__init__()
        self.num_contact_pts = num_contact_pts

        # Shared ICP encoder (no headers, no context)
        cfg = ICPNet.Config(
            dim_in=(num_pts, 3),
            dim_out=encoder_channel,
            keys={},
            headers=[],
            num_query=1,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            pos_embed_type="mlp",
            group_type="fps",
            patch_type="mlp",
            patch_overlap=1.0,
            p_drop=0.0,
            freeze_encoder=False,
            use_adapter=False,
            adapter_dim=64,
            tune_last_layer=False,
            late_late_fusion=False,
            output_attn=False,
            output_hidden=False,
            activate_header=False,
            pre_ln_bias=True,
            ignore_zero=False,
            use_vq=False,
            train_last_ln=True,
            header_inputs=None,
            use_v2_module=False,
        )
        cfg.encoder.num_hidden_layers = 2
        cfg.encoder.layer.hidden_size = encoder_channel
        cfg.encoder.layer.num_attention_heads = 3

        self.icp = ICPNet(cfg)

        if icp_weights_path is not None:
            print(f"[ContactPredictor] Loading ICP weights from {icp_weights_path}")
            self.icp.load(filename=icp_weights_path, verbose=True)

        if freeze_icp:
            for p in self.icp.parameters():
                p.requires_grad_(False)
            self.icp.eval()

        # MLP decoder: concat(tool_global, obj_global) → 5×3
        feat_dim = encoder_channel * 2  # two mean-pooled globals
        layers: list[nn.Module] = []
        in_dim = feat_dim
        for h in head_hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_contact_pts * 6))  # pts(3) + normals(3)
        self.head = nn.Sequential(*layers)

    # ----------------------------------------------------------------------- #

    def _encode(self, pc: torch.Tensor) -> torch.Tensor:
        """Encode a point cloud → global feature.

        Args:
            pc: (B, N, 3)
        Returns:
            (B, encoder_channel)  — mean-pooled patch tokens
        """
        _, tokens = self.icp(pc, ctx={})   # tokens: (B, P, D)
        return tokens.mean(dim=1)          # (B, D)

    def forward(
        self,
        tool_pc: torch.Tensor,
        object_pc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tool_pc:   (B, N, 3) tool points in world frame
            object_pc: (B, N, 3) object points in world frame
        Returns:
            contact_pts:     (B, num_contact_pts, 3)
            contact_normals: (B, num_contact_pts, 3)  raw (not yet normalised)
        """
        tool_feat = self._encode(tool_pc)    # (B, D)
        obj_feat  = self._encode(object_pc)  # (B, D)
        fused = torch.cat([tool_feat, obj_feat], dim=-1)  # (B, 2D)
        out = self.head(fused)               # (B, num_contact_pts * 6)
        out = out.view(-1, self.num_contact_pts, 6)
        return out[..., :3], out[..., 3:]    # pts, normals

    def loss(
        self,
        tool_pc: torch.Tensor,
        object_pc: torch.Tensor,
        contact_pts_gt: torch.Tensor,
        contact_normals_gt: torch.Tensor,
        normal_weight: float = 0.5,
    ) -> tuple[torch.Tensor, dict]:
        """Forward + Chamfer + cosine normal loss.

        Returns:
            (total_loss, metrics_dict)
        """
        pred_pts, pred_normals = self.forward(tool_pc, object_pc)
        return chamfer_distance(
            pred_pts, contact_pts_gt,
            pred_normals, contact_normals_gt,
            normal_weight=normal_weight,
        )
