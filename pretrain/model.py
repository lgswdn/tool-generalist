"""model.py — SDF segmentation wrapper for pretraining the joint-ViT encoder.

This module wraps SDFPointCloudEncoder (from rsl_rl.modules.models.cloud.sdf_encoder)
with lightweight SDF prediction heads and a Huber-loss training objective.

The encoder itself lives in rsl_rl so that the RL policy can import it directly
once pretraining is done — exactly the same pattern as ICPNet.

Architecture
────────────
  SDFSegmentor
    ├── encoder: SDFPointCloudEncoder  (shared ViT backbone)
    ├── tool_head: MLP  (tool-point / tool-patch → SDF scalar)
    └── obj_head:  MLP  (obj-point  / obj-patch  → SDF scalar)

Head modes
──────────
  "point"  — per-point SDF via unpatchify (scatter patch token → points) + MLP
  "patch"  — per-patch SDF directly from patch tokens + global token
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── repo path ────────────────────────────────────────────────────────────────
_PRETRAIN_DIR = Path(__file__).resolve().parent
_REPO_ROOT    = _PRETRAIN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rsl_rl.modules.models.cloud.sdf_encoder import (
    SDFPointCloudEncoder,
    SDFEncoderCfg,
)


# --------------------------------------------------------------------------- #
# Small MLP helper (self-contained, no rsl_rl.common dependency)
# --------------------------------------------------------------------------- #

def _make_mlp(dims: tuple[int, ...]) -> nn.Sequential:
    """Linear → LayerNorm → ELU stack; last layer has no activation."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# GT SDF aggregation for patch mode
# --------------------------------------------------------------------------- #

def _aggregate_sdf(
    sdf_pts:   torch.Tensor,   # (B, N)
    patch_idx: torch.Tensor,   # (B, P, K)
    mode:      str = "mean",
) -> torch.Tensor:             # (B, P)
    B, P, K = patch_idx.shape
    gathered = sdf_pts.gather(1, patch_idx.reshape(B, P * K)).view(B, P, K)
    if mode == "min":  return gathered.min(-1).values
    if mode == "max":  return gathered.max(-1).values
    return gathered.mean(-1)


# --------------------------------------------------------------------------- #
# SDFSegmentor
# --------------------------------------------------------------------------- #

class SDFSegmentor(nn.Module):
    """SDF prediction wrapper around SDFPointCloudEncoder.

    Args:
        head_mode:       "point" | "patch"
        patch_agg:       GT aggregation for patch mode ("mean"/"min"/"max")
        num_pts:         Points per input cloud (N).
        patch_size:      Points per FPS patch (K).
        encoder_channel: Token dimension D.
        vit_depth:       ViT transformer depth.
        vit_heads:       ViT attention heads.
        head_hidden:     Hidden dims for SDF MLP heads.
        freeze_encoder:  If True, freeze the encoder after init.
        icp_weights_path: (ignored, kept for CLI back-compat)
        freeze_icp:       (alias for freeze_encoder, back-compat)
    """

    def __init__(
        self,
        head_mode:       str = "point",
        patch_agg:       str = "mean",
        num_pts:         int = 512,
        patch_size:      int = 32,
        encoder_channel: int = 128,
        vit_depth:       int = 4,
        vit_heads:       int = 4,
        head_hidden:     tuple[int, ...] = (256, 128),
        freeze_encoder:  bool = False,
        # Legacy / back-compat
        icp_weights_path: Optional[str] = None,
        freeze_icp:       bool = False,
    ):
        super().__init__()
        assert head_mode in ("point", "patch")
        assert patch_agg  in ("mean", "min", "max")
        self.head_mode = head_mode
        self.patch_agg = patch_agg
        self.num_pts   = num_pts

        if icp_weights_path:
            print("[SDFSegmentor] NOTE: icp_weights_path is ignored; "
                  "encoder is SDFPointCloudEncoder (not ICPNet).")

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_cfg = SDFEncoderCfg(
            num_pts=num_pts,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            freeze=freeze_encoder or freeze_icp,
        )
        self.encoder = SDFPointCloudEncoder(enc_cfg)
        D = self.encoder.feature_dim
        P = self.encoder.num_patches

        # ── SDF heads ─────────────────────────────────────────────────────────
        if head_mode == "point":
            # Input: per-point xyz embed (D) + scattered patch token (D) + CLS (D) = 3D
            self.xyz_embed = _make_mlp((3, D, D))
            self.tool_head = _make_mlp((3 * D,) + head_hidden + (1,))
            self.obj_head  = _make_mlp((3 * D,) + head_hidden + (1,))
        else:
            # Input: patch token (D) + CLS (D) = 2D
            self.tool_head = _make_mlp((2 * D,) + head_hidden + (1,))
            self.obj_head  = _make_mlp((2 * D,) + head_hidden + (1,))

    # ── Point-level prediction ───────────────────────────────────────────────

    def _predict_point(
        self,
        pc:          torch.Tensor,    # (B, N, 3)
        patch_tokens: torch.Tensor,   # (B, P, D)
        global_feat:  torch.Tensor,   # (B, D)
        patch_idx:    torch.Tensor,   # (B, P, K) — into this stream's N points
        head:         nn.Module,
    ) -> torch.Tensor:                # (B, N)
        B, N, _ = pc.shape
        P, K    = patch_tokens.shape[1], patch_idx.shape[2]
        D       = self.encoder.feature_dim

        # Lift XYZ
        pt_xyz = self.xyz_embed(pc)                            # (B, N, D)

        # Scatter patch tokens → per-point features (last-write for overlaps)
        pt_patch = torch.zeros(B, N, D, device=pc.device, dtype=pc.dtype)
        exp_tok  = patch_tokens.unsqueeze(2).expand(B, P, K, D)   # (B, P, K, D)
        flat_idx = patch_idx.reshape(B, P * K, 1).expand(B, P * K, D)
        flat_tok = exp_tok.reshape(B, P * K, D)
        pt_patch.scatter_(1, flat_idx, flat_tok)               # (B, N, D)

        # Broadcast global feat
        pt_global = global_feat.unsqueeze(1).expand(B, N, D)   # (B, N, D)

        feat = torch.cat([pt_xyz, pt_patch, pt_global], dim=-1)  # (B, N, 3D)
        return head(feat).squeeze(-1)                             # (B, N)

    # ── Patch-level prediction ───────────────────────────────────────────────

    def _predict_patch(
        self,
        patch_tokens: torch.Tensor,   # (B, P, D)
        global_feat:  torch.Tensor,   # (B, D)
        head:         nn.Module,
    ) -> torch.Tensor:                # (B, P)
        B, P, D = patch_tokens.shape
        pt_global = global_feat.unsqueeze(1).expand(B, P, D)      # (B, P, D)
        feat = torch.cat([patch_tokens, pt_global], dim=-1)        # (B, P, 2D)
        return head(feat).squeeze(-1)                               # (B, P)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        tool_pc: torch.Tensor,   # (B, N, 3)
        obj_pc:  torch.Tensor,   # (B, N, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tool_sdf : (B, N) in point mode  |  (B, P) in patch mode
            obj_sdf  : (B, N) in point mode  |  (B, P) in patch mode
        """
        res = self.encoder.encode(tool_pc, obj_pc)

        if self.head_mode == "point":
            tool_sdf = self._predict_point(
                tool_pc, res.tool_tokens, res.global_feat,
                res.tool_patch_idx, self.tool_head,
            )
            obj_sdf = self._predict_point(
                obj_pc, res.obj_tokens, res.global_feat,
                res.obj_patch_idx, self.obj_head,
            )
        else:
            tool_sdf = self._predict_patch(res.tool_tokens, res.global_feat, self.tool_head)
            obj_sdf  = self._predict_patch(res.obj_tokens,  res.global_feat, self.obj_head)

        return tool_sdf, obj_sdf

    # ── Loss ────────────────────────────────────────────────────────────────

    def loss(
        self,
        tool_pc:     torch.Tensor,
        obj_pc:      torch.Tensor,
        tool_sdf_gt: torch.Tensor,
        obj_sdf_gt:  torch.Tensor,
        sdf_weight:  float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Huber (smooth-L1) loss on predicted vs. GT SDF for both streams."""
        # Forward pass (also gives us patch indices for GT agg in patch mode)
        res = self.encoder.encode(tool_pc, obj_pc)

        if self.head_mode == "point":
            tool_pred = self._predict_point(
                tool_pc, res.tool_tokens, res.global_feat,
                res.tool_patch_idx, self.tool_head,
            )
            obj_pred = self._predict_point(
                obj_pc, res.obj_tokens, res.global_feat,
                res.obj_patch_idx, self.obj_head,
            )
        else:
            # Aggregate GT to patch level
            tool_sdf_gt = _aggregate_sdf(tool_sdf_gt, res.tool_patch_idx, self.patch_agg)
            obj_sdf_gt  = _aggregate_sdf(obj_sdf_gt,  res.obj_patch_idx,  self.patch_agg)
            tool_pred = self._predict_patch(res.tool_tokens, res.global_feat, self.tool_head)
            obj_pred  = self._predict_patch(res.obj_tokens,  res.global_feat, self.obj_head)

        tool_loss = F.smooth_l1_loss(tool_pred, tool_sdf_gt)
        obj_loss  = F.smooth_l1_loss(obj_pred,  obj_sdf_gt)
        total     = sdf_weight * (tool_loss + obj_loss)

        return total, {
            "tool_sdf_loss": tool_loss.item(),
            "obj_sdf_loss":  obj_loss.item(),
            "total":         total.item(),
        }
