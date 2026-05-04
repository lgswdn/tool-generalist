"""model.py — RPDiff-style ContactDiffusionModel: joint SDF + pose denoising.

Architecture:
  encoder          : SDFPointCloudEncoder (shared ViT, from existing pipeline)
  pose_cross_attn  : PoseCrossAttention (injects noised pose into encoder tokens)
  sdf_head         : Pose-conditioned SDF prediction (point or patch mode)
  denoising_head   : RPDiff-style MLP heads for translation + rotation

Reuses:
  - rpdiff.utils.torch_util.SinusoidalPosEmb for timestep embedding
  - rpdiff.utils.torch3d_util.matrix_to_quaternion for rotation conversion
  - rpdiff.training.losses.TransformChamferWrapper for denoising loss
  - Existing SDFPointCloudEncoder for geometry encoding
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
_NEW_PRETRAIN_DIR = Path(__file__).resolve().parent
_PRETRAIN_DIR     = _NEW_PRETRAIN_DIR.parent
_REPO_ROOT        = _PRETRAIN_DIR.parent
_RPDIFF_SRC       = _PRETRAIN_DIR / "rpdiff" / "src"

# NOTE: do NOT add _PRETRAIN_DIR to sys.path — it has its own config.py,
# model.py, dataset.py that would shadow our new_pretrain versions.
for p in [str(_REPO_ROOT), str(_RPDIFF_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Reuse from existing pipeline ─────────────────────────────────────────────
from rsl_rl.modules.models.cloud.sdf_encoder import (
    SDFPointCloudEncoder,
    SDFEncoderCfg,
)

# ── Reuse from RPDiff ────────────────────────────────────────────────────────
from rpdiff.utils.torch_util import SinusoidalPosEmb
from rpdiff.utils.torch3d_util import matrix_to_quaternion
from rpdiff.training.losses import TransformChamferWrapper


# --------------------------------------------------------------------------- #
# Small helpers (inlined from pretrain/model.py to avoid module name collision)
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


def _aggregate_sdf(
    sdf_pts:   torch.Tensor,
    patch_idx: torch.Tensor,
    mode:      str = "mean",
) -> torch.Tensor:
    B, P, K = patch_idx.shape
    gathered = sdf_pts.gather(1, patch_idx.reshape(B, P * K)).view(B, P, K)
    if mode == "min":  return gathered.min(-1).values
    if mode == "max":  return gathered.max(-1).values
    return gathered.mean(-1)


def _split_tokens(res, P: int):
    """Split fused_tokens (B, 2P, D) into tool_tokens and obj_tokens."""
    return res.fused_tokens[:, :P, :], res.fused_tokens[:, P:, :]


# ============================================================================ #
# PoseCrossAttention — inject noised pose into encoder tokens
# ============================================================================ #

class PoseCrossAttention(nn.Module):
    """Cross-attention module that conditions encoder tokens on a noised tool pose.

    The noised pose (7D: trans + quaternion) and timestep are projected and used
    as query tokens to attend to the encoder's patch tokens.
    Output is added as a residual to the original tokens.
    """

    def __init__(
        self,
        token_dim: int,
        pose_dim: int = 7,
        n_heads: int = 4,
        n_layers: int = 2,
        max_timestep: int = 100,
    ):
        super().__init__()
        self.token_dim = token_dim

        # Project pose (7D) to token dimension
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # Timestep embedding (reuse RPDiff's SinusoidalPosEmb)
        self.time_emb = SinusoidalPosEmb(dim=token_dim, max_pos=max_timestep)
        self.time_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )

        # Cross-attention layers: pose queries attend to encoder tokens
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=token_dim,
                    num_heads=n_heads,
                    batch_first=True,
                ),
                "norm1": nn.LayerNorm(token_dim),
                "norm2": nn.LayerNorm(token_dim),
                "ff": nn.Sequential(
                    nn.Linear(token_dim, token_dim * 4),
                    nn.GELU(),
                    nn.Linear(token_dim * 4, token_dim),
                ),
            }))

    def forward(
        self,
        tokens: torch.Tensor,       # (B, 2P, D) encoder patch tokens
        pose_7d: torch.Tensor,       # (B, 7) noised pose: trans(3) + quat(4)
        timestep: torch.Tensor,      # (B,) int timestep index
    ) -> torch.Tensor:
        """Returns pose-conditioned tokens P' with same shape (B, 2P, D)."""
        B = tokens.shape[0]

        # Project pose to embedding
        pose_emb = self.pose_proj(pose_7d)              # (B, D)

        # Timestep embedding
        time_emb = self.time_emb(timestep.float())      # (B, D)
        time_emb = self.time_proj(time_emb)              # (B, D)

        # Combine pose + time as a single conditioning token
        cond = (pose_emb + time_emb).unsqueeze(1)        # (B, 1, D)

        # Cross-attention: tokens attend to pose condition
        out = tokens
        for layer in self.layers:
            # Cross-attention: queries=tokens, keys/values=cond
            residual = out
            out_norm = layer["norm1"](out)
            attn_out, _ = layer["cross_attn"](
                query=out_norm,
                key=cond,
                value=cond,
            )
            out = residual + attn_out

            # Feedforward
            residual = out
            out = residual + layer["ff"](layer["norm2"](out))

        return out  # (B, 2P, D)


# ============================================================================ #
# DenoisingHead — RPDiff-style MLP heads for translation + rotation
# ============================================================================ #

class DenoisingHead(nn.Module):
    """RPDiff-style denoising head: predicts one-step inverse transform.

    Follows RPDiff's policy_feat_encoder.py exactly:
      out_trans: Linear → ReLU → Linear → 3D translation
      out_vec1:  Linear → ReLU → Linear → 3D (first rotation vector)
      out_vec2:  Linear → ReLU → Linear → 3D (second rotation vector)

    Gram-Schmidt orthogonalization on (vec1, vec2) → rotation matrix → quaternion.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Exactly RPDiff's output heads (2-layer MLPs)
        self.out_trans = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3)
        )
        self.out_vec1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3)
        )
        self.out_vec2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3)
        )

    def forward(
        self,
        pooled_features: torch.Tensor,  # (B, 2*D) separately pooled tool+obj tokens, concatenated
    ) -> dict:
        """Predict one-step denoising transform.

        Returns dict with:
            "trans":      (B, 3) predicted translation
            "rot_mat":    (B, 3, 3) predicted rotation (orthogonalized)
            "quat":       (B, 4) predicted quaternion
            "unnorm_quat": (B, 4) un-normalized quaternion (for norm loss)
        """
        pred_trans = self.out_trans(pooled_features)   # (B, 3)
        pred_v1 = self.out_vec1(pooled_features)       # (B, 3)
        pred_v2 = self.out_vec2(pooled_features)       # (B, 3)

        # Gram-Schmidt orthogonalization → rotation matrix
        rot_mat = self._gram_schmidt(pred_v1, pred_v2)  # (B, 3, 3)

        # Convert to quaternion (using RPDiff's matrix_to_quaternion)
        # matrix_to_quaternion expects (B, 3, 4) or (B, 3, 3)
        quat = matrix_to_quaternion(rot_mat)            # (B, 4)
        unnorm_quat = quat.clone()
        quat = F.normalize(quat, dim=-1)

        return {
            "trans": pred_trans,
            "rot_mat": rot_mat,
            "quat": quat,
            "unnorm_quat": unnorm_quat,
        }

    @staticmethod
    def _gram_schmidt(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt orthogonalization: (v1, v2) → rotation matrix.

        Args:
            v1: (B, 3) first vector
            v2: (B, 3) second vector

        Returns:
            (B, 3, 3) rotation matrix
        """
        u1 = F.normalize(v1, dim=-1)
        u2 = v2 - (u1 * v2).sum(dim=-1, keepdim=True) * u1
        u2 = F.normalize(u2, dim=-1)
        u3 = torch.cross(u1, u2, dim=-1)
        return torch.stack([u1, u2, u3], dim=-1)  # (B, 3, 3) columns


# ============================================================================ #
# ContactDiffusionModel — ties everything together
# ============================================================================ #

class ContactDiffusionModel(nn.Module):
    """RPDiff-style joint SDF + pose denoising model.

    Architecture:
      1. encoder: SDFPointCloudEncoder — encodes canonical tool + object → tokens P
      2. pose_cross_attn: PoseCrossAttention — conditions P on noised pose → P'
      3. SDF heads: predict tool/obj SDF from P' (point or patch mode)
      4. denoising_head: RPDiff-style MLP → one-step inverse transform

    The encoder ALWAYS sees canonical tool (origin, R=I) + object (world pose).
    Pose variation enters ONLY via cross-attention.
    """

    def __init__(
        self,
        # SDF head
        head_mode: str = "point",
        patch_agg: str = "mean",
        head_hidden: tuple[int, ...] = (256, 128),
        # Encoder
        num_pts: int = 512,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 4,
        vit_heads: int = 4,
        freeze_encoder: bool = False,
        # Cross-attention
        cross_attn_heads: int = 4,
        cross_attn_layers: int = 2,
        pose_dim: int = 7,
        # Denoising
        denoise_hidden: int = 256,
        # Loss
        sdf_weight: float = 1.0,
        denoise_weight: float = 1.0,
        chamfer_weight: float = 1.0,
        quat_norm_beta: float = 0.1,
        # Diffusion
        num_diffusion_steps: int = 10,
        # Task
        task: str = "sdf-diff",
    ):
        super().__init__()
        assert head_mode in ("point", "patch")
        assert task in ("sdf", "sdf-diff")

        self.head_mode = head_mode
        self.patch_agg = patch_agg
        self.task = task
        self.sdf_weight = sdf_weight
        self.denoise_weight = denoise_weight
        self.chamfer_weight = chamfer_weight
        self.quat_norm_beta = quat_norm_beta
        self.num_diffusion_steps = num_diffusion_steps

        # ── Shared encoder ───────────────────────────────────────────────
        enc_cfg = SDFEncoderCfg(
            num_pts=num_pts,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            freeze=freeze_encoder,
        )
        self.encoder = SDFPointCloudEncoder(enc_cfg)
        D = self.encoder.feature_dim
        self.num_patches = self.encoder.num_patches

        # ── SDF heads (same architecture as existing SDFSegmentor) ───────
        if head_mode == "point":
            self.xyz_embed = _make_mlp((3, D, D))
            self.tool_sdf_head = _make_mlp((2 * D,) + head_hidden + (1,))
            self.obj_sdf_head = _make_mlp((2 * D,) + head_hidden + (1,))
        else:
            self.tool_sdf_head = _make_mlp((D,) + head_hidden + (1,))
            self.obj_sdf_head = _make_mlp((D,) + head_hidden + (1,))

        # ── Pose conditioning (cross-attention) ──────────────────────────
        if task == "sdf-diff":
            self.pose_cross_attn = PoseCrossAttention(
                token_dim=D,
                pose_dim=pose_dim,
                n_heads=cross_attn_heads,
                n_layers=cross_attn_layers,
                max_timestep=num_diffusion_steps + 1,
            )

            # ── Denoising head (RPDiff-style) ────────────────────────────
            # Input: separately pooled tool tokens + object tokens concatenated (2*D)
            self.denoising_head = DenoisingHead(
                input_dim=2 * D,
                hidden_dim=denoise_hidden,
            )

            # ── RPDiff loss function (reuse directly) ────────────────────
            self.denoise_loss_fn = TransformChamferWrapper(l1=False).tf_chamfer

    # ── SDF prediction (point mode) ──────────────────────────────────────

    def _predict_point_sdf(
        self,
        pc: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_idx: torch.Tensor,
        patch_centers: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        """Per-point SDF prediction. Same logic as SDFSegmentor._predict_point."""
        B, N, _ = pc.shape
        D = self.encoder.feature_dim
        P, K = patch_tokens.shape[1], patch_idx.shape[2]

        # Scatter patch centers to per-point
        pt_centers = torch.zeros(B, N, 3, device=pc.device, dtype=pc.dtype)
        exp_ctr = patch_centers.unsqueeze(2).expand(B, P, K, 3)
        flat_ctr_idx = patch_idx.reshape(B, P * K, 1).expand(B, P * K, 3)
        flat_ctr = exp_ctr.reshape(B, P * K, 3)
        pt_centers.scatter_(1, flat_ctr_idx, flat_ctr)

        # Relative coordinate embedding
        rel_xyz = pc - pt_centers
        pt_rel_xyz = self.xyz_embed(rel_xyz)

        # Scatter patch tokens to per-point
        pt_patch = torch.zeros(B, N, D, device=pc.device, dtype=pc.dtype)
        exp_tok = patch_tokens.unsqueeze(2).expand(B, P, K, D)
        flat_idx = patch_idx.reshape(B, P * K, 1).expand(B, P * K, D)
        flat_tok = exp_tok.reshape(B, P * K, D)
        pt_patch.scatter_(1, flat_idx, flat_tok)

        feat = torch.cat([pt_rel_xyz, pt_patch], dim=-1)
        return head(feat).squeeze(-1)

    def _predict_patch_sdf(
        self,
        patch_tokens: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        """Per-patch SDF prediction. Same logic as SDFSegmentor._predict_patch."""
        return head(patch_tokens).squeeze(-1)

    # ── Forward (routes to loss for DDP) ──────────────────────────────────

    def forward(self, *args, **kwargs):
        """Route through loss() so DDP gradient sync hooks fire."""
        return self.loss(*args, **kwargs)

    # ── Joint loss computation ────────────────────────────────────────────

    def loss(
        self,
        tool_canonical: torch.Tensor,     # (B, P, 3) canonical tool (origin, R=I)
        obj_pc: torch.Tensor,             # (B, Q, 3) object (world frame)
        tool_sdf_gt: torch.Tensor,        # (B, P) signed SDF at current pose
        obj_sdf_gt: torch.Tensor,         # (B, Q) signed SDF at current pose
        # Diffusion inputs (only used when task="sdf-diff")
        noised_pose_7d: torch.Tensor = None,    # (B, 7) noised pose: trans(3) + quat(4)
        timestep: torch.Tensor = None,          # (B,) int
        target_trans: torch.Tensor = None,      # (B, 3) denoising target: translation
        target_rot_mat: torch.Tensor = None,    # (B, 3, 3) denoising target: rotation
        # For chamfer loss: the noised child point cloud
        child_start_pcd: torch.Tensor = None,   # (B, Q_child, 3)
        child_final_pcd: torch.Tensor = None,   # (B, Q_child, 3)
        # Encoder result (optional, to share across steps)
        encoder_result=None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute joint SDF + denoising loss.

        The encoder ALWAYS sees canonical tool + object, producing tokens P.
        For sdf-diff, cross-attention with noised pose → P', then SDF from P'.
        For sdf-only, SDF directly from P (no cross-attention).
        """
        metrics = {}

        # ── 1. Encode (canonical tool + object) ─────────────────────────
        if encoder_result is None:
            encoder_result = self.encoder.encode(tool_canonical, obj_pc)
        P = self.num_patches
        tool_tok, obj_tok = _split_tokens(encoder_result, P)

        # ── 2. Optionally apply pose cross-attention ─────────────────────
        if self.task == "sdf-diff" and noised_pose_7d is not None:
            fused = encoder_result.fused_tokens  # (B, 2P, D)
            fused_conditioned = self.pose_cross_attn(fused, noised_pose_7d, timestep)
            tool_tok_cond = fused_conditioned[:, :P, :]
            obj_tok_cond = fused_conditioned[:, P:, :]
        else:
            tool_tok_cond = tool_tok
            obj_tok_cond = obj_tok

        # ── 3. SDF loss ──────────────────────────────────────────────────
        if self.head_mode == "point":
            tool_sdf_pred = self._predict_point_sdf(
                tool_canonical, tool_tok_cond,
                encoder_result.tool_patch_idx,
                encoder_result.tool_patch_centers,
                self.tool_sdf_head,
            )
            obj_sdf_pred = self._predict_point_sdf(
                obj_pc, obj_tok_cond,
                encoder_result.obj_patch_idx,
                encoder_result.obj_patch_centers,
                self.obj_sdf_head,
            )
        else:
            tool_sdf_gt = _aggregate_sdf(
                tool_sdf_gt, encoder_result.tool_patch_idx, self.patch_agg
            )
            obj_sdf_gt = _aggregate_sdf(
                obj_sdf_gt, encoder_result.obj_patch_idx, self.patch_agg
            )
            tool_sdf_pred = self._predict_patch_sdf(tool_tok_cond, self.tool_sdf_head)
            obj_sdf_pred = self._predict_patch_sdf(obj_tok_cond, self.obj_sdf_head)

        tool_sdf_loss = F.smooth_l1_loss(tool_sdf_pred, tool_sdf_gt)
        obj_sdf_loss = F.smooth_l1_loss(obj_sdf_pred, obj_sdf_gt)
        sdf_loss = self.sdf_weight * (tool_sdf_loss + obj_sdf_loss)
        metrics["tool_sdf_loss"] = tool_sdf_loss.item()
        metrics["obj_sdf_loss"] = obj_sdf_loss.item()
        metrics["sdf_loss"] = sdf_loss.item()

        total_loss = sdf_loss

        # ── 4. Denoising loss (only for sdf-diff) ────────────────────────
        if (self.task == "sdf-diff"
                and noised_pose_7d is not None
                and target_trans is not None):

            # Separately pool tool and object conditioned tokens → (B, 2*D)
            # Preserves tool vs. object geometry distinction (critical for pose prediction).
            tool_cond = fused_conditioned[:, :P, :]   # (B, P, D)
            obj_cond  = fused_conditioned[:, P:, :]   # (B, P, D)
            pooled = torch.cat([
                tool_cond.mean(dim=1),   # (B, D)
                obj_cond.mean(dim=1),    # (B, D)
            ], dim=-1)                   # (B, 2*D)

            # Predict one-step denoising
            denoise_out = self.denoising_head(pooled)

            # Build child_pcd_final_pred for chamfer loss
            # Apply predicted (R, t) to child_start_pcd
            if child_start_pcd is not None:
                child_start_mean = child_start_pcd.mean(dim=1, keepdim=True)
                child_start_centered = child_start_pcd - child_start_mean
                child_rot = torch.bmm(
                    denoise_out["rot_mat"],
                    child_start_centered.transpose(1, 2)
                ).transpose(1, 2)
                child_pcd_final_pred = child_rot + child_start_mean + denoise_out["trans"].unsqueeze(1)
                denoise_out["child_pcd_final_pred"] = child_pcd_final_pred

            # Ground truth dict (RPDiff format)
            gt_dict = {
                "trans": target_trans,
                "rot_mat": target_rot_mat,
            }
            if child_final_pcd is not None:
                gt_dict["child_final_pcd"] = child_final_pcd

            # Use RPDiff's loss function directly
            denoise_loss_dict = self.denoise_loss_fn(
                denoise_out, gt_dict,
                quat_norm_beta=self.quat_norm_beta,
            )

            d_loss = denoise_loss_dict["trans"] + denoise_loss_dict["rot"]
            if "chamf" in denoise_loss_dict:
                d_loss = d_loss + self.chamfer_weight * denoise_loss_dict["chamf"]

            total_loss = total_loss + self.denoise_weight * d_loss

            metrics["denoise_trans_loss"] = denoise_loss_dict["trans"].item()
            metrics["denoise_rot_loss"] = denoise_loss_dict["rot"].item()
            if "chamf" in denoise_loss_dict:
                metrics["denoise_chamf_loss"] = denoise_loss_dict["chamf"].item()
            metrics["denoise_loss"] = d_loss.item()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
