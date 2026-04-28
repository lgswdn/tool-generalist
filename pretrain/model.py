"""model.py — Joint encoder pretraining: SDF prediction + Pose flow matching.

Uses Conditional Flow Matching (Rectified Flow) for pose prediction.
  Training:  x_t = (1-t)·x_0 + t·ε,  predict velocity v = ε - x_0
  Inference: Euler integration from noise (t=1) to data (t=0)

Pretraining tasks and their locality constraints:
  - Pointwise SDF : patch token + relative xyz (point − patch_center)  → per-point SDF
  - Patchwise SDF : patch token only                                    → per-patch SDF
  - Movement pred : patch token + relative xyz, cross-attn with tool Δpose → per-point displacement
  - Flow matching : ALL patch tokens (global conditioning)               → delta pose
  - Aux regression: ALL tokens mean-pooled → init_pose (unchanged)

Architecture:
  JointModel (shared encoder + task heads)
    ├── encoder: SDFPointCloudEncoder  (ViT backbone, FPS patches)
    ├── sdf_head: SDFSegmentor          (local-only: no global_feat)
    ├── movement_head: MovementPredictionHead  (per-point displacement)
    ├── velocity_net: MLPVelocityNet    (MLP, fast for horizon=1)
    └── aux_reg_head: AuxRegressionHead (optional, pooled cond → init_pose)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# No external scheduler needed — flow matching uses simple linear interpolation

# ── repo path ────────────────────────────────────────────────────────────────
_PRETRAIN_DIR = Path(__file__).resolve().parent
_REPO_ROOT    = _PRETRAIN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── diffusion_policy path ────────────────────────────────────────────────────
_DIFFUSION_POLICY_DIR = _PRETRAIN_DIR / "diffusion_policy_repo"
if str(_DIFFUSION_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_DIFFUSION_POLICY_DIR))

from rsl_rl.modules.models.cloud.sdf_encoder import (
    SDFPointCloudEncoder,
    SDFEncoderCfg,
)  # import directly, not via modules/__init__.py (avoid pytorch3d chain)

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion


# --------------------------------------------------------------------------- #
# Small MLP helper
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
    """Split fused_tokens (B, 2P, D) into tool_tokens (B, P, D) and obj_tokens (B, P, D)."""
    return res.fused_tokens[:, :P, :], res.fused_tokens[:, P:, :]


# --------------------------------------------------------------------------- #
# MLPVelocityNet — direct MLP for horizon=1 flow matching
# --------------------------------------------------------------------------- #

class SinTimeEmb(nn.Module):
    """Sinusoidal time embedding for continuous t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) continuous in [0,1] → (B, dim)"""
        half = self.dim // 2
        freqs = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -freqs)
        # Scale t to [0, 1000] range for richer embeddings
        args = (t.float() * 1000.0).unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLPVelocityNet(nn.Module):
    """Direct MLP: [x_t, t, pooled_cond] → velocity.

    Designed for horizon=1 (single 9D pose) flow matching.
    Mean-pools condition tokens, concatenates with interpolated input
    and continuous timestep, then predicts the velocity field.
    """

    def __init__(self, pose_dim: int = 9, cond_dim: int = 128,
                 hidden: int = 256, n_layers: int = 4):
        super().__init__()
        self.time_emb = SinTimeEmb(hidden)
        self.time_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, hidden), nn.GELU())
        self.input_proj = nn.Sequential(nn.Linear(pose_dim, hidden), nn.GELU())

        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Linear(hidden * 3 if i == 0 else hidden, hidden),
                nn.LayerNorm(hidden), nn.GELU(),
            ])
        layers.append(nn.Linear(hidden, pose_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """Args:
            sample:   (B, 1, 9) interpolated x_t
            timestep: (B,) continuous t ∈ [0, 1]
            cond:     (B, N, D) condition tokens
        Returns: (B, 1, 9) predicted velocity
        """
        B = sample.shape[0]
        x = sample.squeeze(1)                                     # (B, 9)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.float)
        t = self.time_proj(self.time_emb(timestep.expand(B)))      # (B, H)
        c = self.cond_proj(cond.mean(dim=1))                       # (B, H)
        h = torch.cat([self.input_proj(x), t, c], dim=-1)          # (B, 3H)
        return self.mlp(h).unsqueeze(1)                            # (B, 1, 9)


# Backward compat alias
MLPNoisePredictor = MLPVelocityNet


# --------------------------------------------------------------------------- #
# AuxRegressionHead — pooled condition → init_pose
# --------------------------------------------------------------------------- #

class AuxRegressionHead(nn.Module):
    """Auxiliary regression: pooled condition → init_pose.

    Predicts the initial pose from encoder features. This is directly
    solvable (the encoder sees tool@init_pose), forcing the encoder to
    produce position-aware features and preventing representation collapse.
    """

    def __init__(self, cond_dim: int = 128, pose_dim: int = 9):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.GELU(),
            nn.Linear(cond_dim, pose_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, N_tokens, D) → (B, pose_dim)"""
        pooled = cond.mean(dim=1)
        return self.head(pooled)


# --------------------------------------------------------------------------- #
# MovementPredictionHead — per-point displacement from object patch features
# --------------------------------------------------------------------------- #

class MovementPredictionHead(nn.Module):
    """Per-point movement prediction conditioned on tool delta pose.

    For each object surface point, predicts its displacement when the tool
    pushes. Uses only the point's OWN patch feature (local) plus the point's
    relative coordinate within that patch, fused with tool delta pose via
    cross-attention.

    Architecture:
        query  = patch_token + rel_xyz_embed(point − patch_center)  (B, N, D)
        kv     = tool_delta_pose_projected                          (B, 1, D)
        fused  = cross_attn(query, kv, kv)                          (B, N, D)
        output = MLP(fused) → displacement                          (B, N, 3)
    """

    def __init__(self, patch_dim: int = 128, pose_dim: int = 9, n_heads: int = 4):
        super().__init__()
        self.patch_dim = patch_dim

        # Embed relative xyz to patch feature space
        self.xyz_embed = _make_mlp((3, patch_dim, patch_dim))

        # Project tool delta pose to key/value space
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, patch_dim),
            nn.LayerNorm(patch_dim),
            nn.GELU(),
        )

        # Cross-attention: query = per-point feature, kv = tool delta pose
        self.cross_attn = nn.MultiheadAttention(
            patch_dim, n_heads, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(patch_dim)

        # Output MLP: fused feature → 3D displacement
        self.head = _make_mlp((patch_dim, patch_dim // 2, 3))

    def forward(
        self,
        pc: torch.Tensor,              # (B, N, 3) — object point cloud
        patch_tokens: torch.Tensor,     # (B, P, D) — object patch tokens
        patch_idx: torch.Tensor,        # (B, P, K) — point indices per patch
        patch_centers: torch.Tensor,    # (B, P, 3) — FPS patch centers
        tool_delta_pose: torch.Tensor,  # (B, 9) — tool delta pose conditioning
    ) -> torch.Tensor:
        """Predict per-point displacement.

        Returns: (B, N, 3) — predicted displacement for each object point.
        """
        B, N, _ = pc.shape
        D = self.patch_dim
        P, K = patch_tokens.shape[1], patch_idx.shape[2]

        # ---- 1. Scatter patch tokens to per-point ----
        pt_patch = torch.zeros(B, N, D, device=pc.device, dtype=pc.dtype)
        exp_tok = patch_tokens.unsqueeze(2).expand(B, P, K, D)
        flat_idx = patch_idx.reshape(B, P * K, 1).expand(B, P * K, D)
        flat_tok = exp_tok.reshape(B, P * K, D)
        pt_patch.scatter_(1, flat_idx, flat_tok)

        # ---- 2. Scatter patch centers to per-point ----
        pt_centers = torch.zeros(B, N, 3, device=pc.device, dtype=pc.dtype)
        exp_ctr = patch_centers.unsqueeze(2).expand(B, P, K, 3)
        flat_ctr_idx = patch_idx.reshape(B, P * K, 1).expand(B, P * K, 3)
        flat_ctr = exp_ctr.reshape(B, P * K, 3)
        pt_centers.scatter_(1, flat_ctr_idx, flat_ctr)

        # ---- 3. Relative coordinate embedding ----
        rel_xyz = pc - pt_centers                    # (B, N, 3)
        pt_rel_embed = self.xyz_embed(rel_xyz)       # (B, N, D)

        # ---- 4. Per-point query = patch_token + rel_xyz_embed ----
        query = pt_patch + pt_rel_embed              # (B, N, D)

        # ---- 5. Tool delta pose as key/value ----
        kv = self.pose_proj(tool_delta_pose)         # (B, D)
        kv = kv.unsqueeze(1)                         # (B, 1, D)

        # ---- 6. Cross-attention (all points attend to single tool pose token) ----
        # query (B, N, D) attends to kv (B, 1, D) — efficient batched attention
        fused, _ = self.cross_attn(query, kv, kv)    # (B, N, D)
        fused = self.cross_norm(query + fused)        # residual + norm

        # ---- 7. Predict displacement ----
        return self.head(fused)                       # (B, N, 3)


# --------------------------------------------------------------------------- #
# SDFSegmentor — local-only SDF prediction (no global_feat)
# --------------------------------------------------------------------------- #

class SDFSegmentor(nn.Module):
    """SDF prediction wrapper around SDFPointCloudEncoder.

    Locality constraints:
      - Pointwise: each point uses ONLY its own patch token + relative coordinate
      - Patchwise: each patch uses ONLY its own patch token
    No global_feat (CLS token) is used in either mode.
    """

    def __init__(
        self,
        head_mode: str = "point",
        patch_agg: str = "mean",
        num_pts: int = 512,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 4,
        vit_heads: int = 4,
        head_hidden: tuple[int, ...] = (256, 128),
        freeze_encoder: bool = False,
    ):
        super().__init__()
        assert head_mode in ("point", "patch")
        assert patch_agg in ("mean", "min", "max")
        self.head_mode = head_mode
        self.patch_agg = patch_agg

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

        if head_mode == "point":
            # Embed relative xyz (point − patch_center) to feature space
            self.xyz_embed = _make_mlp((3, D, D))
            # Input: rel_xyz_embed (D) + patch_token (D) = 2*D
            self.tool_head = _make_mlp((2 * D,) + head_hidden + (1,))
            self.obj_head = _make_mlp((2 * D,) + head_hidden + (1,))
        else:
            # Input: patch_token only (D)
            self.tool_head = _make_mlp((D,) + head_hidden + (1,))
            self.obj_head = _make_mlp((D,) + head_hidden + (1,))

    def _predict_point(
        self,
        pc: torch.Tensor,              # (B, N, 3) — point cloud
        patch_tokens: torch.Tensor,     # (B, P, D) — patch tokens
        patch_idx: torch.Tensor,        # (B, P, K) — point indices per patch
        patch_centers: torch.Tensor,    # (B, P, 3) — FPS patch centers
        head: nn.Module,
    ) -> torch.Tensor:
        """Predict per-point SDF using local patch features only."""
        B, N, _ = pc.shape
        D = self.encoder.feature_dim
        P, K = patch_tokens.shape[1], patch_idx.shape[2]

        # ---- Scatter patch centers to per-point ----
        pt_centers = torch.zeros(B, N, 3, device=pc.device, dtype=pc.dtype)
        exp_ctr = patch_centers.unsqueeze(2).expand(B, P, K, 3)
        flat_ctr_idx = patch_idx.reshape(B, P * K, 1).expand(B, P * K, 3)
        flat_ctr = exp_ctr.reshape(B, P * K, 3)
        pt_centers.scatter_(1, flat_ctr_idx, flat_ctr)

        # ---- Relative coordinate embedding ----
        rel_xyz = pc - pt_centers                    # (B, N, 3)
        pt_rel_xyz = self.xyz_embed(rel_xyz)         # (B, N, D)

        # ---- Scatter patch tokens to per-point ----
        pt_patch = torch.zeros(B, N, D, device=pc.device, dtype=pc.dtype)
        exp_tok = patch_tokens.unsqueeze(2).expand(B, P, K, D)
        flat_idx = patch_idx.reshape(B, P * K, 1).expand(B, P * K, D)
        flat_tok = exp_tok.reshape(B, P * K, D)
        pt_patch.scatter_(1, flat_idx, flat_tok)

        # ---- Concatenate: [rel_xyz_embed, patch_token] ----
        feat = torch.cat([pt_rel_xyz, pt_patch], dim=-1)  # (B, N, 2*D)
        return head(feat).squeeze(-1)

    def _predict_patch(
        self,
        patch_tokens: torch.Tensor,    # (B, P, D) — patch tokens
        head: nn.Module,
    ) -> torch.Tensor:
        """Predict per-patch SDF using patch token only."""
        return head(patch_tokens).squeeze(-1)

    def forward(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        tool_sdf_gt: torch.Tensor = None,
        obj_sdf_gt: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Route through loss() so DDP gradient sync hooks fire."""
        if tool_sdf_gt is not None and obj_sdf_gt is not None:
            return self.loss(tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt, **kwargs)
        # Inference path: just predict SDF values
        res = self.encoder.encode(tool_pc, obj_pc)
        P = self.encoder.num_patches
        tool_tok, obj_tok = _split_tokens(res, P)
        if self.head_mode == "point":
            tool_sdf = self._predict_point(
                tool_pc, tool_tok,
                res.tool_patch_idx, res.tool_patch_centers, self.tool_head,
            )
            obj_sdf = self._predict_point(
                obj_pc, obj_tok,
                res.obj_patch_idx, res.obj_patch_centers, self.obj_head,
            )
        else:
            tool_sdf = self._predict_patch(tool_tok, self.tool_head)
            obj_sdf = self._predict_patch(obj_tok, self.obj_head)
        return tool_sdf, obj_sdf

    def loss(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        tool_sdf_gt: torch.Tensor,
        obj_sdf_gt: torch.Tensor,
        # SDF inputs at INIT pose (optional)
        tool_pc_init: torch.Tensor = None,
        init_tool_sdf_gt: torch.Tensor = None,
        init_obj_sdf_gt: torch.Tensor = None,
        sdf_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute SDF loss at both contact and init poses (if available)."""
        res = self.encoder.encode(tool_pc, obj_pc)
        P = self.encoder.num_patches
        tool_tok, obj_tok = _split_tokens(res, P)

        # ---- SDF at CONTACT pose ----
        if self.head_mode == "point":
            tool_pred = self._predict_point(
                tool_pc, tool_tok,
                res.tool_patch_idx, res.tool_patch_centers, self.tool_head,
            )
            obj_pred = self._predict_point(
                obj_pc, obj_tok,
                res.obj_patch_idx, res.obj_patch_centers, self.obj_head,
            )
        else:
            tool_sdf_gt_agg = _aggregate_sdf(tool_sdf_gt, res.tool_patch_idx, self.patch_agg)
            obj_sdf_gt_agg = _aggregate_sdf(obj_sdf_gt, res.obj_patch_idx, self.patch_agg)
            tool_pred = self._predict_patch(tool_tok, self.tool_head)
            obj_pred = self._predict_patch(obj_tok, self.obj_head)
            tool_sdf_gt = tool_sdf_gt_agg
            obj_sdf_gt = obj_sdf_gt_agg

        tool_loss = F.smooth_l1_loss(tool_pred, tool_sdf_gt)
        obj_loss = F.smooth_l1_loss(obj_pred, obj_sdf_gt)
        total = sdf_weight * (tool_loss + obj_loss)
        metrics = {
            "tool_sdf_loss": tool_loss.item(),
            "obj_sdf_loss": obj_loss.item(),
        }

        # ---- SDF at INIT pose (optional) ----
        has_init_sdf = (
            tool_pc_init is not None
            and init_tool_sdf_gt is not None
            and init_obj_sdf_gt is not None
        )
        if has_init_sdf:
            res_init = self.encoder.encode(tool_pc_init, obj_pc)
            init_tool_tok, init_obj_tok = _split_tokens(res_init, P)

            if self.head_mode == "point":
                init_tool_pred = self._predict_point(
                    tool_pc_init, init_tool_tok,
                    res_init.tool_patch_idx, res_init.tool_patch_centers, self.tool_head,
                )
                init_obj_pred = self._predict_point(
                    obj_pc, init_obj_tok,
                    res_init.obj_patch_idx, res_init.obj_patch_centers, self.obj_head,
                )
            else:
                init_tool_sdf_gt_agg = _aggregate_sdf(init_tool_sdf_gt, res_init.tool_patch_idx, self.patch_agg)
                init_obj_sdf_gt_agg = _aggregate_sdf(init_obj_sdf_gt, res_init.obj_patch_idx, self.patch_agg)
                init_tool_pred = self._predict_patch(init_tool_tok, self.tool_head)
                init_obj_pred = self._predict_patch(init_obj_tok, self.obj_head)
                init_tool_sdf_gt = init_tool_sdf_gt_agg
                init_obj_sdf_gt = init_obj_sdf_gt_agg

            init_tool_loss = F.smooth_l1_loss(init_tool_pred, init_tool_sdf_gt)
            init_obj_loss = F.smooth_l1_loss(init_obj_pred, init_obj_sdf_gt)
            total = total + sdf_weight * (init_tool_loss + init_obj_loss)
            metrics["init_tool_sdf_loss"] = init_tool_loss.item()
            metrics["init_obj_sdf_loss"] = init_obj_loss.item()

        return total, metrics


# --------------------------------------------------------------------------- #
# DiffusionModel - Combined SDF + Flow Matching training
# --------------------------------------------------------------------------- #

class DiffusionModel(nn.Module):
    """Joint training of SDF prediction + pose flow matching (no movement).

    Uses shared encoder with:
        - SDFSegmentor (local-only SDF prediction)
        - Velocity net: MLPVelocityNet (default) or TransformerForDiffusion
        - AuxRegressionHead (optional, for regression warmup)

    The velocity net (flow matching) uses ALL patch tokens as conditioning.
    SDF heads use only local patch features (no global_feat).
    """

    def __init__(
        self,
        # SDF head
        head_mode: str = "point",
        patch_agg: str = "mean",
        head_hidden: tuple[int, ...] = (256, 128),
        # Encoder (shared)
        num_pts: int = 512,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 4,
        vit_heads: int = 4,
        freeze_encoder: bool = False,
        # Flow matching / diffusion head
        n_layer: int = 4,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.0,
        use_mlp_head: bool = True,
        # Auxiliary regression
        aux_reg: bool = True,
        # Loss weights
        sdf_weight: float = 1.0,
        diffusion_weight: float = 1.0,
        aux_weight: float = 1.0,
    ):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.diffusion_weight = diffusion_weight
        self.aux_weight = aux_weight
        self.use_mlp_head = use_mlp_head

        # Shared encoder
        enc_cfg = SDFEncoderCfg(
            num_pts=num_pts,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            freeze=freeze_encoder,
        )
        self.encoder = SDFPointCloudEncoder(enc_cfg)
        D = self.encoder.feature_dim  # 128
        P = self.encoder.num_patches   # 16

        # SDF head (shares encoder — avoid duplicate param registration for DDP)
        self.sdf_head = SDFSegmentor(
            head_mode=head_mode,
            patch_agg=patch_agg,
            head_hidden=head_hidden,
            num_pts=num_pts,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            freeze_encoder=freeze_encoder,
        )
        # Delete the duplicate encoder that SDFSegmentor created internally,
        # and replace with a plain (non-registered) reference to the shared one.
        del self.sdf_head.encoder
        object.__setattr__(self.sdf_head, 'encoder', self.encoder)

        # Velocity network (flow matching)
        if use_mlp_head:
            self.velocity_net = MLPVelocityNet(
                pose_dim=9, cond_dim=D, hidden=n_emb, n_layers=n_layer,
            )
        else:
            self.velocity_net = TransformerForDiffusion(
                input_dim=9,
                output_dim=9,
                horizon=1,
                n_obs_steps=2 * P,
                cond_dim=D,
                n_layer=n_layer,
                n_head=n_head,
                n_emb=n_emb,
                p_drop_emb=p_drop_emb,
                p_drop_attn=p_drop_attn,
                obs_as_cond=True,
                time_as_cond=True,
                n_cond_layers=1,
            )
        # Backward compat aliases
        self.noise_predictor = self.velocity_net
        self.transformer = self.velocity_net

        # Position-aware conditioning: project tool-obj centroids into token space
        self.cond_pos_proj = nn.Linear(6, D)

        # Auxiliary regression head (optional)
        self.aux_reg_head = AuxRegressionHead(cond_dim=D) if aux_reg else None

        # Flow matching inference steps
        self.n_inference_steps = 20

    def build_condition(self, tool_pc_init: torch.Tensor,
                        obj_pc: torch.Tensor) -> torch.Tensor:
        """Encode tool@init + object → condition tokens with position bias."""
        enc_result = self.encoder.encode(tool_pc_init, obj_pc)
        cond = enc_result.fused_tokens  # (B, 2P, D) — already [tool, obj] concatenated

        tool_centroid = tool_pc_init.mean(dim=1)
        obj_centroid = obj_pc.mean(dim=1)
        pos_bias = self.cond_pos_proj(
            torch.cat([tool_centroid, obj_centroid], dim=-1)
        )
        cond = cond + pos_bias.unsqueeze(1)
        return cond

    def forward(self, *args, **kwargs):
        """Route through forward() so DDP gradient sync hooks fire."""
        return self.loss(*args, **kwargs)

    def loss(
        self,
        # SDF inputs (tool at CONTACT pose)
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        tool_sdf_gt: torch.Tensor,
        obj_sdf_gt: torch.Tensor,
        # SDF inputs (tool at INIT pose - optional)
        init_tool_sdf_gt: torch.Tensor = None,
        init_obj_sdf_gt: torch.Tensor = None,
        # Flow matching inputs (tool at INIT pose)
        tool_pc_init: torch.Tensor = None,
        delta_pose_gt: torch.Tensor = None,
        init_pose_gt: torch.Tensor = None,
        # Phase control
        enable_flow: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """Joint SDF + flow matching loss computation."""
        metrics = {}
        total_loss = torch.tensor(0.0, device=tool_pc.device)

        # ---- SDF task at CONTACT pose ----
        enc_result_contact = self.encoder.encode(tool_pc, obj_pc)
        P = self.encoder.num_patches
        contact_tool_tok, contact_obj_tok = _split_tokens(enc_result_contact, P)

        if self.sdf_head.head_mode == "point":
            tool_pred = self.sdf_head._predict_point(
                tool_pc, contact_tool_tok,
                enc_result_contact.tool_patch_idx,
                enc_result_contact.tool_patch_centers,
                self.sdf_head.tool_head,
            )
            obj_pred = self.sdf_head._predict_point(
                obj_pc, contact_obj_tok,
                enc_result_contact.obj_patch_idx,
                enc_result_contact.obj_patch_centers,
                self.sdf_head.obj_head,
            )
        else:
            tool_sdf_gt_agg = _aggregate_sdf(tool_sdf_gt, enc_result_contact.tool_patch_idx, self.sdf_head.patch_agg)
            obj_sdf_gt_agg = _aggregate_sdf(obj_sdf_gt, enc_result_contact.obj_patch_idx, self.sdf_head.patch_agg)
            tool_pred = self.sdf_head._predict_patch(contact_tool_tok, self.sdf_head.tool_head)
            obj_pred = self.sdf_head._predict_patch(contact_obj_tok, self.sdf_head.obj_head)
            tool_sdf_gt = tool_sdf_gt_agg
            obj_sdf_gt = obj_sdf_gt_agg

        tool_loss = F.smooth_l1_loss(tool_pred, tool_sdf_gt)
        obj_loss = F.smooth_l1_loss(obj_pred, obj_sdf_gt)
        sdf_loss = tool_loss + obj_loss
        total_loss = total_loss + self.sdf_weight * sdf_loss
        metrics["tool_sdf_loss"] = tool_loss.item()
        metrics["obj_sdf_loss"] = obj_loss.item()

        # ---- SDF task at INIT pose (optional) ----
        has_init_sdf = init_tool_sdf_gt is not None and init_obj_sdf_gt is not None and tool_pc_init is not None
        if has_init_sdf:
            enc_result_init = self.encoder.encode(tool_pc_init, obj_pc)
            init_tool_tok, init_obj_tok = _split_tokens(enc_result_init, P)

            if self.sdf_head.head_mode == "point":
                init_tool_pred = self.sdf_head._predict_point(
                    tool_pc_init, init_tool_tok,
                    enc_result_init.tool_patch_idx,
                    enc_result_init.tool_patch_centers,
                    self.sdf_head.tool_head,
                )
                init_obj_pred = self.sdf_head._predict_point(
                    obj_pc, init_obj_tok,
                    enc_result_init.obj_patch_idx,
                    enc_result_init.obj_patch_centers,
                    self.sdf_head.obj_head,
                )
            else:
                init_tool_sdf_gt_agg = _aggregate_sdf(init_tool_sdf_gt, enc_result_init.tool_patch_idx, self.sdf_head.patch_agg)
                init_obj_sdf_gt_agg = _aggregate_sdf(init_obj_sdf_gt, enc_result_init.obj_patch_idx, self.sdf_head.patch_agg)
                init_tool_pred = self.sdf_head._predict_patch(init_tool_tok, self.sdf_head.tool_head)
                init_obj_pred = self.sdf_head._predict_patch(init_obj_tok, self.sdf_head.obj_head)
                init_tool_sdf_gt = init_tool_sdf_gt_agg
                init_obj_sdf_gt = init_obj_sdf_gt_agg

            init_tool_loss = F.smooth_l1_loss(init_tool_pred, init_tool_sdf_gt)
            init_obj_loss = F.smooth_l1_loss(init_obj_pred, init_obj_sdf_gt)
            init_sdf_loss = init_tool_loss + init_obj_loss
            total_loss = total_loss + self.sdf_weight * init_sdf_loss
            metrics["init_tool_sdf_loss"] = init_tool_loss.item()
            metrics["init_obj_sdf_loss"] = init_obj_loss.item()

        # ---- Flow matching + auxiliary regression ----
        has_init = tool_pc_init is not None
        has_delta = delta_pose_gt is not None
        has_init_pose = init_pose_gt is not None

        if has_init and (has_delta or has_init_pose):
            B = tool_pc_init.shape[0]
            device = tool_pc_init.device

            cond = self.build_condition(tool_pc_init, obj_pc)

            # Auxiliary regression: predict init_pose (directly solvable)
            # Active during BOTH warmup and joint phases — keeps encoder
            # discriminative and provides position-aware gradients.
            if self.aux_reg_head is not None and has_init_pose:
                ip_pred = self.aux_reg_head(cond)
                aux_loss = F.mse_loss(ip_pred, init_pose_gt)
                total_loss = total_loss + self.aux_weight * aux_loss
                metrics["aux_loss"] = aux_loss.item()

            # Flow matching (disabled during warmup phase)
            if enable_flow and has_delta:
                x_0 = delta_pose_gt.unsqueeze(1)       # (B, 1, 9) data
                eps = torch.randn_like(x_0)             # (B, 1, 9) noise

                # Logit-normal time sampling (SD3 recipe):
                # Concentrates near t=0 and t=1 where signal is clearest,
                # unlike uniform which overweights the hard t≈0.5 region.
                sigma_min = 1e-4
                u = torch.randn(B, device=device) * 0.5  # std=0.5 for mild concentration
                t = torch.sigmoid(u)                       # logit-normal in (0, 1)
                t = t * (1.0 - sigma_min) + sigma_min      # clamp to [σ_min, 1]

                # Linear interpolation: x_t = (1-t)·x_0 + t·ε
                t_expand = t[:, None, None]             # (B, 1, 1)
                x_t = (1.0 - t_expand) * x_0 + t_expand * eps

                # Velocity target: v = ε - x_0
                v_target = eps - x_0

                # Predict velocity
                v_pred = self.velocity_net(
                    sample=x_t, timestep=t, cond=cond,
                )
                # Huber loss: linear for large errors → bounds gradients from
                # Gaussian-tail outliers without biasing the target distribution.
                flow_loss = F.smooth_l1_loss(v_pred, v_target)
                total_loss = total_loss + self.diffusion_weight * flow_loss
                metrics["flow_loss"] = flow_loss.item()

        metrics["total"] = total_loss.item()
        return total_loss, metrics

    @torch.no_grad()
    def sample(
        self,
        tool_pc_init: torch.Tensor,
        obj_pc: torch.Tensor,
        n_steps: int = None,
    ) -> torch.Tensor:
        """Generate delta_pose via Euler integration from noise to data.

        Args:
            tool_pc_init: (B, P, 3) tool point cloud at initial pose
            obj_pc:       (B, Q, 3) object point cloud
            n_steps:      number of Euler steps (default: self.n_inference_steps)

        Returns: (B, 9) predicted delta_pose
        """
        if n_steps is None:
            n_steps = self.n_inference_steps
        B = tool_pc_init.shape[0]
        device = tool_pc_init.device

        cond = self.build_condition(tool_pc_init, obj_pc)
        dt = 1.0 / n_steps

        # Start from pure noise (t=1)
        x = torch.randn(B, 1, 9, device=device)

        # Euler integration: t goes from 1 → 0
        for i in range(n_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=device)
            v = self.velocity_net(sample=x, timestep=t, cond=cond)
            x = x - v * dt  # step toward data

        return x.squeeze(1)  # (B, 9)


# Backward compat alias
JointModel = DiffusionModel


# --------------------------------------------------------------------------- #
# MovementModel — SDF + per-point movement prediction (no flow matching)
# --------------------------------------------------------------------------- #

class MovementModel(nn.Module):
    """Joint SDF prediction + per-object-point movement prediction.

    Architecture:
        encoder: SDFPointCloudEncoder (shared)
        sdf_head: SDFSegmentor (local-only, patch or point mode)
        movement_head: MovementPredictionHead (local + cross-attn)

    Given tool+object at contact configuration and a tool action (ΔT),
    predicts how each object point moves (the object response ΔO).
    """

    def __init__(
        self,
        # SDF head
        head_mode: str = "patch",
        patch_agg: str = "min",
        head_hidden: tuple[int, ...] = (128, 64),
        # Encoder (shared)
        num_pts: int = 512,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 4,
        vit_heads: int = 4,
        freeze_encoder: bool = False,
        # Movement head
        movement_n_heads: int = 4,
        # Loss weights
        sdf_weight: float = 1.0,
        movement_weight: float = 1.0,
    ):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.movement_weight = movement_weight

        # Shared encoder
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

        # SDF head (shares encoder)
        self.sdf_head = SDFSegmentor(
            head_mode=head_mode,
            patch_agg=patch_agg,
            head_hidden=head_hidden,
            num_pts=num_pts,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            freeze_encoder=freeze_encoder,
        )
        del self.sdf_head.encoder
        object.__setattr__(self.sdf_head, 'encoder', self.encoder)

        # Movement prediction head (local patch + cross-attn)
        self.movement_head = MovementPredictionHead(
            patch_dim=D,
            pose_dim=9,
            n_heads=movement_n_heads,
        )

    def forward(self, *args, **kwargs):
        """Route through loss() so DDP gradient sync hooks fire."""
        return self.loss(*args, **kwargs)

    def loss(
        self,
        # SDF inputs (tool at CONTACT pose)
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        tool_sdf_gt: torch.Tensor,
        obj_sdf_gt: torch.Tensor,
        # SDF inputs at INIT pose (optional)
        tool_pc_init: torch.Tensor = None,
        init_tool_sdf_gt: torch.Tensor = None,
        init_obj_sdf_gt: torch.Tensor = None,
        # Movement inputs
        tool_delta_action: torch.Tensor = None,
        obj_displacement_gt: torch.Tensor = None,
        # Unused (for interface compat with run_epoch)
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Joint SDF + movement loss."""
        metrics = {}
        total_loss = torch.tensor(0.0, device=tool_pc.device)

        # ---- Encode at CONTACT pose ----
        enc = self.encoder.encode(tool_pc, obj_pc)
        P = self.encoder.num_patches
        tool_tok, obj_tok = _split_tokens(enc, P)

        # ---- SDF task at CONTACT pose ----
        if self.sdf_head.head_mode == "point":
            tool_pred = self.sdf_head._predict_point(
                tool_pc, tool_tok,
                enc.tool_patch_idx, enc.tool_patch_centers,
                self.sdf_head.tool_head,
            )
            obj_pred = self.sdf_head._predict_point(
                obj_pc, obj_tok,
                enc.obj_patch_idx, enc.obj_patch_centers,
                self.sdf_head.obj_head,
            )
        else:
            tool_sdf_gt_agg = _aggregate_sdf(tool_sdf_gt, enc.tool_patch_idx, self.sdf_head.patch_agg)
            obj_sdf_gt_agg = _aggregate_sdf(obj_sdf_gt, enc.obj_patch_idx, self.sdf_head.patch_agg)
            tool_pred = self.sdf_head._predict_patch(tool_tok, self.sdf_head.tool_head)
            obj_pred = self.sdf_head._predict_patch(obj_tok, self.sdf_head.obj_head)
            tool_sdf_gt = tool_sdf_gt_agg
            obj_sdf_gt = obj_sdf_gt_agg

        tool_loss = F.smooth_l1_loss(tool_pred, tool_sdf_gt)
        obj_loss = F.smooth_l1_loss(obj_pred, obj_sdf_gt)
        sdf_loss = tool_loss + obj_loss
        total_loss = total_loss + self.sdf_weight * sdf_loss
        metrics["tool_sdf_loss"] = tool_loss.item()
        metrics["obj_sdf_loss"] = obj_loss.item()

        # ---- SDF task at INIT pose (optional) ----
        has_init_sdf = (
            tool_pc_init is not None
            and init_tool_sdf_gt is not None
            and init_obj_sdf_gt is not None
        )
        if has_init_sdf:
            enc_init = self.encoder.encode(tool_pc_init, obj_pc)
            init_tool_tok, init_obj_tok = _split_tokens(enc_init, P)

            if self.sdf_head.head_mode == "point":
                init_tool_pred = self.sdf_head._predict_point(
                    tool_pc_init, init_tool_tok,
                    enc_init.tool_patch_idx, enc_init.tool_patch_centers,
                    self.sdf_head.tool_head,
                )
                init_obj_pred = self.sdf_head._predict_point(
                    obj_pc, init_obj_tok,
                    enc_init.obj_patch_idx, enc_init.obj_patch_centers,
                    self.sdf_head.obj_head,
                )
            else:
                init_tool_sdf_gt_agg = _aggregate_sdf(init_tool_sdf_gt, enc_init.tool_patch_idx, self.sdf_head.patch_agg)
                init_obj_sdf_gt_agg = _aggregate_sdf(init_obj_sdf_gt, enc_init.obj_patch_idx, self.sdf_head.patch_agg)
                init_tool_pred = self.sdf_head._predict_patch(init_tool_tok, self.sdf_head.tool_head)
                init_obj_pred = self.sdf_head._predict_patch(init_obj_tok, self.sdf_head.obj_head)
                init_tool_sdf_gt = init_tool_sdf_gt_agg
                init_obj_sdf_gt = init_obj_sdf_gt_agg

            init_tool_loss = F.smooth_l1_loss(init_tool_pred, init_tool_sdf_gt)
            init_obj_loss = F.smooth_l1_loss(init_obj_pred, init_obj_sdf_gt)
            init_sdf_loss = init_tool_loss + init_obj_loss
            total_loss = total_loss + self.sdf_weight * init_sdf_loss
            metrics["init_tool_sdf_loss"] = init_tool_loss.item()
            metrics["init_obj_sdf_loss"] = init_obj_loss.item()

        # ---- Movement prediction (local patch + cross-attn) ----
        if tool_delta_action is not None and obj_displacement_gt is not None:
            disp_pred = self.movement_head(
                pc=obj_pc,
                patch_tokens=obj_tok,
                patch_idx=enc.obj_patch_idx,
                patch_centers=enc.obj_patch_centers,
                tool_delta_pose=tool_delta_action,
            )  # (B, N, 3)
            movement_loss = F.smooth_l1_loss(disp_pred, obj_displacement_gt)
            total_loss = total_loss + self.movement_weight * movement_loss
            metrics["movement_loss"] = movement_loss.item()
            # Log average absolute displacement magnitude for diagnostics
            metrics["movement_gt_mean_abs"] = obj_displacement_gt.abs().mean().item()
            metrics["movement_gt_max_abs"] = obj_displacement_gt.abs().max().item()

        metrics["total"] = total_loss.item()
        return total_loss, metrics