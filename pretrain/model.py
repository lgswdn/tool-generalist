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


# --------------------------------------------------------------------------- #
# PoseDiffusion - Diffusion head for pose prediction
# --------------------------------------------------------------------------- #

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # (B,) -> (B, dim)
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]  # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return emb


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to 6D representation."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)  # first two columns


class PoseDiffusion(nn.Module):
    """Diffusion model for predicting delta pose (translation + 6D rotation).

    Conditioning:
        - encoder features: tool_tokens, obj_tokens, global_feat
        - contact points: (K, 3) → MLP → pooling

    Output:
        - delta pose: (9,) = translation (3) + 6D rotation (6)
    """

    def __init__(
        self,
        encoder: SDFPointCloudEncoder,  # shared encoder (frozen or not)
        contact_hidden: tuple[int, ...] = (128, 128),
        diffusion_hidden: tuple[int, ...] = (256, 256, 256),
        time_embed_dim: int = 128,
        num_steps: int = 100,
        contact_pool: str = "mean",  # "mean" or "max"
    ):
        super().__init__()
        self.encoder = encoder
        self.num_steps = num_steps
        self.contact_pool = contact_pool

        D = encoder.feature_dim  # patch token dimension

        # Contact points encoder (separate MLP, not shared)
        self.contact_encoder = _make_mlp((3, contact_hidden[-1], contact_hidden[-1]))

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, diffusion_hidden[0]),
            nn.LayerNorm(diffusion_hidden[0]),
            nn.ELU(),
        )

        # Conditioning: concatenate all features
        # tool_tokens (P, D) → pool → D
        # obj_tokens (P, D) → pool → D
        # global_feat: D
        # contact_feat: contact_hidden[-1]
        cond_dim = 3 * D + contact_hidden[-1]

        # Denoising MLP
        # Input: cond (cond_dim) + time_embed (diffusion_hidden[0]) + noisy_pose (9)
        input_dim = cond_dim + diffusion_hidden[0] + 9

        layers = [nn.Linear(input_dim, diffusion_hidden[0])]
        for i in range(len(diffusion_hidden) - 1):
            layers.append(nn.LayerNorm(diffusion_hidden[i]))
            layers.append(nn.ELU())
            layers.append(nn.Linear(diffusion_hidden[i], diffusion_hidden[i + 1]))
        layers.append(nn.Linear(diffusion_hidden[-1], 9))  # output: 9D pose
        self.denoiser = nn.Sequential(*layers)

        # Beta schedule (linear)
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool patch tokens to single feature."""
        if self.contact_pool == "max":
            return tokens.max(dim=1).values
        return tokens.mean(dim=1)  # default: mean

    def _encode_contacts(self, contact_pts: torch.Tensor) -> torch.Tensor:
        """Encode contact points.

        Args:
            contact_pts: (B, K, 3) world-frame contact points

        Returns:
            contact_feat: (B, contact_hidden[-1])
        """
        B, K, _ = contact_pts.shape
        # Encode each point
        pts_feat = self.contact_encoder(contact_pts)  # (B, K, D)
        # Pool
        if self.contact_pool == "max":
            return pts_feat.max(dim=1).values
        return pts_feat.mean(dim=1)

    def get_conditioning(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        contact_pts: torch.Tensor,
    ) -> torch.Tensor:
        """Get conditioning features from encoder + contacts."""
        # Encode clouds (get patch tokens)
        res = self.encoder.encode(tool_pc, obj_pc)

        # Pool tokens
        tool_feat = self._pool_tokens(res.tool_tokens)  # (B, D)
        obj_feat = self._pool_tokens(res.obj_tokens)    # (B, D)
        global_feat = res.global_feat                    # (B, D)

        # Encode contacts
        contact_feat = self._encode_contacts(contact_pts)  # (B, contact_hidden[-1])

        # Concatenate
        cond = torch.cat([tool_feat, obj_feat, global_feat, contact_feat], dim=-1)
        return cond

    def forward(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        contact_pts: torch.Tensor,
        t: torch.Tensor,  # (B,) timestep
        noisy_pose: torch.Tensor,  # (B, 9) noisy delta pose
    ) -> torch.Tensor:  # (B, 9) predicted clean pose
        """Denoise pose at timestep t."""
        # Get conditioning
        cond = self.get_conditioning(tool_pc, obj_pc, contact_pts)

        # Time embedding
        t_emb = self.time_mlp(t)

        # Concatenate
        x = torch.cat([cond, t_emb, noisy_pose], dim=-1)

        # Predict clean pose
        return self.denoiser(x)

    def add_noise(
        self,
        pose: torch.Tensor,  # (B, 9) clean pose
        t: torch.Tensor,     # (B,) timestep
    ) -> tuple[torch.Tensor, torch.Tensor]:  # noisy_pose, noise
        """Add noise at timestep t."""
        device = pose.device
        alpha_t = self.alphas_cumprod[t].to(device)  # (B,)

        noise = torch.randn_like(pose)
        noisy_pose = torch.sqrt(alpha_t[:, None]) * pose + torch.sqrt(1 - alpha_t[:, None]) * noise

        return noisy_pose, noise

    def loss(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        contact_pts: torch.Tensor,
        delta_pose_gt: torch.Tensor,  # (B, 9) translation + 6D rotation
    ) -> tuple[torch.Tensor, dict]:
        """Training loss: predict noise added."""
        B = delta_pose_gt.shape[0]
        device = delta_pose_gt.device

        # Sample random timestep
        t = torch.randint(0, self.num_steps, (B,), device=device)

        # Add noise
        noisy_pose, noise = self.add_noise(delta_pose_gt, t)

        # Predict noise (or clean pose - standard DDPM predicts noise)
        pred = self.forward(tool_pc, obj_pc, contact_pts, t, noisy_pose)

        # Loss: predict the noise (standard diffusion training)
        loss = F.mse_loss(pred, noise)

        return loss, {"diffusion_loss": loss.item()}

    @torch.no_grad()
    def sample(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        contact_pts: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:  # (B, 9) predicted delta pose
        """Sample from diffusion model (DDPM sampling)."""
        B = tool_pc.shape[0]
        device = tool_pc.device

        # Start from noise
        pose = torch.randn(B, 9, device=device)

        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise
            pred_noise = self.forward(tool_pc, obj_pc, contact_pts, t_tensor, pose)

            # DDPM update
            alpha_t = self.alphas_cumprod[t].to(device)
            alpha_prev = self.alphas_cumprod[max(t - 1, 0)].to(device)
            beta_t = self.betas[t].to(device)

            if t > 0:
                noise = torch.randn_like(pose)
            else:
                noise = torch.zeros_like(pose)

            # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_t) * pred_noise) + sqrt(beta_t) * noise
            pose = (1.0 / torch.sqrt(alpha_t)) * (pose - beta_t / torch.sqrt(1 - alpha_t) * pred_noise)
            pose = pose + torch.sqrt(beta_t) * noise

        return pose


# --------------------------------------------------------------------------- #
# JointModel - Combined SDF + Diffusion training
# --------------------------------------------------------------------------- #

class JointModel(nn.Module):
    """Joint training of SDF prediction and pose diffusion.

    Uses shared encoder with two heads:
        - SDFSegmentor (SDF prediction)
        - PoseDiffusion (pose prediction)
    """

    def __init__(
        self,
        # SDF head
        head_mode: str = "point",
        patch_agg: str = "mean",
        # Encoder (shared)
        num_pts: int = 512,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 4,
        vit_heads: int = 4,
        freeze_encoder: bool = False,
        # Diffusion head
        diffusion_hidden: tuple[int, ...] = (256, 256, 256),
        diffusion_steps: int = 100,
        contact_hidden: tuple[int, ...] = (128, 128),
        contact_pool: str = "mean",
        # Loss weights
        sdf_weight: float = 1.0,
        diffusion_weight: float = 1.0,
    ):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.diffusion_weight = diffusion_weight

        # Shared encoder config
        enc_cfg = SDFEncoderCfg(
            num_pts=num_pts,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            freeze=freeze_encoder,
        )

        # Shared encoder (created once, passed to both heads)
        self.encoder = SDFPointCloudEncoder(enc_cfg)

        # SDF head
        self.sdf_head = SDFSegmentor(
            head_mode=head_mode,
            patch_agg=patch_agg,
            num_pts=num_pts,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            freeze_encoder=freeze_encoder,
        )
        # Replace encoder with shared one
        self.sdf_head.encoder = self.encoder

        # Diffusion head
        self.diffusion_head = PoseDiffusion(
            encoder=self.encoder,
            contact_hidden=contact_hidden,
            diffusion_hidden=diffusion_hidden,
            num_steps=diffusion_steps,
            contact_pool=contact_pool,
        )

    def forward_sdf(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SDF prediction forward."""
        return self.sdf_head(tool_pc, obj_pc)

    def forward_diffusion(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        contact_pts: torch.Tensor,
        t: torch.Tensor,
        noisy_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Diffusion forward."""
        return self.diffusion_head(tool_pc, obj_pc, contact_pts, t, noisy_pose)

    def loss(
        self,
        # SDF inputs
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        tool_sdf_gt: torch.Tensor,
        obj_sdf_gt: torch.Tensor,
        # Diffusion inputs (optional)
        contact_pts: torch.Tensor = None,
        delta_pose_gt: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """Joint loss computation."""
        metrics = {}
        total_loss = torch.tensor(0.0, device=tool_pc.device)

        # SDF loss
        sdf_loss, sdf_metrics = self.sdf_head.loss(
            tool_pc, obj_pc, tool_sdf_gt, obj_sdf_gt,
            sdf_weight=self.sdf_weight,
        )
        total_loss = total_loss + sdf_loss
        metrics.update(sdf_metrics)

        # Diffusion loss (if inputs provided)
        if contact_pts is not None and delta_pose_gt is not None:
            diff_loss, diff_metrics = self.diffusion_head.loss(
                tool_pc, obj_pc, contact_pts, delta_pose_gt,
            )
            total_loss = total_loss + self.diffusion_weight * diff_loss
            metrics.update(diff_metrics)

        metrics["total"] = total_loss.item()
        return total_loss, metrics

    @torch.no_grad()
    def sample_pose(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        contact_pts: torch.Tensor,
    ) -> torch.Tensor:
        """Sample pose from diffusion head."""
        return self.diffusion_head.sample(tool_pc, obj_pc, contact_pts)
