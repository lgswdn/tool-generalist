"""model.py — Joint encoder pretraining: SDF prediction + Pose diffusion.

Uses TransformerForDiffusion from diffusion_policy for pose prediction.
Uses DDPMScheduler from diffusers for noise schedule.

Architecture:
  JointModel (shared encoder + two heads)
    ├── encoder: SDFPointCloudEncoder  (ViT backbone, FPS patches)
    ├── sdf_head: SDFSegmentor          (point/patch → SDF)
    └── transformer: TransformerForDiffusion (cross-attention to patches)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDPMScheduler

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


# --------------------------------------------------------------------------- #
# SDFSegmentor
# --------------------------------------------------------------------------- #

class SDFSegmentor(nn.Module):
    """SDF prediction wrapper around SDFPointCloudEncoder."""

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
            self.xyz_embed = _make_mlp((3, D, D))
            self.tool_head = _make_mlp((3 * D,) + head_hidden + (1,))
            self.obj_head = _make_mlp((3 * D,) + head_hidden + (1,))
        else:
            self.tool_head = _make_mlp((2 * D,) + head_hidden + (1,))
            self.obj_head = _make_mlp((2 * D,) + head_hidden + (1,))

    def _predict_point(
        self,
        pc: torch.Tensor,
        patch_tokens: torch.Tensor,
        global_feat: torch.Tensor,
        patch_idx: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        B, N, _ = pc.shape
        D = self.encoder.feature_dim
        P, K = patch_tokens.shape[1], patch_idx.shape[2]

        pt_xyz = self.xyz_embed(pc)

        pt_patch = torch.zeros(B, N, D, device=pc.device, dtype=pc.dtype)
        exp_tok = patch_tokens.unsqueeze(2).expand(B, P, K, D)
        flat_idx = patch_idx.reshape(B, P * K, 1).expand(B, P * K, D)
        flat_tok = exp_tok.reshape(B, P * K, D)
        pt_patch.scatter_(1, flat_idx, flat_tok)

        pt_global = global_feat.unsqueeze(1).expand(B, N, D)

        feat = torch.cat([pt_xyz, pt_patch, pt_global], dim=-1)
        return head(feat).squeeze(-1)

    def _predict_patch(
        self,
        patch_tokens: torch.Tensor,
        global_feat: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        B, P, D = patch_tokens.shape
        pt_global = global_feat.unsqueeze(1).expand(B, P, D)
        feat = torch.cat([patch_tokens, pt_global], dim=-1)
        return head(feat).squeeze(-1)

    def forward(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            obj_sdf = self._predict_patch(res.obj_tokens, res.global_feat, self.obj_head)

        return tool_sdf, obj_sdf

    def loss(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        tool_sdf_gt: torch.Tensor,
        obj_sdf_gt: torch.Tensor,
        sdf_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
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
            tool_sdf_gt = _aggregate_sdf(tool_sdf_gt, res.tool_patch_idx, self.patch_agg)
            obj_sdf_gt = _aggregate_sdf(obj_sdf_gt, res.obj_patch_idx, self.patch_agg)
            tool_pred = self._predict_patch(res.tool_tokens, res.global_feat, self.tool_head)
            obj_pred = self._predict_patch(res.obj_tokens, res.global_feat, self.obj_head)

        tool_loss = F.smooth_l1_loss(tool_pred, tool_sdf_gt)
        obj_loss = F.smooth_l1_loss(obj_pred, obj_sdf_gt)
        total = sdf_weight * (tool_loss + obj_loss)

        return total, {
            "tool_sdf_loss": tool_loss.item(),
            "obj_sdf_loss": obj_loss.item(),
        }


# --------------------------------------------------------------------------- #
# JointModel - Combined SDF + Diffusion training
# --------------------------------------------------------------------------- #

class JointModel(nn.Module):
    """Joint training of SDF prediction and pose diffusion.

    Uses shared encoder with:
        - SDFSegmentor (SDF prediction)
        - TransformerForDiffusion (cross-attention to 32 encoder patches)

    The transformer takes:
        - sample: noisy pose (B, 9, 1)
        - timestep: diffusion step
        - cond: encoder patches (B, 32, 128) = [tool_tokens, obj_tokens]
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
        # Diffusion transformer
        n_layer: int = 4,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.0,
        # Loss weights
        sdf_weight: float = 1.0,
        diffusion_weight: float = 1.0,
    ):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.diffusion_weight = diffusion_weight

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

        # SDF head
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
        self.sdf_head.encoder = self.encoder

        # TransformerForDiffusion: condition on encoder patches
        # Encoder produces 2*P patches (tool + obj) of dimension D.
        # The transformer cross-attends to these to predict delta pose noise.
        self.transformer = TransformerForDiffusion(
            input_dim=9,
            output_dim=9,
            horizon=1,           # single token = full 9D pose
            n_obs_steps=2 * P,   # 32 encoder patches (16 tool + 16 obj)
            cond_dim=D,          # 128 (encoder feature dim)
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            obs_as_cond=True,
            time_as_cond=True,
            n_cond_layers=1,
        )

        # Position-aware conditioning: project tool-obj centroids into token space
        # This provides an explicit position signal since encoder patches are locally centered
        self.cond_pos_proj = nn.Linear(6, D)

        # DDPM noise scheduler (from diffusers)
        # DEBUG: Use 100 steps and fix timestep to 50 for controlled test
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
        )

    def loss(
        self,
        # SDF inputs (tool at CONTACT pose)
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        tool_sdf_gt: torch.Tensor,
        obj_sdf_gt: torch.Tensor,
        # Diffusion inputs (tool at INIT pose)
        tool_pc_init: torch.Tensor = None,
        delta_pose_gt: torch.Tensor = None,
        noise: torch.Tensor = None,  # optional: provide target noise directly
    ) -> Tuple[torch.Tensor, dict]:
        """Joint loss computation."""
        metrics = {}
        total_loss = torch.tensor(0.0, device=tool_pc.device)

        # SDF task: encode tool at contact pose
        enc_result_contact = self.encoder.encode(tool_pc, obj_pc)

        # SDF loss
        if self.sdf_head.head_mode == "point":
            tool_pred = self.sdf_head._predict_point(
                tool_pc, enc_result_contact.tool_tokens, enc_result_contact.global_feat,
                enc_result_contact.tool_patch_idx, self.sdf_head.tool_head,
            )
            obj_pred = self.sdf_head._predict_point(
                obj_pc, enc_result_contact.obj_tokens, enc_result_contact.global_feat,
                enc_result_contact.obj_patch_idx, self.sdf_head.obj_head,
            )
        else:
            tool_sdf_gt_agg = _aggregate_sdf(tool_sdf_gt, enc_result_contact.tool_patch_idx, self.sdf_head.patch_agg)
            obj_sdf_gt_agg = _aggregate_sdf(obj_sdf_gt, enc_result_contact.obj_patch_idx, self.sdf_head.patch_agg)
            tool_pred = self.sdf_head._predict_patch(enc_result_contact.tool_tokens, enc_result_contact.global_feat, self.sdf_head.tool_head)
            obj_pred = self.sdf_head._predict_patch(enc_result_contact.obj_tokens, enc_result_contact.global_feat, self.sdf_head.obj_head)
            tool_sdf_gt = tool_sdf_gt_agg
            obj_sdf_gt = obj_sdf_gt_agg

        tool_loss = F.smooth_l1_loss(tool_pred, tool_sdf_gt)
        obj_loss = F.smooth_l1_loss(obj_pred, obj_sdf_gt)
        sdf_loss = tool_loss + obj_loss
        total_loss = total_loss + self.sdf_weight * sdf_loss
        metrics["tool_sdf_loss"] = tool_loss.item()
        metrics["obj_sdf_loss"] = obj_loss.item()

        # Diffusion task: condition on encoder features from INIT pose
        if tool_pc_init is not None and delta_pose_gt is not None:
            B = delta_pose_gt.shape[0]
            device = delta_pose_gt.device

            # Encode tool at INITIAL pose + object → encoder patches
            enc_result_init = self.encoder.encode(tool_pc_init, obj_pc)
            cond = torch.cat(
                [enc_result_init.tool_tokens, enc_result_init.obj_tokens], dim=1
            )  # (B, 32, 128)

            # Add explicit position bias: tool-obj centroids → broadcast to all tokens
            tool_centroid = tool_pc_init.mean(dim=1)  # (B, 3)
            obj_centroid = obj_pc.mean(dim=1)          # (B, 3)
            pos_info = torch.cat([tool_centroid, obj_centroid], dim=-1)  # (B, 6)
            pos_bias = self.cond_pos_proj(pos_info)    # (B, D)
            cond = cond + pos_bias.unsqueeze(1)        # broadcast-add to all 32 tokens

            # Full 9D pose as single token: (B, 1, 9)
            clean_data = delta_pose_gt.unsqueeze(1)  # (B, 1, 9)

            # Random noise each step
            noise = torch.randn_like(clean_data)      # (B, 1, 9)

            # Random timesteps per sample
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=device, dtype=torch.long,
            )

            # Proper DDPM noising: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
            noisy_data = self.noise_scheduler.add_noise(clean_data, noise, timesteps)

            # Forward pass: predict noise conditioned on encoder features
            noise_prediction = self.transformer(
                sample=noisy_data,
                timestep=timesteps,
                cond=cond,
            )  # (B, 1, 9)

            # Loss: MSE between predicted and target noise
            diff_loss = F.mse_loss(noise_prediction, noise)
            total_loss = total_loss + self.diffusion_weight * diff_loss
            metrics["diffusion_loss"] = diff_loss.item()

        metrics["total"] = total_loss.item()
        return total_loss, metrics

    def forward_diffusion(
        self,
        tool_pc_init: torch.Tensor,
        obj_pc: torch.Tensor,
        noisy_target: torch.Tensor,  # (B, 9, 1)
        timesteps: torch.Tensor,     # (B,)
    ) -> torch.Tensor:  # (B, 9, 1) predicted noise
        """Forward pass for diffusion: predict noise given conditioning."""
        enc_result = self.encoder.encode(tool_pc_init, obj_pc)
        cond = torch.cat([enc_result.tool_tokens, enc_result.obj_tokens], dim=1)
        return self.transformer(sample=noisy_target, timestep=timesteps, cond=cond)


# --------------------------------------------------------------------------- #
# JointModelMLP - SDF + Simple MLP pose regression (for debugging)
# --------------------------------------------------------------------------- #

class JointModelMLP(nn.Module):
    """Joint training of SDF prediction + simple MLP pose regression.

    For debugging: tests if pose regression works with simple MLP.
    MLP takes delta_pose (answer) as input and predicts delta_pose.
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
        # MLP pose head
        pose_hidden: tuple[int, ...] = (256, 256),
        # Loss weights
        sdf_weight: float = 1.0,
        pose_weight: float = 1.0,
    ):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.pose_weight = pose_weight

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
        self.sdf_head.encoder = self.encoder

        # Simple MLP pose head: answer → answer (identity test)
        pose_layers = []
        in_dim = 9  # delta_pose dimension
        for h in pose_hidden:
            pose_layers.append(nn.Linear(in_dim, h))
            pose_layers.append(nn.LayerNorm(h))
            pose_layers.append(nn.ELU())
            in_dim = h
        pose_layers.append(nn.Linear(in_dim, 9))  # output delta_pose
        self.pose_head = nn.Sequential(*pose_layers)

    def loss(
        self,
        # SDF inputs (tool at CONTACT pose)
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        tool_sdf_gt: torch.Tensor,
        obj_sdf_gt: torch.Tensor,
        # Pose inputs
        delta_pose_gt: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Joint loss computation."""
        metrics = {}
        total_loss = torch.tensor(0.0, device=tool_pc.device)

        # SDF task: encode tool at contact pose
        enc_result_contact = self.encoder.encode(tool_pc, obj_pc)

        # SDF loss
        if self.sdf_head.head_mode == "point":
            tool_pred = self.sdf_head._predict_point(
                tool_pc, enc_result_contact.tool_tokens, enc_result_contact.global_feat,
                enc_result_contact.tool_patch_idx, self.sdf_head.tool_head,
            )
            obj_pred = self.sdf_head._predict_point(
                obj_pc, enc_result_contact.obj_tokens, enc_result_contact.global_feat,
                enc_result_contact.obj_patch_idx, self.sdf_head.obj_head,
            )
        else:
            tool_sdf_gt_agg = _aggregate_sdf(tool_sdf_gt, enc_result_contact.tool_patch_idx, self.sdf_head.patch_agg)
            obj_sdf_gt_agg = _aggregate_sdf(obj_sdf_gt, enc_result_contact.obj_patch_idx, self.sdf_head.patch_agg)
            tool_pred = self.sdf_head._predict_patch(enc_result_contact.tool_tokens, enc_result_contact.global_feat, self.sdf_head.tool_head)
            obj_pred = self.sdf_head._predict_patch(enc_result_contact.obj_tokens, enc_result_contact.global_feat, self.sdf_head.obj_head)
            tool_sdf_gt = tool_sdf_gt_agg
            obj_sdf_gt = obj_sdf_gt_agg

        tool_loss = F.smooth_l1_loss(tool_pred, tool_sdf_gt)
        obj_loss = F.smooth_l1_loss(obj_pred, obj_sdf_gt)
        sdf_loss = tool_loss + obj_loss
        total_loss = total_loss + self.sdf_weight * sdf_loss
        metrics["tool_sdf_loss"] = tool_loss.item()
        metrics["obj_sdf_loss"] = obj_loss.item()

        # Pose task: simple MLP (answer → answer)
        if delta_pose_gt is not None:
            pose_pred = self.pose_head(delta_pose_gt)  # (B, 9)
            pose_loss = F.mse_loss(pose_pred, delta_pose_gt)
            total_loss = total_loss + self.pose_weight * pose_loss
            metrics["pose_loss"] = pose_loss.item()

        metrics["total"] = total_loss.item()
        return total_loss, metrics