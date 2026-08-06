"""Canonical pretrain model: joint SDF heads plus denoising/postcontact heads.

Architecture:
  encoder          : TCEPointCloudEncoder
  pose_cross_attn  : PoseCrossAttention (injects noised pose into encoder tokens)
  sdf_head         : Pose-conditioned SDF prediction (point or patch mode)
  diff head        : translation-only MLP head with translation loss
  post head        : pose9d MLP head with geodesic rotation loss
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geometry.sdf import mutual_signed_sdf_labels_env_frame
from utils.geometry.pose import rotation_from_pose9d

_RPDIFF_SRC = (
    Path(__file__).resolve().parents[1] / "thirdparty" / "rpdiff" / "src"
)
if _RPDIFF_SRC.exists() and str(_RPDIFF_SRC) not in sys.path:
    sys.path.insert(0, str(_RPDIFF_SRC))
from rpdiff.training.losses import TransformChamferWrapper
from rpdiff.utils.torch3d_util import matrix_to_quaternion

__all__ = [
    "ContactDiffusionModel",
    "ConditionQueryGenerator",
    "PoseCrossAttention",
    "Pose9DHead",
    "TranslationHead",
    "TCEEncodeResult",
    "TCEPointCloudEncoder",
    "TCEPointCloudEncoderCfg",
]


@dataclass
class TCEPointCloudEncoderCfg:
    num_pts: int
    patch_size: int
    encoder_channel: int
    vit_depth: int
    vit_heads: int
    freeze: bool
    vit_attention_mode: str | None = None
    kinematic_conditioning: bool = False
    kinematic_attention_layers: int = 1


class TCEEncodeResult(NamedTuple):
    fused_tokens: torch.Tensor
    tool_patch_idx: torch.Tensor
    obj_patch_idx: torch.Tensor
    tool_patch_centers: torch.Tensor
    obj_patch_centers: torch.Tensor


class _FormerPatchEncoder(nn.Module):
    """Former-style PointNet patch encoder: MLP, max+mean pool, projection."""

    def __init__(self, out_dim: int, hidden: tuple[int, int] = (64, 128)):
        super().__init__()
        h0, h1 = hidden
        self.mlp1 = nn.Sequential(nn.Linear(3, h0), nn.LayerNorm(h0), nn.GELU())
        self.mlp2 = nn.Sequential(nn.Linear(h0, h1), nn.LayerNorm(h1), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(h1 * 2, out_dim), nn.LayerNorm(out_dim))

    def forward(self, patch_points: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(patch_points)
        x = self.mlp2(x)
        x = torch.cat((x.max(dim=2).values, x.mean(dim=2)), dim=-1)
        return self.proj(x)


class _FormerPosEmbed(nn.Module):
    """Former-style MLP positional embedding for patch centers."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, out_dim // 2),
            nn.LayerNorm(out_dim // 2),
            nn.GELU(),
            nn.Linear(out_dim // 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        return self.mlp(centers)


class _FormerViTBlock(nn.Module):
    """Former-style pre-norm ViT block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(y)
        return x + self.ffn(self.norm2(x))


class TCEPointCloudEncoder(nn.Module):
    """Tool Contact Encoder with Former-style PointNet patches and joint ViT."""

    def __init__(self, cfg: TCEPointCloudEncoderCfg):
        super().__init__()
        self.cfg = cfg
        self._P = max(1, cfg.num_pts // cfg.patch_size)
        self._D = cfg.encoder_channel
        if cfg.vit_attention_mode not in {"joint_self", "cross_only"}:
            raise ValueError(
                "vit_attention_mode is required and must be joint_self or cross_only, got "
                f"{cfg.vit_attention_mode!r}"
            )
        self.patch_enc = _FormerPatchEncoder(cfg.encoder_channel)
        self.pos_embed = _FormerPosEmbed(cfg.encoder_channel)
        self.type_embed = nn.Parameter(torch.zeros(2, cfg.encoder_channel))
        nn.init.normal_(self.type_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder_channel))
        nn.init.normal_(self.cls_token, std=0.02)
        self.vit = nn.ModuleList(
            [
                _FormerViTBlock(cfg.encoder_channel, cfg.vit_heads, mlp_ratio=4.0, dropout=0.0)
                for _ in range(max(1, cfg.vit_depth))
            ]
        )
        self.norm = nn.LayerNorm(cfg.encoder_channel)
        if cfg.kinematic_conditioning:
            if int(cfg.kinematic_attention_layers) < 1:
                raise ValueError("kinematic_attention_layers must be >= 1")
            self.kinematic_state_embed = nn.Parameter(
                torch.zeros(3, cfg.encoder_channel)
            )
            nn.init.normal_(self.kinematic_state_embed, std=0.02)
            self.kinematic_vit = nn.ModuleList(
                [
                    _FormerViTBlock(
                        cfg.encoder_channel,
                        cfg.vit_heads,
                        mlp_ratio=4.0,
                        dropout=0.0,
                    )
                    for _ in range(int(cfg.kinematic_attention_layers))
                ]
            )
            self.kinematic_norm = nn.LayerNorm(cfg.encoder_channel)
            self.geometry_kinematic_vit = nn.ModuleList(
                [
                    _FormerViTBlock(
                        cfg.encoder_channel,
                        cfg.vit_heads,
                        mlp_ratio=4.0,
                        dropout=0.0,
                    )
                    for _ in range(int(cfg.kinematic_attention_layers))
                ]
            )
            self.geometry_kinematic_norm = nn.LayerNorm(cfg.encoder_channel)
        if cfg.freeze:
            for param in self.parameters():
                param.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return self._D

    @property
    def num_patches(self) -> int:
        return self._P

    def _fps_indices(self, pc: torch.Tensor, num_centers: int) -> torch.Tensor:
        B, N, _ = pc.shape
        centroids = torch.zeros(B, num_centers, dtype=torch.long, device=pc.device)
        distance = torch.full((B, N), float("inf"), device=pc.device, dtype=pc.dtype)
        farthest = torch.zeros(B, dtype=torch.long, device=pc.device)
        batch_indices = torch.arange(B, dtype=torch.long, device=pc.device)
        for i in range(num_centers):
            centroids[:, i] = farthest
            centroid = pc[batch_indices, farthest].view(B, 1, 3)
            dist = ((pc - centroid) ** 2).sum(dim=-1)
            distance = torch.minimum(distance, dist)
            farthest = distance.max(dim=1).indices
        return centroids

    def _knn_patch_indices(self, pc: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        B, N, _ = pc.shape
        K = self.cfg.patch_size
        k_eff = min(K, N)
        dist = torch.cdist(centers, pc)
        idx = dist.topk(k=k_eff, dim=-1, largest=False).indices
        if k_eff < K:
            pad = idx[..., -1:].expand(B, centers.shape[1], K - k_eff)
            idx = torch.cat((idx, pad), dim=-1)
        return idx

    def _encode_one(self, pc: torch.Tensor, type_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = pc.shape
        P = min(self._P, N)
        center_idx = self._fps_indices(pc, P)
        batch = torch.arange(B, device=pc.device).view(B, 1)
        centers = pc[batch, center_idx]
        idx = self._knn_patch_indices(pc, centers)
        batch = torch.arange(B, device=pc.device).view(B, 1, 1)
        patches = pc[batch, idx]
        relative_patch_coords = patches - centers.unsqueeze(2)
        patch_features = self.patch_enc(relative_patch_coords)
        type_ids = torch.full((B, P), int(type_id), dtype=torch.long, device=pc.device)
        tokens = (
            patch_features
            + self.pos_embed(centers)
            + self.type_embed[type_ids]
        )
        return tokens, idx, centers

    def _pad_to_num_patches(
        self,
        tokens: torch.Tensor,
        idx: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, D = tokens.shape
        if P == self._P:
            return tokens, idx, centers
        pad_p = self._P - P
        tokens = torch.cat((tokens, tokens[:, -1:, :].expand(B, pad_p, D)), dim=1)
        idx = torch.cat((idx, idx[:, -1:, :].expand(B, pad_p, idx.shape[-1])), dim=1)
        centers = torch.cat((centers, centers[:, -1:, :].expand(B, pad_p, 3)), dim=1)
        return tokens, idx, centers

    def _encode_kinematic_states(
        self, kinematic_tool_clouds: torch.Tensor
    ) -> torch.Tensor:
        if tuple(kinematic_tool_clouds.shape[1:]) != (
            3,
            self.cfg.num_pts,
            3,
        ):
            raise ValueError(
                "kinematic_tool_clouds must have shape "
                f"(B, 3, {self.cfg.num_pts}, 3), got "
                f"{tuple(kinematic_tool_clouds.shape)}"
            )
        state_tokens = []
        for state_index in range(3):
            tokens, indices, centers = self._encode_one(
                kinematic_tool_clouds[:, state_index], type_id=0
            )
            tokens, _, _ = self._pad_to_num_patches(tokens, indices, centers)
            state_tokens.append(
                tokens.mean(dim=1)
                + self.kinematic_state_embed[state_index].view(1, -1)
            )
        kinematic_tokens = torch.stack(state_tokens, dim=1)
        for block in self.kinematic_vit:
            kinematic_tokens = block(kinematic_tokens)
        return self.kinematic_norm(kinematic_tokens)

    def encode(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        *,
        kinematic_tool_clouds: torch.Tensor | None = None,
    ) -> TCEEncodeResult:
        if self.cfg.kinematic_conditioning and kinematic_tool_clouds is None:
            raise ValueError(
                "kinematic_tool_clouds is required by this TCE encoder"
            )
        if not self.cfg.kinematic_conditioning and kinematic_tool_clouds is not None:
            raise ValueError(
                "kinematic_tool_clouds was supplied to a standard TCE encoder"
            )
        tool_tok, tool_idx, tool_centers = self._encode_one(tool_pc, type_id=0)
        obj_tok, obj_idx, obj_centers = self._encode_one(obj_pc, type_id=1)
        tool_tok, tool_idx, tool_centers = self._pad_to_num_patches(tool_tok, tool_idx, tool_centers)
        obj_tok, obj_idx, obj_centers = self._pad_to_num_patches(obj_tok, obj_idx, obj_centers)
        fused = torch.cat((tool_tok, obj_tok), dim=1)
        cls = self.cls_token.expand(fused.shape[0], -1, -1)
        fused = torch.cat((cls, fused), dim=1)
        attn_mask = None
        if self.cfg.vit_attention_mode == "cross_only":
            # [CLS, tool_0..tool_P-1, object_0..object_P-1]. Patch-token
            # queries may read only the opposite body. CLS may summarize all
            # tokens, but patch queries cannot read CLS, preventing a global
            # shortcut through this one simultaneous attention update.
            size = 1 + 2 * self._P
            attn_mask = torch.ones(size, size, dtype=torch.bool, device=fused.device)
            attn_mask[0, :] = False
            attn_mask[1 : 1 + self._P, 1 + self._P :] = False
            attn_mask[1 + self._P :, 1 : 1 + self._P] = False
        for block in self.vit:
            fused = block(fused, attn_mask=attn_mask)
        fused = self.norm(fused)
        fused = fused[:, 1:, :]
        if self.cfg.kinematic_conditioning:
            kinematic_tokens = self._encode_kinematic_states(
                kinematic_tool_clouds
            )
            fused = torch.cat((fused, kinematic_tokens), dim=1)
            for block in self.geometry_kinematic_vit:
                fused = block(fused)
            fused = self.geometry_kinematic_norm(fused)
        return TCEEncodeResult(
            fused_tokens=fused,
            tool_patch_idx=tool_idx,
            obj_patch_idx=obj_idx,
            tool_patch_centers=tool_centers,
            obj_patch_centers=obj_centers,
        )

    def forward(
        self,
        tool_pc: torch.Tensor,
        obj_pc: torch.Tensor,
        *,
        kinematic_tool_clouds: torch.Tensor | None = None,
    ) -> TCEEncodeResult:
        return self.encode(
            tool_pc,
            obj_pc,
            kinematic_tool_clouds=kinematic_tool_clouds,
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_pos: int | None = None):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_pos is not None:
            x = torch.clamp(x, 0, self.max_pos)
        half_dim = self.dim // 2
        scale = torch.log(torch.tensor(10000.0, device=x.device, dtype=x.dtype))
        scale = scale / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -scale)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


def _matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    B = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    m1 = trace > 0
    if m1.any():
        s = torch.sqrt(trace[m1] + 1.0) * 2
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
        q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s
        q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s

    m2 = (~m1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if m2.any():
        s = torch.sqrt(1.0 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2]) * 2
        q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s
        q[m2, 1] = 0.25 * s
        q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
        q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s

    m3 = (~m1) & (~m2) & (R[:, 1, 1] > R[:, 2, 2])
    if m3.any():
        s = torch.sqrt(1.0 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2]) * 2
        q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s
        q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s
        q[m3, 2] = 0.25 * s
        q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s

    m4 = (~m1) & (~m2) & (~m3)
    if m4.any():
        s = torch.sqrt(1.0 + R[m4, 2, 2] - R[m4, 0, 0] - R[m4, 1, 1]) * 2
        q[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s
        q[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s
        q[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s
        q[m4, 3] = 0.25 * s

    return F.normalize(q, dim=-1)


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
    return res.fused_tokens[:, :P, :], res.fused_tokens[:, P : 2 * P, :]


# ============================================================================ #
# PoseCrossAttention — inject noised pose into encoder tokens
# ============================================================================ #

class ConditionQueryGenerator(nn.Module):
    """Generate A/B/C/D query groups for TCE decoder cross-attention."""

    def __init__(
        self,
        token_dim: int,
        hidden_dims: tuple[int, ...],
        num_query_A: int,
        num_query_B: int,
        num_query_C: int,
        num_query_D: int,
        pose_dim: int,
        movement_cond_dim: int,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_query_A = int(num_query_A)
        self.num_query_B = int(num_query_B)
        self.num_query_C = int(num_query_C)
        self.num_query_D = int(num_query_D)
        self.pose_dim = int(pose_dim)
        self.movement_cond_dim = int(movement_cond_dim)
        self.query_A = _make_mlp((pose_dim,) + hidden_dims + (self.num_query_A * token_dim,))
        self.query_B = _make_mlp((9,) + hidden_dims + (self.num_query_B * token_dim,))
        self.query_C = _make_mlp((9,) + hidden_dims + (self.num_query_C * token_dim,))
        self.query_D = _make_mlp((7,) + hidden_dims + (self.num_query_D * token_dim,))

    def forward(self, pose_signal: torch.Tensor, movement_cond: torch.Tensor) -> torch.Tensor:
        B = pose_signal.shape[0]
        tool_delta = movement_cond[..., :9]
        object_delta = movement_cond[..., 9:18]
        physics = movement_cond[..., 18:25]
        if tool_delta.shape[-1] < 9:
            tool_delta = F.pad(tool_delta, (0, 9 - tool_delta.shape[-1]))
        if object_delta.shape[-1] < 9:
            object_delta = F.pad(object_delta, (0, 9 - object_delta.shape[-1]))
        if physics.shape[-1] < 7:
            physics = F.pad(physics, (0, 7 - physics.shape[-1]))

        queries = [
            self.query_A(pose_signal).view(B, self.num_query_A, self.token_dim),
            self.query_B(tool_delta).view(B, self.num_query_B, self.token_dim),
            self.query_C(object_delta).view(B, self.num_query_C, self.token_dim),
            self.query_D(physics).view(B, self.num_query_D, self.token_dim),
        ]
        return torch.cat(queries, dim=1)


class PoseCrossAttention(nn.Module):
    """A/B/C/D query decoder over joint TCE tokens."""

    def __init__(
        self,
        token_dim: int,
        pose_dim: int = 7,
        movement_cond_dim: int = 14,
        n_heads: int = 4,
        n_layers: int = 2,
        condition_mlp_hidden_dims: tuple[int, ...] = (128, 128),
        num_query_A: int = 4,
        num_query_B: int = 4,
        num_query_C: int = 4,
        num_query_D: int = 4,
    ):
        super().__init__()
        self.token_dim = token_dim

        self.query_generator = ConditionQueryGenerator(
            token_dim=token_dim,
            hidden_dims=condition_mlp_hidden_dims,
            num_query_A=num_query_A,
            num_query_B=num_query_B,
            num_query_C=num_query_C,
            num_query_D=num_query_D,
            pose_dim=pose_dim,
            movement_cond_dim=movement_cond_dim,
        )
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "query_cross_attn": nn.MultiheadAttention(
                    embed_dim=token_dim,
                    num_heads=n_heads,
                    batch_first=True,
                ),
                "token_cross_attn": nn.MultiheadAttention(
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
        movement_cond: torch.Tensor, # (B, 14) delta_T trans+quat and delta_O trans+quat
    ) -> torch.Tensor:
        """Returns pose-conditioned tokens P' with same shape (B, 2P, D)."""
        queries = self.query_generator(pose_7d, movement_cond)

        out = tokens
        for layer in self.layers:
            query_residual = queries
            queries_norm = layer["norm1"](queries)
            query_out, _ = layer["query_cross_attn"](
                query=queries_norm,
                key=out,
                value=out,
            )
            queries = query_residual + query_out
            queries = queries + layer["ff"](layer["norm2"](queries))

            residual = out
            out_norm = layer["norm1"](out)
            attn_out, _ = layer["token_cross_attn"](
                query=out_norm,
                key=queries,
                value=queries,
            )
            out = residual + attn_out
            residual = out
            out = residual + layer["ff"](layer["norm2"](out))

        return out  # (B, 2P, D)


def _make_relu_mlp(dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class _ConditionalBatchNorm1d(nn.Module):
    """Conditional BN used by the UniCORN paper contact decoder."""

    def __init__(self, num_features: int, condition_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.affine = nn.Linear(condition_dim, 2 * num_features)
        with torch.no_grad():
            self.affine.weight.zero_()
            self.affine.bias[:num_features].fill_(1.0)
            self.affine.bias[num_features:].zero_()

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, patches, channels = features.shape
        normalized = self.bn(features.reshape(batch * patches, channels)).reshape(
            batch, patches, channels
        )
        gamma, beta = self.affine(condition).chunk(2, dim=-1)
        return normalized * gamma.unsqueeze(1) + beta.unsqueeze(1)


class _PaperContactCMLP(nn.Module):
    """Three-layer conditional MLP with two 128D CBN hidden layers."""

    def __init__(
        self,
        token_dim: int,
        hidden_dims: tuple[int, ...],
        condition_dim: int,
    ):
        super().__init__()
        if tuple(hidden_dims) != (128, 128):
            raise ValueError(
                "paper_cmlp_cbn requires decoder hidden dimensions (128, 128)"
            )
        self.input = nn.Linear(token_dim, hidden_dims[0])
        self.hidden = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.cbn_input = _ConditionalBatchNorm1d(hidden_dims[0], condition_dim)
        self.cbn_hidden = _ConditionalBatchNorm1d(hidden_dims[1], condition_dim)
        self.output = nn.Linear(hidden_dims[1], 1)

    def forward(self, tokens: torch.Tensor, opposite_global: torch.Tensor) -> torch.Tensor:
        features = F.gelu(self.cbn_input(self.input(tokens), opposite_global))
        residual = features
        features = F.gelu(
            self.cbn_hidden(self.hidden(features), opposite_global)
        )
        return self.output(features + residual).squeeze(-1)


def _euler_xyz_rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Return batched Rz @ Ry @ Rx matrices for XYZ Euler samples."""

    ax, ay, az = angles.unbind(dim=-1)
    sx, sy, sz = torch.sin(ax), torch.sin(ay), torch.sin(az)
    cx, cy, cz = torch.cos(ax), torch.cos(ay), torch.cos(az)
    return torch.stack(
        (
            cy * cz,
            cz * sx * sy - cx * sz,
            sx * sz + cx * cz * sy,
            cy * sz,
            cx * cz + sx * sy * sz,
            cx * sy * sz - cz * sx,
            -sy,
            cy * sx,
            cx * cy,
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


class Pose9DHead(nn.Module):
    """Small MLP head that predicts translation + 6D rotation columns."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 128)):
        super().__init__()
        self.net = _make_relu_mlp((input_dim,) + hidden_dims + (9,))
        self._init_identity_delta_output()

    def _init_identity_delta_output(self) -> None:
        final = self.net[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Pose9DHead expects the final module to be nn.Linear")
        nn.init.zeros_(final.weight)
        identity_delta = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            dtype=final.bias.dtype,
            device=final.bias.device,
        )
        with torch.no_grad():
            final.bias.copy_(identity_delta)

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_features)


class TranslationHead(nn.Module):
    """Small MLP head that predicts an xyz translation delta."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 128)):
        super().__init__()
        self.net = _make_relu_mlp((input_dim,) + hidden_dims + (3,))
        self._init_zero_delta_output()

    def _init_zero_delta_output(self) -> None:
        final = self.net[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("TranslationHead expects the final module to be nn.Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_features)


def _pose9d_to_rotation_matrix(pose9d: torch.Tensor) -> torch.Tensor:
    return rotation_from_pose9d(pose9d)


def _pose9d_delta_magnitudes(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    trans_norm = target[..., :3].norm(dim=-1)
    target_R = _pose9d_to_rotation_matrix(target)
    trace = target_R[..., 0, 0] + target_R[..., 1, 1] + target_R[..., 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    rot_angle_deg = torch.acos(cos_angle) * (180.0 / math.pi)
    return trans_norm, rot_angle_deg


def _pose9d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    child_points: torch.Tensor,
    rot_weight: float,
    chamfer_weight: float,
    quat_norm_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_t = pred[..., :3]
    target_t = target[..., :3]
    pred_R = _pose9d_to_rotation_matrix(pred)
    target_R = _pose9d_to_rotation_matrix(target)

    if child_points.ndim != 3 or child_points.shape[0] != pred.shape[0] or child_points.shape[-1] != 3:
        raise ValueError(
            "RPDiff pose loss requires child_points with shape (B, N, 3), "
            f"got {tuple(child_points.shape)} for pred {tuple(pred.shape)}"
        )
    child_pred = child_points @ pred_R.transpose(-1, -2) + pred_t[:, None, :]
    child_target = child_points @ target_R.transpose(-1, -2) + target_t[:, None, :]
    rpdiff_loss = TransformChamferWrapper(l1=False, trans_offset=False)
    loss_dict = rpdiff_loss.tf_chamfer(
        model_outputs={
            "trans": pred_t,
            "quat": matrix_to_quaternion(pred_R),
            "unnorm_quat": matrix_to_quaternion(pred_R),
            "child_pcd_final_pred": child_pred,
        },
        ground_truth={
            "trans": target_t,
            "rot_mat": target_R,
            "child_final_pcd": child_target,
        },
        quat_norm_beta=quat_norm_beta,
    )
    trans_loss = loss_dict["trans"]
    rot_loss = loss_dict["rot"]
    chamfer_loss = loss_dict["chamf"]
    total = trans_loss + float(rot_weight) * rot_loss + float(chamfer_weight) * chamfer_loss
    return total, {
        "pose_trans_loss": trans_loss,
        "pose_rot_geodesic_loss": rot_loss,
        "pose_chamfer_loss": chamfer_loss,
    }


def _translation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = F.mse_loss(pred, target)
    return loss, {"translation_loss": loss}


def _sdf_supervision_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    relative: bool,
    eps: float,
) -> torch.Tensor:
    if not relative:
        return F.smooth_l1_loss(pred, target)
    denom = target.detach().abs().clamp_min(float(eps))
    return ((pred - target) / denom).abs().mean()


# ============================================================================ #
# ContactDiffusionModel — ties everything together
# ============================================================================ #

class ContactDiffusionModel(nn.Module):
    """Joint SDF + pose denoising model.

    Architecture:
      1. encoder: TCEPointCloudEncoder
           - Tool input:  tool_rotated  (canonical pts rotated to current/noised pose)
           - Object input: obj_pc       (centered at origin)
      2. pose_cross_attn: PoseCrossAttention — run once per head:
           - SDF pass:      (pose_3d, zeros_movement)
           - Denoise pass:  (pose_3d, zeros_movement) [sdf-diff only]
      3. SDF heads: per-point/per-patch SDF from SDF-conditioned tokens
      4. pose9d heads: separately pooled tool+object tokens → 9D pose deltas

    Both heads receive the SAME pose_3d = noised_tool_centroid - obj_centroid.
    Diffusion conditioning uses only the noised pose signal and timestep; post/physics
    movement conditioning is reserved for the postcontact head.
    """

    def __init__(
        self,
        # SDF head
        head_mode: str = "point",
        patch_agg: str = "min",
        head_hidden: tuple[int, ...] = (256, 128),
        # Encoder
        num_pts: int = 512,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 4,
        vit_heads: int = 4,
        vit_attention_mode: str | None = None,
        freeze_encoder: bool = False,
        kinematic_conditioning: bool = False,
        kinematic_attention_layers: int = 1,
        # Cross-attention
        cross_attn_heads: int = 4,
        cross_attn_layers: int = 2,
        pose_dim: int = 3,
        movement_cond_dim: int = 25,
        condition_mlp_hidden_dims: tuple[int, ...] = (128, 128),
        num_query_A: int = 4,
        num_query_B: int = 4,
        num_query_C: int = 4,
        num_query_D: int = 4,
        condition_mean: tuple[float, ...] | None = None,
        condition_std: tuple[float, ...] | None = None,
        condition_norm_eps: float = 1e-4,
        # Denoising
        denoise_hidden: tuple[int, ...] = (256,),
        postcontact_hidden: tuple[int, ...] = (256,),
        # Loss
        sdf_weight: float = 1.0,
        denoise_weight: float = 1.0,
        postcontact_weight: float = 1.0,
        denoise_rot_weight: float = 1.0,
        chamfer_weight: float = 1.0,
        quat_norm_beta: float = 0.1,
        loss_weights: dict[str, float] | None = None,
        # Diffusion
        num_diffusion_steps: int = 10,
        sdf_backend: str = "kaolin",
        sdf_chunk_size: int = 8192,
        sdf_relative_loss: bool = False,
        sdf_relative_eps: float = 0.005,
        encoder_input_centering: str = "bbox_center",
        contact_eps: float = 0.002,
        contact_label_source: str = "mesh_sdf",
        contact_positive_patch_fraction: float = 0.5,
        contact_patch_positive_rule: str = "any",
        contact_positive_min_points: int = 1,
        contact_decoder_type: str = "relu_mlp",
        contact_decoder_hidden: tuple[int, ...] = (128, 128),
        contact_pair_augmentation: bool = False,
        contact_aug_rotation_range: tuple[float, float] = (0.0, 0.0),
        contact_aug_translation_range: tuple[float, float] = (0.0, 0.0),
        contact_aug_log_scale_range: tuple[float, float] = (0.0, 0.0),
        contact_aug_noise_std: float = 0.0,
        # Task
        task: str = "sdf-diff",
        enabled_heads: tuple[str, ...] | list[str] | None = None,
    ):
        super().__init__()
        if vit_attention_mode not in {"joint_self", "cross_only"}:
            raise ValueError(
                "vit_attention_mode is required and must be joint_self or "
                f"cross_only, got {vit_attention_mode!r}"
            )
        assert head_mode in ("point", "patch")
        assert task in ("sdf", "sdf-diff")
        if enabled_heads is None:
            enabled_heads = ("sdf", "diff") if task == "sdf-diff" else ("sdf",)
        enabled_heads = tuple(enabled_heads)
        invalid_heads = sorted(set(enabled_heads).difference({"sdf", "diff", "postcontact", "contact"}))
        if invalid_heads:
            raise ValueError(f"Unknown enabled_heads: {invalid_heads}")

        self.head_mode = head_mode
        self.patch_agg = patch_agg
        self.task = task
        self.enabled_heads = enabled_heads
        self.sdf_backend = str(sdf_backend)
        self.sdf_chunk_size = int(sdf_chunk_size)
        self.sdf_relative_loss = bool(sdf_relative_loss)
        self.sdf_relative_eps = float(sdf_relative_eps)
        self.encoder_input_centering = str(encoder_input_centering)
        self.contact_eps = float(contact_eps)
        self.contact_label_source = str(contact_label_source)
        self.contact_positive_patch_fraction = float(contact_positive_patch_fraction)
        self.contact_patch_positive_rule = str(contact_patch_positive_rule)
        self.contact_positive_min_points = int(contact_positive_min_points)
        self.contact_decoder_type = str(contact_decoder_type)
        self.contact_decoder_hidden = tuple(int(v) for v in contact_decoder_hidden)
        self.contact_pair_augmentation = bool(contact_pair_augmentation)
        self.contact_aug_rotation_range = tuple(
            float(v) for v in contact_aug_rotation_range
        )
        self.contact_aug_translation_range = tuple(
            float(v) for v in contact_aug_translation_range
        )
        self.contact_aug_log_scale_range = tuple(
            float(v) for v in contact_aug_log_scale_range
        )
        self.contact_aug_noise_std = float(contact_aug_noise_std)
        self.kinematic_conditioning = bool(kinematic_conditioning)
        self.kinematic_attention_layers = int(kinematic_attention_layers)
        if self.kinematic_conditioning:
            if set(self.enabled_heads) != {"contact"}:
                raise ValueError(
                    "Kinematic conditioning is defined only for binary contact pretraining"
                )
            if self.contact_decoder_type != "paper_cmlp_cbn":
                raise ValueError(
                    "Kinematic conditioning requires contact_decoder_type='paper_cmlp_cbn'"
                )
        if self.encoder_input_centering not in {"bbox_center", "object_center"}:
            raise ValueError(
                "encoder_input_centering must be 'bbox_center' or "
                f"'object_center', got {self.encoder_input_centering!r}"
            )
        if self.sdf_relative_eps <= 0.0:
            raise ValueError("sdf_relative_eps must be > 0")
        if self.contact_eps < 0.0:
            raise ValueError("contact_eps must be >= 0")
        if self.contact_label_source not in {
            "mesh_sdf",
            "precomputed_convex_union",
            "precomputed_mesh_sdf",
        }:
            raise ValueError(
                "contact_label_source must be mesh_sdf, "
                "precomputed_convex_union, or precomputed_mesh_sdf"
            )
        if (
            "sdf" in self.enabled_heads
            and self.contact_label_source == "precomputed_mesh_sdf"
            and (
                self.contact_aug_log_scale_range != (0.0, 0.0)
                or self.contact_aug_noise_std != 0.0
            )
        ):
            raise ValueError(
                "Precomputed SDF regression requires zero scale augmentation "
                "and zero point jitter"
            )
        if not 0.0 < self.contact_positive_patch_fraction < 1.0:
            raise ValueError("contact_positive_patch_fraction must be in (0, 1)")
        if self.contact_patch_positive_rule not in {"any", "count"}:
            raise ValueError("contact_patch_positive_rule must be 'any' or 'count'")
        if self.contact_decoder_type not in {"relu_mlp", "paper_cmlp_cbn"}:
            raise ValueError(
                "contact_decoder_type must be relu_mlp or paper_cmlp_cbn"
            )
        merged_weights = {
            "sdf": float(sdf_weight),
            "diff": float(denoise_weight),
            "postcontact": float(postcontact_weight),
            "contact": 1.0,
        }
        if loss_weights:
            merged_weights.update({str(k): float(v) for k, v in loss_weights.items()})
        self.loss_weights = merged_weights
        self.sdf_weight = merged_weights["sdf"]
        self.denoise_weight = merged_weights["diff"]
        self.postcontact_weight = merged_weights["postcontact"]
        self.denoise_rot_weight = denoise_rot_weight
        self.chamfer_weight = chamfer_weight
        self.quat_norm_beta = quat_norm_beta
        self.num_diffusion_steps = num_diffusion_steps
        self.movement_cond_dim = int(movement_cond_dim)
        self.condition_norm_eps = float(condition_norm_eps)
        if self.condition_norm_eps <= 0.0:
            raise ValueError("condition_norm_eps must be > 0")
        if condition_mean is None or condition_std is None:
            condition_mean_tensor = torch.zeros(self.movement_cond_dim, dtype=torch.float32)
            condition_std_tensor = torch.ones(self.movement_cond_dim, dtype=torch.float32)
            condition_normalization_enabled = False
        else:
            condition_mean_tensor = torch.as_tensor(condition_mean, dtype=torch.float32)
            condition_std_tensor = torch.as_tensor(condition_std, dtype=torch.float32)
            if tuple(condition_mean_tensor.shape) != (self.movement_cond_dim,):
                raise ValueError(
                    f"condition_mean must have shape ({self.movement_cond_dim},), "
                    f"got {tuple(condition_mean_tensor.shape)}"
                )
            if tuple(condition_std_tensor.shape) != (self.movement_cond_dim,):
                raise ValueError(
                    f"condition_std must have shape ({self.movement_cond_dim},), "
                    f"got {tuple(condition_std_tensor.shape)}"
                )
            condition_normalization_enabled = True
        self.register_buffer("condition_mean", condition_mean_tensor, persistent=False)
        self.register_buffer("condition_std", condition_std_tensor.clamp_min(self.condition_norm_eps), persistent=False)
        self.register_buffer(
            "condition_normalization_enabled",
            torch.tensor(condition_normalization_enabled, dtype=torch.bool),
            persistent=False,
        )

        # ── Shared encoder ───────────────────────────────────────────────
        enc_cfg = TCEPointCloudEncoderCfg(
            num_pts=num_pts,
            patch_size=patch_size,
            encoder_channel=encoder_channel,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            freeze=freeze_encoder,
            vit_attention_mode=vit_attention_mode,
            kinematic_conditioning=self.kinematic_conditioning,
            kinematic_attention_layers=self.kinematic_attention_layers,
        )
        self.encoder = TCEPointCloudEncoder(enc_cfg)
        D = self.encoder.feature_dim
        self.num_patches = self.encoder.num_patches

        # ── SDF heads (same architecture as existing SDFSegmentor) ───────
        # Only create these when the SDF objective is active. Diff-only DDP
        # should not carry permanently unused SDF parameters.
        if "sdf" in self.enabled_heads:
            if head_mode == "point":
                self.xyz_embed = _make_mlp((3, D, D))
                self.tool_sdf_head = _make_mlp((2 * D,) + head_hidden + (1,))
                self.obj_sdf_head = _make_mlp((2 * D,) + head_hidden + (1,))
            else:
                self.tool_sdf_head = _make_mlp((D,) + head_hidden + (1,))
                self.obj_sdf_head = _make_mlp((D,) + head_hidden + (1,))

        # ── Pose conditioning (cross-attention) ─────────────────────────
        # SDF/diff/postcontact need pose conditioning. Contact-only TCE
        # pretraining keeps this module absent so DDP has no unused params.
        if {"sdf", "diff", "postcontact"}.intersection(self.enabled_heads):
            self.pose_cross_attn = PoseCrossAttention(
                token_dim=D,
                pose_dim=pose_dim,
                movement_cond_dim=movement_cond_dim,
                n_heads=cross_attn_heads,
                n_layers=cross_attn_layers,
                condition_mlp_hidden_dims=condition_mlp_hidden_dims,
                num_query_A=num_query_A,
                num_query_B=num_query_B,
                num_query_C=num_query_C,
                num_query_D=num_query_D,
            )

        if "diff" in self.enabled_heads:
            self.diff_time_emb = SinusoidalPosEmb(dim=2 * D, max_pos=num_diffusion_steps + 1)
            self.diff_translation_head = TranslationHead(
                input_dim=2 * D,
                hidden_dims=denoise_hidden,
            )
        if "postcontact" in self.enabled_heads:
            self.postcontact_head = _make_relu_mlp(
                (2 * D,) + postcontact_hidden + (9,)
            )
        if "contact" in self.enabled_heads:
            if self.kinematic_conditioning:
                self.openness_delta_embed = _make_relu_mlp((1, D, D))
            if self.contact_decoder_type == "paper_cmlp_cbn":
                contact_condition_dim = (
                    2 * D if self.kinematic_conditioning else D
                )
                self.tool_contact_head = _PaperContactCMLP(
                    D, self.contact_decoder_hidden, contact_condition_dim
                )
                self.obj_contact_head = _PaperContactCMLP(
                    D, self.contact_decoder_hidden, contact_condition_dim
                )
            else:
                self.tool_contact_head = _make_relu_mlp(
                    (D,) + head_hidden + (1,)
                )
                self.obj_contact_head = _make_relu_mlp(
                    (D,) + head_hidden + (1,)
                )

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

    def _patch_contact_labels(
        self,
        point_labels: torch.Tensor,
        patch_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, P, K = patch_idx.shape
        gathered = point_labels.gather(1, patch_idx.reshape(B, P * K)).view(B, P, K)
        if self.contact_patch_positive_rule == "count":
            return (gathered.sum(dim=-1) >= max(1, self.contact_positive_min_points)).to(point_labels.dtype)
        return gathered.any(dim=-1).to(point_labels.dtype)

    def _balanced_contact_bce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        pos = labels > 0.5
        neg = ~pos
        num_pos = pos.sum().to(raw.dtype)
        num_neg = neg.sum().to(raw.dtype)
        pos_mass = raw.new_tensor(self.contact_positive_patch_fraction)
        neg_mass = torch.where(num_pos > 0, raw.new_tensor(1.0) - pos_mass, raw.new_tensor(1.0))
        weights = (
            pos.to(raw.dtype) * pos_mass / num_pos.clamp_min(1.0)
            + neg.to(raw.dtype) * neg_mass / num_neg.clamp_min(1.0)
        )
        return (raw * weights).sum(), {"empty_positive_patch_count": float((num_pos <= 0).detach().cpu())}

    def _augment_contact_pair_inputs(
        self,
        tool_points: torch.Tensor,
        object_points: torch.Tensor,
        rel_tool_object_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the paper's shared rigid/scale augmentation plus point noise."""

        if not (self.training and self.contact_pair_augmentation):
            return tool_points, object_points, rel_tool_object_t
        batch = int(tool_points.shape[0])
        dtype = tool_points.dtype
        device = tool_points.device

        rotation_low, rotation_high = self.contact_aug_rotation_range
        angles = torch.empty(batch, 3, dtype=dtype, device=device).uniform_(
            rotation_low, rotation_high
        )
        rotation = _euler_xyz_rotation_matrices(angles)
        log_scale_low, log_scale_high = self.contact_aug_log_scale_range
        scale = torch.empty(batch, 1, 1, dtype=dtype, device=device).uniform_(
            log_scale_low, log_scale_high
        ).exp()
        translation_low, translation_high = self.contact_aug_translation_range
        translation = torch.empty(
            batch, 1, 3, dtype=dtype, device=device
        ).uniform_(translation_low, translation_high)

        tool_aug = (
            torch.einsum("bij,bnj->bni", rotation, tool_points) * scale
            + translation
        )
        object_aug = (
            torch.einsum("bij,bnj->bni", rotation, object_points) * scale
            + translation
        )
        rel_aug = (
            torch.einsum("bij,bj->bi", rotation, rel_tool_object_t)
            * scale.squeeze(-1)
        )
        if self.contact_aug_noise_std > 0.0:
            tool_aug = tool_aug + torch.randn_like(tool_aug) * self.contact_aug_noise_std
            object_aug = (
                object_aug
                + torch.randn_like(object_aug) * self.contact_aug_noise_std
            )
        return (
            tool_aug.contiguous(),
            object_aug.contiguous(),
            rel_aug.contiguous(),
        )

    # ── Forward (routes to loss for DDP) ──────────────────────────────────

    def forward(self, *args, **kwargs):
        """Route through loss() so DDP gradient sync hooks fire."""
        return self.loss(*args, **kwargs)

    def _compose_condition(
        self,
        cond_tool_post_delta9d: torch.Tensor | None,
        cond_object_post_delta9d: torch.Tensor | None,
        physics: torch.Tensor | None,
        *,
        include_object_delta: bool,
    ) -> torch.Tensor:
        if not include_object_delta:
            cond_object_post_delta9d = torch.zeros_like(cond_object_post_delta9d)
        cond = torch.cat((cond_tool_post_delta9d, cond_object_post_delta9d, physics), dim=-1)
        if cond.shape[-1] < self.movement_cond_dim:
            cond = F.pad(cond, (0, self.movement_cond_dim - cond.shape[-1]))
        elif cond.shape[-1] > self.movement_cond_dim:
            cond = cond[..., : self.movement_cond_dim]
        if bool(self.condition_normalization_enabled.item()):
            mean = self.condition_mean.to(device=cond.device, dtype=cond.dtype)
            std = self.condition_std.to(device=cond.device, dtype=cond.dtype).clamp_min(self.condition_norm_eps)
            cond = (cond - mean) / std
        return cond

    def _pool_conditioned_tokens(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        P = self.num_patches
        tool_cond = fused_tokens[:, :P, :]
        obj_cond = fused_tokens[:, P : 2 * P, :]
        return torch.cat((tool_cond.mean(dim=1), obj_cond.mean(dim=1)), dim=-1)

    def _signed_mesh_sdf_labels(
        self,
        *,
        tool_points_E_k: torch.Tensor,
        object_points_E_k: torch.Tensor,
        object_mesh_vertices,
        object_mesh_faces,
        tool_mesh_vertices,
        tool_mesh_faces,
        object_rotation_E: torch.Tensor,
        object_bbox_center_E: torch.Tensor,
        tool_rotation_E_k: torch.Tensor,
        tool_translation_E_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [
            name
            for name, value in {
                "object_mesh_vertices": object_mesh_vertices,
                "object_mesh_faces": object_mesh_faces,
                "tool_mesh_vertices": tool_mesh_vertices,
                "tool_mesh_faces": tool_mesh_faces,
                "object_rotation_E": object_rotation_E,
                "object_bbox_center_E": object_bbox_center_E,
                "tool_rotation_E_k": tool_rotation_E_k,
                "tool_translation_E_k": tool_translation_E_k,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"SDF head requires on-the-fly signed mesh SDF fields: missing {missing}")
        tool_query_points_E = tool_points_E_k + tool_translation_E_k.unsqueeze(-2)
        object_query_points_E = object_points_E_k + object_bbox_center_E[:, None, None, :]
        return mutual_signed_sdf_labels_env_frame(
            tool_query_points_E=tool_query_points_E,
            object_query_points_E=object_query_points_E,
            object_mesh_vertices=object_mesh_vertices,
            object_mesh_faces=object_mesh_faces,
            tool_mesh_vertices=tool_mesh_vertices,
            tool_mesh_faces=tool_mesh_faces,
            object_rotation_E=object_rotation_E,
            object_bbox_center_E=object_bbox_center_E,
            tool_rotation_E_k=tool_rotation_E_k,
            tool_translation_E_k=tool_translation_E_k,
            chunk_size=self.sdf_chunk_size,
            backend=self.sdf_backend,
        )

    def _loss_contact_v1(
        self,
        *,
        tool_points_E_k: torch.Tensor,
        object_points_E_k: torch.Tensor,
        rel_tool_object_t_k: torch.Tensor,
        cond_tool_post_delta9d: torch.Tensor,
        cond_object_post_delta9d: torch.Tensor,
        physics: torch.Tensor,
        object_mesh_vertices=None,
        object_mesh_faces=None,
        tool_mesh_vertices=None,
        tool_mesh_faces=None,
        object_rotation_E: torch.Tensor | None = None,
        object_bbox_center_E: torch.Tensor | None = None,
        tool_rotation_E_k: torch.Tensor | None = None,
        tool_translation_E_k: torch.Tensor | None = None,
        tool_point_inside_object: torch.Tensor | None = None,
        object_point_inside_tool: torch.Tensor | None = None,
        tool_point_object_signed_sdf: torch.Tensor | None = None,
        object_point_tool_signed_sdf: torch.Tensor | None = None,
        kinematic_tool_clouds: torch.Tensor | None = None,
        openness_delta: torch.Tensor | None = None,
        target_tool_denoise_pose9d_k: torch.Tensor | None = None,
        target_object_post_delta9d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, T, N, _ = tool_points_E_k.shape
        if self.kinematic_conditioning:
            if T != 1:
                raise ValueError(
                    "Kinematic contact pretraining requires one contact state per sample"
                )
            if kinematic_tool_clouds is None or openness_delta is None:
                raise ValueError(
                    "Kinematic contact pretraining requires kinematic_tool_clouds "
                    "and openness_delta"
                )
            if tuple(openness_delta.shape) != (B,):
                raise ValueError(
                    f"openness_delta must have shape ({B},), got "
                    f"{tuple(openness_delta.shape)}"
                )
        elif kinematic_tool_clouds is not None or openness_delta is not None:
            raise ValueError(
                "Kinematic inputs were supplied to a standard contact model"
            )
        device = tool_points_E_k.device
        K = max(T - 1, 0)
        encoder_tool_points_E_k = tool_points_E_k
        encoder_object_points_E_k = object_points_E_k
        if self.encoder_input_centering == "object_center":
            if rel_tool_object_t_k is None:
                raise ValueError("object_center encoder input requires rel_tool_object_t_k")
            encoder_tool_points_E_k = tool_points_E_k + rel_tool_object_t_k.unsqueeze(-2)
        tool_flat = encoder_tool_points_E_k.reshape(B * T, N, 3)
        obj_flat = encoder_object_points_E_k.reshape(B * T, N, 3)
        rel_flat = rel_tool_object_t_k.reshape(B * T, 3)
        if (
            "contact" in self.enabled_heads
            or (
                "sdf" in self.enabled_heads
                and self.contact_label_source == "precomputed_mesh_sdf"
            )
        ):
            tool_flat, obj_flat, rel_flat = self._augment_contact_pair_inputs(
                tool_flat, obj_flat, rel_flat
            )

        encoder_result = self.encoder.encode(
            tool_flat,
            obj_flat,
            kinematic_tool_clouds=kinematic_tool_clouds,
        )
        fused = encoder_result.fused_tokens
        metrics: dict[str, float] = {}
        total_loss = tool_flat.new_zeros(())

        if "contact" in self.enabled_heads:
            contact_slice = torch.arange(B * T, device=device).reshape(B, T)[:, 0]
            P = self.num_patches
            contact_fused = fused.index_select(0, contact_slice)
            tool_contact_tok = contact_fused[:, :P, :]
            obj_contact_tok = contact_fused[:, P : 2 * P, :]
            if self.contact_label_source == "precomputed_convex_union":
                if (
                    tool_point_inside_object is None
                    or object_point_inside_tool is None
                ):
                    raise ValueError(
                        "precomputed_convex_union contact labels are required "
                        "but missing from the candidate dataset"
                    )
                tool_contact_points = tool_point_inside_object.to(
                    device=device, dtype=tool_flat.dtype
                )
                obj_contact_points = object_point_inside_tool.to(
                    device=device, dtype=tool_flat.dtype
                )
            elif self.contact_label_source == "precomputed_mesh_sdf":
                if (
                    tool_point_object_signed_sdf is None
                    or object_point_tool_signed_sdf is None
                ):
                    raise ValueError(
                        "precomputed_mesh_sdf contact labels are required "
                        "but missing from the candidate dataset"
                    )
                tool_contact_points = (
                    tool_point_object_signed_sdf.to(device=device)
                    <= self.contact_eps
                ).to(tool_flat.dtype)
                obj_contact_points = (
                    object_point_tool_signed_sdf.to(device=device)
                    <= self.contact_eps
                ).to(tool_flat.dtype)
            else:
                tool_sdf_gt_full, obj_sdf_gt_full = self._signed_mesh_sdf_labels(
                    tool_points_E_k=tool_points_E_k,
                    object_points_E_k=object_points_E_k,
                    object_mesh_vertices=object_mesh_vertices,
                    object_mesh_faces=object_mesh_faces,
                    tool_mesh_vertices=tool_mesh_vertices,
                    tool_mesh_faces=tool_mesh_faces,
                    object_rotation_E=object_rotation_E,
                    object_bbox_center_E=object_bbox_center_E,
                    tool_rotation_E_k=tool_rotation_E_k,
                    tool_translation_E_k=tool_translation_E_k,
                )
                tool_contact_points = (
                    tool_sdf_gt_full[:, 0, :] <= self.contact_eps
                ).to(tool_flat.dtype)
                obj_contact_points = (
                    obj_sdf_gt_full[:, 0, :] <= self.contact_eps
                ).to(tool_flat.dtype)
            tool_patch_idx = encoder_result.tool_patch_idx.index_select(0, contact_slice)
            obj_patch_idx = encoder_result.obj_patch_idx.index_select(0, contact_slice)
            tool_contact_labels = self._patch_contact_labels(tool_contact_points, tool_patch_idx)
            obj_contact_labels = self._patch_contact_labels(obj_contact_points, obj_patch_idx)
            if self.contact_decoder_type == "paper_cmlp_cbn":
                if self.kinematic_conditioning:
                    delta_context = self.openness_delta_embed(
                        openness_delta.unsqueeze(-1).to(tool_contact_tok)
                    )
                    tool_context = torch.cat(
                        (obj_contact_tok.mean(dim=1), delta_context), dim=-1
                    )
                    object_context = torch.cat(
                        (tool_contact_tok.mean(dim=1), delta_context), dim=-1
                    )
                else:
                    tool_context = obj_contact_tok.mean(dim=1)
                    object_context = tool_contact_tok.mean(dim=1)
                tool_contact_logits = self.tool_contact_head(
                    tool_contact_tok,
                    tool_context,
                )
                obj_contact_logits = self.obj_contact_head(
                    obj_contact_tok,
                    object_context,
                )
            else:
                tool_contact_logits = self.tool_contact_head(
                    tool_contact_tok
                ).squeeze(-1)
                obj_contact_logits = self.obj_contact_head(
                    obj_contact_tok
                ).squeeze(-1)
            tool_bce, tool_bce_meta = self._balanced_contact_bce(tool_contact_logits, tool_contact_labels)
            obj_bce, obj_bce_meta = self._balanced_contact_bce(obj_contact_logits, obj_contact_labels)
            contact_loss = tool_bce + obj_bce
            total_loss = total_loss + self.loss_weights["contact"] * contact_loss

            with torch.no_grad():
                logits_all = torch.cat((tool_contact_logits, obj_contact_logits), dim=1)
                labels_all = torch.cat((tool_contact_labels, obj_contact_labels), dim=1)
                pred_all = logits_all.sigmoid() >= 0.5
                true_all = labels_all >= 0.5
                true_pos = (pred_all & true_all).sum().float()
                pred_pos = pred_all.sum().float()
                label_pos = true_all.sum().float()
                total = true_all.numel()
                metrics["contact_acc"] = float((pred_all == true_all).float().mean().detach().cpu())
                metrics["contact_precision"] = float((true_pos / pred_pos.clamp_min(1.0)).detach().cpu())
                metrics["contact_recall"] = float((true_pos / label_pos.clamp_min(1.0)).detach().cpu())
                metrics["patch_pos_frac_A"] = float(tool_contact_labels.mean().detach().cpu())
                metrics["patch_pos_frac_B"] = float(obj_contact_labels.mean().detach().cpu())
                metrics["empty_positive_patch_count"] = (
                    tool_bce_meta["empty_positive_patch_count"] + obj_bce_meta["empty_positive_patch_count"]
                )
                metrics["contact_positive_patches"] = float(label_pos.detach().cpu())
                metrics["contact_total_patches"] = float(total)
                if self.kinematic_conditioning:
                    metrics["openness_delta_mean"] = float(
                        openness_delta.mean().detach().cpu()
                    )
                    metrics["openness_delta_abs_mean"] = float(
                        openness_delta.abs().mean().detach().cpu()
                    )
            metrics["bce_A"] = float(tool_bce.detach().cpu())
            metrics["bce_B"] = float(obj_bce.detach().cpu())
            metrics["contact_loss"] = float(contact_loss.detach().cpu())

        if "sdf" in self.enabled_heads:
            zero_cond = torch.zeros(B * T, self.movement_cond_dim, device=device, dtype=tool_flat.dtype)
            fused_sdf = self.pose_cross_attn(fused, rel_flat, zero_cond)
            P = self.num_patches
            tool_tok_cond = fused_sdf[:, :P, :]
            obj_tok_cond = fused_sdf[:, P : 2 * P, :]
            if self.contact_label_source == "precomputed_mesh_sdf":
                if T != 1:
                    raise ValueError(
                        "Precomputed SDF regression requires exactly one "
                        "contact state per sample"
                    )
                if (
                    tool_point_object_signed_sdf is None
                    or object_point_tool_signed_sdf is None
                ):
                    raise ValueError(
                        "Precomputed SDF regression requires mutual signed "
                        "distance arrays in every candidate"
                    )
                tool_sdf_gt_full = tool_point_object_signed_sdf.to(
                    device=device, dtype=tool_flat.dtype
                ).unsqueeze(1)
                obj_sdf_gt_full = object_point_tool_signed_sdf.to(
                    device=device, dtype=tool_flat.dtype
                ).unsqueeze(1)
            else:
                tool_sdf_gt_full, obj_sdf_gt_full = self._signed_mesh_sdf_labels(
                    tool_points_E_k=tool_points_E_k,
                    object_points_E_k=object_points_E_k,
                    object_mesh_vertices=object_mesh_vertices,
                    object_mesh_faces=object_mesh_faces,
                    tool_mesh_vertices=tool_mesh_vertices,
                    tool_mesh_faces=tool_mesh_faces,
                    object_rotation_E=object_rotation_E,
                    object_bbox_center_E=object_bbox_center_E,
                    tool_rotation_E_k=tool_rotation_E_k,
                    tool_translation_E_k=tool_translation_E_k,
                )
            tool_sdf_gt = tool_sdf_gt_full.reshape(B * T, N)
            obj_sdf_gt = obj_sdf_gt_full.reshape(B * T, N)

            if self.head_mode == "point":
                tool_sdf_pred = self._predict_point_sdf(
                    tool_flat,
                    tool_tok_cond,
                    encoder_result.tool_patch_idx,
                    encoder_result.tool_patch_centers,
                    self.tool_sdf_head,
                )
                obj_sdf_pred = self._predict_point_sdf(
                    obj_flat,
                    obj_tok_cond,
                    encoder_result.obj_patch_idx,
                    encoder_result.obj_patch_centers,
                    self.obj_sdf_head,
                )
            else:
                tool_sdf_gt = _aggregate_sdf(tool_sdf_gt, encoder_result.tool_patch_idx, self.patch_agg)
                obj_sdf_gt = _aggregate_sdf(obj_sdf_gt, encoder_result.obj_patch_idx, self.patch_agg)
                tool_sdf_pred = self._predict_patch_sdf(tool_tok_cond, self.tool_sdf_head)
                obj_sdf_pred = self._predict_patch_sdf(obj_tok_cond, self.obj_sdf_head)

            tool_sdf_loss = _sdf_supervision_loss(
                tool_sdf_pred,
                tool_sdf_gt,
                relative=self.sdf_relative_loss,
                eps=self.sdf_relative_eps,
            )
            obj_sdf_loss = _sdf_supervision_loss(
                obj_sdf_pred,
                obj_sdf_gt,
                relative=self.sdf_relative_loss,
                eps=self.sdf_relative_eps,
            )
            sdf_loss = tool_sdf_loss + obj_sdf_loss
            total_loss = total_loss + self.loss_weights["sdf"] * sdf_loss
            metrics["tool_sdf_loss"] = float(tool_sdf_loss.detach().cpu())
            metrics["obj_sdf_loss"] = float(obj_sdf_loss.detach().cpu())
            metrics["sdf_loss"] = float(sdf_loss.detach().cpu())

        if "diff" in self.enabled_heads and K > 0:
            if target_tool_denoise_pose9d_k is None:
                raise ValueError("diff head requires target_tool_denoise_pose9d_k")
            pre_slice = torch.arange(B * T, device=device).reshape(B, T)[:, 1:].reshape(-1)
            diff_fused = fused.index_select(0, pre_slice)
            diff_rel = rel_tool_object_t_k[:, 1:, :].reshape(B * K, 3)
            diff_time = torch.arange(1, T, device=device).repeat(B)
            diff_cond = torch.zeros(B * K, self.movement_cond_dim, device=device, dtype=tool_flat.dtype)
            diff_tokens = self.pose_cross_attn(diff_fused, diff_rel, diff_cond)
            diff_head_input = self._pool_conditioned_tokens(diff_tokens) + self.diff_time_emb(diff_time.float())
            diff_pred = self.diff_translation_head(diff_head_input)
            diff_target = target_tool_denoise_pose9d_k.reshape(B * K, 9)
            diff_loss, diff_parts = _translation_loss(
                diff_pred,
                diff_target[..., :3],
            )
            total_loss = total_loss + self.loss_weights["diff"] * diff_loss
            metrics["denoise_loss"] = float(diff_loss.detach().cpu())
            metrics["denoise_translation_loss"] = float(diff_parts["translation_loss"].detach().cpu())
            metrics["denoise_pose_trans_loss"] = metrics["denoise_translation_loss"]

        if "postcontact" in self.enabled_heads:
            if target_object_post_delta9d is None:
                raise ValueError("postcontact head requires target_object_post_delta9d")
            if cond_tool_post_delta9d is None or cond_object_post_delta9d is None or physics is None:
                raise ValueError("postcontact head requires post delta and physics conditioning fields")
            contact_slice = torch.arange(B * T, device=device).reshape(B, T)[:, 0]
            post_fused = fused.index_select(0, contact_slice)
            post_rel = rel_tool_object_t_k[:, 0, :]
            post_cond = self._compose_condition(
                cond_tool_post_delta9d,
                cond_object_post_delta9d,
                physics,
                include_object_delta=False,
            )
            post_tokens = self.pose_cross_attn(post_fused, post_rel, post_cond)
            post_pred = self.postcontact_head(self._pool_conditioned_tokens(post_tokens))
            post_child_points = object_points_E_k[:, 0, :, :]
            gt_trans_mag, gt_rot_mag_deg = _pose9d_delta_magnitudes(target_object_post_delta9d)
            pred_trans_mag, pred_rot_mag_deg = _pose9d_delta_magnitudes(post_pred)
            post_loss, post_parts = _pose9d_loss(
                post_pred,
                target_object_post_delta9d,
                child_points=post_child_points,
                rot_weight=self.denoise_rot_weight,
                chamfer_weight=self.chamfer_weight,
                quat_norm_beta=self.quat_norm_beta,
            )
            total_loss = total_loss + self.loss_weights["postcontact"] * post_loss
            metrics["postcontact_loss"] = float(post_loss.detach().cpu())
            metrics["postcontact_gt_translation_abs_mean"] = float(gt_trans_mag.mean().detach().cpu())
            metrics["postcontact_gt_rotation_abs_deg_mean"] = float(gt_rot_mag_deg.mean().detach().cpu())
            metrics["postcontact_pred_translation_abs_mean"] = float(pred_trans_mag.mean().detach().cpu())
            metrics["postcontact_pred_rotation_abs_deg_mean"] = float(pred_rot_mag_deg.mean().detach().cpu())
            metrics["postcontact_pose_trans_loss"] = float(post_parts["pose_trans_loss"].detach().cpu())
            metrics["postcontact_pose_rot_geodesic_loss"] = float(post_parts["pose_rot_geodesic_loss"].detach().cpu())
            metrics["postcontact_pose_chamfer_loss"] = float(post_parts["pose_chamfer_loss"].detach().cpu())

        if not metrics:
            raise ValueError("At least one enabled head must produce a loss")
        metrics["total_loss"] = float(total_loss.detach().cpu())
        return total_loss, metrics

    # ── Joint loss computation ────────────────────────────────────────────

    def loss(
        self,
        tool_points_E_k: torch.Tensor | None = None,
        object_points_E_k: torch.Tensor | None = None,
        rel_tool_object_t_k: torch.Tensor | None = None,
        cond_tool_post_delta9d: torch.Tensor | None = None,
        cond_object_post_delta9d: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        object_mesh_vertices=None,
        object_mesh_faces=None,
        tool_mesh_vertices=None,
        tool_mesh_faces=None,
        object_rotation_E: torch.Tensor | None = None,
        object_bbox_center_E: torch.Tensor | None = None,
        tool_rotation_E_k: torch.Tensor | None = None,
        tool_translation_E_k: torch.Tensor | None = None,
        tool_point_inside_object: torch.Tensor | None = None,
        object_point_inside_tool: torch.Tensor | None = None,
        tool_point_object_signed_sdf: torch.Tensor | None = None,
        object_point_tool_signed_sdf: torch.Tensor | None = None,
        kinematic_tool_clouds: torch.Tensor | None = None,
        openness_delta: torch.Tensor | None = None,
        target_tool_denoise_pose9d_k: torch.Tensor | None = None,
        target_object_post_delta9d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute contact_pt_env_v1 pretrain losses from schema dataset batches."""
        if tool_points_E_k is not None:
            return self._loss_contact_v1(
                tool_points_E_k=tool_points_E_k,
                object_points_E_k=object_points_E_k,
                rel_tool_object_t_k=rel_tool_object_t_k,
                cond_tool_post_delta9d=cond_tool_post_delta9d,
                cond_object_post_delta9d=cond_object_post_delta9d,
                physics=physics,
                object_mesh_vertices=object_mesh_vertices,
                object_mesh_faces=object_mesh_faces,
                tool_mesh_vertices=tool_mesh_vertices,
                tool_mesh_faces=tool_mesh_faces,
                object_rotation_E=object_rotation_E,
                object_bbox_center_E=object_bbox_center_E,
                tool_rotation_E_k=tool_rotation_E_k,
                tool_translation_E_k=tool_translation_E_k,
                tool_point_inside_object=tool_point_inside_object,
                object_point_inside_tool=object_point_inside_tool,
                tool_point_object_signed_sdf=tool_point_object_signed_sdf,
                object_point_tool_signed_sdf=object_point_tool_signed_sdf,
                kinematic_tool_clouds=kinematic_tool_clouds,
                openness_delta=openness_delta,
                target_tool_denoise_pose9d_k=target_tool_denoise_pose9d_k,
                target_object_post_delta9d=target_object_post_delta9d,
            )

        raise NotImplementedError(
            "Legacy pretrain loss path is disabled. Use contact_pt_env_v1 batches from "
            "pretrain.dataset and configure diff/postcontact pose heads explicitly."
        )
