"""UniCORN contact-patch pretraining model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geometry.sdf import mutual_signed_sdf_labels_env_frame


@dataclass
class UnicornGeometryEncoderCfg:
    num_points: int = 512
    num_patches: int = 16
    patch_size: int = 32
    encoder_channel: int = 128
    vit_depth: int = 4
    vit_heads: int = 4


class UnicornEncodeResult(NamedTuple):
    patch_tokens: torch.Tensor
    global_token: torch.Tensor
    patch_idx: torch.Tensor
    patch_centers: torch.Tensor


class PatchTokenizer(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.proj = nn.Sequential(nn.Linear(256, out_dim), nn.LayerNorm(out_dim))

    def forward(self, patch_points: torch.Tensor) -> torch.Tensor:
        x = self.net(patch_points)
        pooled = torch.cat((x.max(dim=2).values, x.mean(dim=2)), dim=-1)
        return self.proj(pooled)


class SinusoidalPointPosEmbed(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        if out_dim < 6:
            raise ValueError("SinusoidalPointPosEmbed requires out_dim >= 6")
        self.out_dim = int(out_dim)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        dtype = xyz.dtype
        device = xyz.device
        bands = max(1, self.out_dim // 6)
        freq = torch.exp(
            torch.linspace(0.0, 1.0, bands, device=device, dtype=dtype)
            * torch.log(torch.tensor(10000.0, device=device, dtype=dtype))
        )
        angles = xyz.unsqueeze(-1) * freq
        emb = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        if emb.shape[-1] < self.out_dim:
            emb = F.pad(emb, (0, self.out_dim - emb.shape[-1]))
        return emb[..., : self.out_dim]


class UnicornGeometryEncoder(nn.Module):
    """Single-cloud patch transformer encoder used Siamese-style by UniCORN."""

    def __init__(self, cfg: UnicornGeometryEncoderCfg):
        super().__init__()
        self.cfg = cfg
        self.patch_tokenizer = PatchTokenizer(cfg.encoder_channel)
        self.pos_embed = SinusoidalPointPosEmbed(cfg.encoder_channel)
        self.emb_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder_channel))
        nn.init.normal_(self.emb_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.encoder_channel,
            nhead=cfg.vit_heads,
            dim_feedforward=cfg.encoder_channel * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.vit_depth)
        self.norm = nn.LayerNorm(cfg.encoder_channel)

    @property
    def feature_dim(self) -> int:
        return self.cfg.encoder_channel

    @property
    def num_patches(self) -> int:
        return self.cfg.num_patches

    def forward(self, points: torch.Tensor) -> UnicornEncodeResult:
        B, N, _ = points.shape
        P = min(int(self.cfg.num_patches), N)
        centers_idx = self._fps_indices(points, P)
        batch = torch.arange(B, device=points.device).view(B, 1)
        centers = points[batch, centers_idx]
        patch_idx = self._knn_patch_indices(points, centers)
        batch_patch = torch.arange(B, device=points.device).view(B, 1, 1)
        patches = points[batch_patch, patch_idx]
        patch_tokens = self.patch_tokenizer(patches - centers.unsqueeze(2))
        patch_tokens = patch_tokens + self.pos_embed(centers)
        if P < self.cfg.num_patches:
            patch_tokens, patch_idx, centers = self._pad_patches(patch_tokens, patch_idx, centers)
        emb = self.emb_token.expand(B, -1, -1)
        tokens = torch.cat((patch_tokens, emb), dim=1)
        encoded = self.norm(self.encoder(tokens))
        return UnicornEncodeResult(
            patch_tokens=encoded[:, : self.cfg.num_patches, :],
            global_token=encoded[:, self.cfg.num_patches, :],
            patch_idx=patch_idx,
            patch_centers=centers,
        )

    def _fps_indices(self, points: torch.Tensor, num_centers: int) -> torch.Tensor:
        B, N, _ = points.shape
        centroids = torch.zeros(B, num_centers, dtype=torch.long, device=points.device)
        distance = torch.full((B, N), float("inf"), device=points.device, dtype=points.dtype)
        farthest = torch.zeros(B, dtype=torch.long, device=points.device)
        batch = torch.arange(B, dtype=torch.long, device=points.device)
        for i in range(num_centers):
            centroids[:, i] = farthest
            centroid = points[batch, farthest].view(B, 1, 3)
            dist = ((points - centroid) ** 2).sum(dim=-1)
            distance = torch.minimum(distance, dist)
            farthest = distance.max(dim=1).indices
        return centroids

    def _knn_patch_indices(self, points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        B, N, _ = points.shape
        k_eff = min(int(self.cfg.patch_size), N)
        idx = torch.cdist(centers, points).topk(k=k_eff, dim=-1, largest=False).indices
        if k_eff < self.cfg.patch_size:
            pad = idx[..., -1:].expand(B, centers.shape[1], self.cfg.patch_size - k_eff)
            idx = torch.cat((idx, pad), dim=-1)
        return idx

    def _pad_patches(
        self,
        tokens: torch.Tensor,
        idx: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, D = tokens.shape
        pad_p = self.cfg.num_patches - P
        tokens = torch.cat((tokens, tokens[:, -1:, :].expand(B, pad_p, D)), dim=1)
        idx = torch.cat((idx, idx[:, -1:, :].expand(B, pad_p, idx.shape[-1])), dim=1)
        centers = torch.cat((centers, centers[:, -1:, :].expand(B, pad_p, 3)), dim=1)
        return tokens, idx, centers


class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.to_affine = nn.Linear(cond_dim, num_features * 2)
        nn.init.zeros_(self.to_affine.weight)
        with torch.no_grad():
            self.to_affine.bias[:num_features].fill_(1.0)
            self.to_affine.bias[num_features:].zero_()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, P, H = x.shape
        y = self.bn(x.reshape(B * P, H)).view(B, P, H)
        gamma, beta = self.to_affine(cond).chunk(2, dim=-1)
        return y * gamma.unsqueeze(1) + beta.unsqueeze(1)


class CMLPResidualBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.cbn = ConditionalBatchNorm1d(dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = self.cbn(y, cond)
        return x + F.gelu(y)


class UnicornContactDecoder(nn.Module):
    def __init__(self, token_dim: int, hidden_dims: tuple[int, ...] = (128, 128)):
        super().__init__()
        hidden = int(hidden_dims[0]) if hidden_dims else token_dim
        self.input = nn.Linear(token_dim, hidden)
        self.blocks = nn.ModuleList(CMLPResidualBlock(hidden, token_dim) for _ in range(3))
        self.output = nn.Linear(hidden, 1)

    def forward(self, local_tokens: torch.Tensor, global_token: torch.Tensor) -> torch.Tensor:
        x = self.input(local_tokens)
        for block in self.blocks:
            x = block(x, global_token)
        return self.output(x).squeeze(-1)


class UnicornPretrainModel(nn.Module):
    def __init__(
        self,
        *,
        num_points: int = 512,
        num_patches: int = 16,
        patch_size: int = 32,
        encoder_channel: int = 128,
        vit_depth: int = 4,
        vit_heads: int = 4,
        decoder_hidden_dims: tuple[int, ...] = (128, 128),
        positive_patch_fraction: float = 0.5,
        patch_positive_rule: str = "any",
        positive_min_points: int = 1,
        label_backend: str = "kaolin",
        contact_eps: float = 0.002,
        label_chunk_size: int = 8192,
    ):
        super().__init__()
        self.model_family = "unicorn"
        self.positive_patch_fraction = float(positive_patch_fraction)
        self.patch_positive_rule = str(patch_positive_rule)
        self.positive_min_points = int(positive_min_points)
        self.label_backend = str(label_backend)
        self.contact_eps = float(contact_eps)
        self.label_chunk_size = int(label_chunk_size)
        self.encoder = UnicornGeometryEncoder(
            UnicornGeometryEncoderCfg(
                num_points=num_points,
                num_patches=num_patches,
                patch_size=patch_size,
                encoder_channel=encoder_channel,
                vit_depth=vit_depth,
                vit_heads=vit_heads,
            )
        )
        self.decoder = UnicornContactDecoder(encoder_channel, decoder_hidden_dims)
        self.num_patches = num_patches
        self.enabled_heads = ("contact",)
        self.loss_weights = {"contact": 1.0}

    def forward(
        self,
        *,
        points_A: torch.Tensor,
        points_B: torch.Tensor,
        label_points_A_E: torch.Tensor,
        label_points_B_E: torch.Tensor,
        object_mesh_vertices,
        object_mesh_faces,
        tool_mesh_vertices,
        tool_mesh_faces,
        object_rotation_E: torch.Tensor,
        object_bbox_center_E: torch.Tensor,
        tool_rotation_E_k: torch.Tensor,
        tool_translation_E_k: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        enc_A = self.encoder(points_A)
        enc_B = self.encoder(points_B)
        labels_A_point, labels_B_point = self._point_contact_labels(
            label_points_A_E=label_points_A_E,
            label_points_B_E=label_points_B_E,
            object_mesh_vertices=object_mesh_vertices,
            object_mesh_faces=object_mesh_faces,
            tool_mesh_vertices=tool_mesh_vertices,
            tool_mesh_faces=tool_mesh_faces,
            object_rotation_E=object_rotation_E,
            object_bbox_center_E=object_bbox_center_E,
            tool_rotation_E_k=tool_rotation_E_k,
            tool_translation_E_k=tool_translation_E_k,
        )
        labels_A_patch = self._patch_labels(labels_A_point, enc_A.patch_idx)
        labels_B_patch = self._patch_labels(labels_B_point, enc_B.patch_idx)

        logits_A = self.decoder(enc_A.patch_tokens, enc_B.global_token)
        logits_B = self.decoder(enc_B.patch_tokens, enc_A.global_token)
        loss_A, stats_A = self._balanced_bce(logits_A, labels_A_patch)
        loss_B, stats_B = self._balanced_bce(logits_B, labels_B_patch)
        loss = loss_A + loss_B

        with torch.no_grad():
            logits = torch.cat((logits_A, logits_B), dim=1)
            labels = torch.cat((labels_A_patch, labels_B_patch), dim=1)
            pred = logits.sigmoid() >= 0.5
            target = labels >= 0.5
            tp = (pred & target).sum().float()
            fp = (pred & ~target).sum().float()
            fn = (~pred & target).sum().float()
            acc = (pred == target).float().mean()
            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / (tp + fn).clamp_min(1.0)

        metrics = {
            "total_loss": float(loss.detach().cpu()),
            "contact_loss": float(loss.detach().cpu()),
            "bce_A": float(loss_A.detach().cpu()),
            "bce_B": float(loss_B.detach().cpu()),
            "patch_pos_frac_A": float(labels_A_patch.mean().detach().cpu()),
            "patch_pos_frac_B": float(labels_B_patch.mean().detach().cpu()),
            "empty_positive_patch_count": float(stats_A["empty_pos"] + stats_B["empty_pos"]),
            "contact_acc": float(acc.detach().cpu()),
            "contact_precision": float(precision.detach().cpu()),
            "contact_recall": float(recall.detach().cpu()),
        }
        return loss, metrics

    def _point_contact_labels(
        self,
        *,
        label_points_A_E: torch.Tensor,
        label_points_B_E: torch.Tensor,
        object_mesh_vertices,
        object_mesh_faces,
        tool_mesh_vertices,
        tool_mesh_faces,
        object_rotation_E: torch.Tensor,
        object_bbox_center_E: torch.Tensor,
        tool_rotation_E_k: torch.Tensor,
        tool_translation_E_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tool_sdf, object_sdf = mutual_signed_sdf_labels_env_frame(
                tool_query_points_E=label_points_A_E.unsqueeze(1),
                object_query_points_E=label_points_B_E.unsqueeze(1),
                object_mesh_vertices=object_mesh_vertices,
                object_mesh_faces=object_mesh_faces,
                tool_mesh_vertices=tool_mesh_vertices,
                tool_mesh_faces=tool_mesh_faces,
                object_rotation_E=object_rotation_E,
                object_bbox_center_E=object_bbox_center_E,
                tool_rotation_E_k=tool_rotation_E_k,
                tool_translation_E_k=tool_translation_E_k,
                chunk_size=self.label_chunk_size,
                backend=self.label_backend,
            )
            labels_A = (tool_sdf[:, 0, :] <= self.contact_eps).to(dtype=label_points_A_E.dtype)
            labels_B = (object_sdf[:, 0, :] <= self.contact_eps).to(dtype=label_points_B_E.dtype)
        return labels_A, labels_B

    def _patch_labels(self, point_labels: torch.Tensor, patch_idx: torch.Tensor) -> torch.Tensor:
        B, P, K = patch_idx.shape
        labels = point_labels.gather(1, patch_idx.reshape(B, P * K)).view(B, P, K)
        if self.patch_positive_rule == "count":
            return (labels.sum(dim=-1) >= max(1, self.positive_min_points)).float()
        return labels.any(dim=-1).float()

    def _balanced_bce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        pos = labels > 0.5
        neg = ~pos
        num_pos = pos.sum().to(dtype=logits.dtype)
        num_neg = neg.sum().to(dtype=logits.dtype)
        f = torch.as_tensor(self.positive_patch_fraction, device=logits.device, dtype=logits.dtype)
        neg_mass = torch.where(num_pos > 0, 1.0 - f, torch.ones_like(f))
        weights = (
            pos.to(dtype=logits.dtype) * f / num_pos.clamp_min(1.0)
            + neg.to(dtype=logits.dtype) * neg_mass / num_neg.clamp_min(1.0)
        )
        empty_pos = float((num_pos <= 0).detach().cpu())
        return (raw * weights).sum(), {"empty_pos": empty_pos}
