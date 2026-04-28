"""sdf_encoder.py — Joint ViT Point Cloud Encoder (geometry pretraining + RL backbone).

Designed to replace ICPNet in the SDF segmentation pretraining pipeline and to serve
as the geometry backbone in the downstream RL policy (ActorCriticMomentum-style).

Architecture
────────────

  Input: tool_pc [B, N, 3]  +  obj_pc [B, N, 3]

  1. Concatenate           →  [B, 2N, 3]
  2. Per-group FPS + KNN   →  patches [B, 2P, K, 3]    (FPS stays within each N-point group)
  3. PointNetPatchEncoder  →  patch tokens [B, 2P, D]   (max-pool + mean-pool, perm-invariant)
  4. Positional embedding  →  add patch-centre pos enc
  5. Type embedding        →  add learnable (tool=0 / object=1) per-patch embedding
  6. CLS token prepended   →  [B, 1+2P, D]
  7. Joint ViT             →  self-attn over all tokens (implicit cross-stream reasoning)
  8. Strip CLS token       →  fused_tokens [B, 2P, D] (no global_feat returned)

Token ordering after ViT: [tool_patches (P), obj_patches (P)]
First P tokens are tool patches, last P tokens are object patches.

For RL use: freeze the encoder, consume fused_tokens (all patch tokens) in the
actor-critic's feature-fusion stage (e.g. SD-Cross attention with robot state).
Cross-attention fusion happens AFTER the encoder, not by extracting global_feat.

Public API
──────────
    SDFPointCloudEncoder(cfg: SDFEncoderCfg)
        .encode(tool_pc, obj_pc) → SDFEncodeResult
        .feature_dim             → D (token dimension)
        .num_patches             → P (patches per cloud)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points, knn_points


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class SDFEncoderCfg:
    """Configuration for SDFPointCloudEncoder."""

    # Cloud geometry
    num_pts:   int = 512   # N — points per cloud
    patch_size: int = 32   # K — points per FPS patch

    # Patch encoder
    patch_hidden: tuple[int, int] = (64, 128)  # hidden dims for per-point MLP

    # Token dimension (D)
    encoder_channel: int = 128

    # ViT transformer
    vit_depth:    int = 4    # number of ViT blocks
    vit_heads:    int = 4    # attention heads
    vit_mlp_ratio: float = 4.0
    vit_dropout:  float = 0.0

    # Positional embedding type (passed to get_pos_enc_module)
    pos_embed_type: str = "mlp"   # "mlp" | "sine" | "nerf" | "linear"

    # Whether to use a CLS token (global summary)
    use_cls_token: bool = True

    # Pretrained weights to load at init (encoder only, not SDF heads)
    weights_path: Optional[str] = None

    # Freeze encoder after loading
    freeze: bool = False


# --------------------------------------------------------------------------- #
# Output container
# --------------------------------------------------------------------------- #

class SDFEncodeResult(NamedTuple):
    """Output of SDFPointCloudEncoder.encode()."""

    fused_tokens: torch.Tensor   # (B, 2P, D) — all patch tokens after joint ViT (no CLS)
    tool_patch_idx: torch.Tensor # (B, P, K) — point indices into tool_pc
    obj_patch_idx:  torch.Tensor # (B, P, K) — point indices into obj_pc


# --------------------------------------------------------------------------- #
# Building blocks (self-contained, no rsl_rl imports needed)
# --------------------------------------------------------------------------- #

class PointNetPatchEncoder(nn.Module):
    """PointNet-style patch encoder: [B, P, K, C] → [B, P, D].

    Per-point shared MLP → max-pool + mean-pool → linear projection.
    Permutation-invariant for points within a patch.
    """

    def __init__(self, in_dim: int, hidden: tuple[int, int], out_dim: int):
        super().__init__()
        h0, h1 = hidden
        self.mlp1 = nn.Sequential(nn.Linear(in_dim, h0), nn.LayerNorm(h0), nn.GELU())
        self.mlp2 = nn.Sequential(nn.Linear(h0,     h1), nn.LayerNorm(h1), nn.GELU())
        self.proj  = nn.Sequential(nn.Linear(h1 * 2, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, P, K, C)  →  (B, P, D)"""
        x = self.mlp1(x)                           # (B, P, K, h0)
        x = self.mlp2(x)                           # (B, P, K, h1)
        x = torch.cat([x.max(2).values,
                        x.mean(2)], dim=-1)        # (B, P, 2*h1)
        return self.proj(x)                         # (B, P, D)


class SinePosEmbed(nn.Module):
    """Simple MLP positional embedding for 3D patch centres: R^3 → R^D."""

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
        """centers: (B, P, 3)  →  (B, P, D)"""
        return self.mlp(centers)


class ViTBlock(nn.Module):
    """Pre-norm ViT block: LN → MHA → residual → LN → FFN → residual."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop(y)
        x = x + self.ffn(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Per-group FPS + KNN
# --------------------------------------------------------------------------- #

def _fps_knn_per_group(
    pc:         torch.Tensor,  # (B, 2N, 3)
    group_size: int,           # N
    patch_size: int,           # K
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run FPS+KNN within each half of the concatenated cloud.

    Returns:
        patches:   (B, 2P, K, 3) — centred local patches
        centers:   (B, 2P, 3)    — patch centre positions
        patch_idx: (B, 2P, K)    — indices into the full 2N cloud
    """
    B = pc.shape[0]
    device = pc.device
    P = group_size // patch_size  # patches per group

    all_patches, all_centers, all_idx = [], [], []
    for g in range(2):
        start = g * group_size
        pts   = pc[:, start:start + group_size, :].contiguous()  # (B, N, 3)

        # FPS
        ctr, _ = sample_farthest_points(pts, K=P, random_start_point=False)  # (B, P, 3)

        # KNN
        _, nn_idx, _ = knn_points(ctr, pts, K=patch_size,
                                   return_nn=False, return_sorted=True)  # (B, P, K)

        # Gather + centre
        bi = (torch.arange(B, device=device)
              .view(B, 1, 1).expand(B, P, patch_size))
        patch_pts = pts[bi, nn_idx, :]                           # (B, P, K, 3)
        patch_pts = patch_pts - ctr.unsqueeze(2)                  # centre

        # Global indices
        nn_idx_global = nn_idx + start

        all_patches.append(patch_pts)
        all_centers.append(ctr)
        all_idx.append(nn_idx_global)

    return (torch.cat(all_patches, 1),
            torch.cat(all_centers, 1),
            torch.cat(all_idx,     1))


# --------------------------------------------------------------------------- #
# Main encoder
# --------------------------------------------------------------------------- #

class SDFPointCloudEncoder(nn.Module):
    """Joint ViT encoder for tool + object point clouds.

    Takes two 3-D point clouds and produces cross-stream-aware patch tokens
    via a single joint ViT self-attention.  Suitable both for SDF pretraining
    (add SDF heads on top) and downstream RL (freeze encoder, fuse tokens).
    """

    def __init__(self, cfg: SDFEncoderCfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.encoder_channel
        self._D = D
        self._P = cfg.num_pts // cfg.patch_size

        # ── Patch encoder (shared for both clouds) ───────────────────────────
        self.patch_enc = PointNetPatchEncoder(
            in_dim=3,
            hidden=cfg.patch_hidden,
            out_dim=D,
        )

        # ── Positional embedding ─────────────────────────────────────────────
        self.pos_embed = SinePosEmbed(D)

        # ── Type embeddings: 0 = tool, 1 = object ────────────────────────────
        self.type_embed = nn.Parameter(torch.zeros(2, D))
        nn.init.normal_(self.type_embed, std=0.02)

        # ── CLS token ────────────────────────────────────────────────────────
        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # ── Joint ViT ────────────────────────────────────────────────────────
        self.vit = nn.ModuleList([
            ViTBlock(D, cfg.vit_heads, cfg.vit_mlp_ratio, cfg.vit_dropout)
            for _ in range(cfg.vit_depth)
        ])
        self.norm = nn.LayerNorm(D)

        # ── Optional pretrained weights ──────────────────────────────────────
        if cfg.weights_path:
            self._load_weights(cfg.weights_path)

        if cfg.freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            print("[SDFPointCloudEncoder] All parameters frozen.")

        P2 = 2 * self._P
        print(
            f"[SDFPointCloudEncoder] "
            f"N={cfg.num_pts}  K={cfg.patch_size}  P_per_cloud={self._P}  "
            f"total_tokens={P2}  D={D}  "
            f"vit_depth={cfg.vit_depth}  vit_heads={cfg.vit_heads}  "
            f"cls={cfg.use_cls_token}"
        )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        """Token dimension D."""
        return self._D

    @property
    def num_patches(self) -> int:
        """Patches per cloud P."""
        return self._P

    # ── Forward ─────────────────────────────────────────────────────────────

    def encode(
        self,
        tool_pc: torch.Tensor,   # (B, N, 3)
        obj_pc:  torch.Tensor,   # (B, N, 3)
    ) -> SDFEncodeResult:
        """Encode two point clouds jointly.

        Returns an SDFEncodeResult namedtuple with:
            tool_tokens,  obj_tokens   — (B, P, D) cross-stream-aware patch tokens
            global_feat               — (B, D) scene-level CLS summary
            tool_patch_idx            — (B, P, K) point indices into tool_pc
            obj_patch_idx             — (B, P, K) point indices into obj_pc
        """
        B  = tool_pc.size(0)
        D  = self._D
        P  = self._P
        N  = self.cfg.num_pts

        # 1. Concatenate
        pc = torch.cat([tool_pc, obj_pc], dim=1)   # (B, 2N, 3)

        # 2. Per-group FPS + KNN
        patches, centers, patch_idx = _fps_knn_per_group(pc, N, self.cfg.patch_size)
        # patches:   (B, 2P, K, 3)
        # centers:   (B, 2P, 3)
        # patch_idx: (B, 2P, K)  — indices into the 2N cloud

        # 3. Encode patches
        tokens = self.patch_enc(patches)            # (B, 2P, D)

        # 4. Positional embedding
        tokens = tokens + self.pos_embed(centers)   # (B, 2P, D)

        # 5. Type embedding: first P = tool (0), last P = object (1)
        type_ids = torch.cat([
            torch.zeros(P, dtype=torch.long, device=tokens.device),
            torch.ones( P, dtype=torch.long, device=tokens.device),
        ])                                          # (2P,)
        tokens = tokens + self.type_embed[type_ids]  # (B, 2P, D)

        # 6. Prepend CLS
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x   = torch.cat([cls, tokens], dim=1)   # (B, 1+2P, D)
        else:
            x = tokens                               # (B, 2P, D)

        # 7. Joint ViT
        for blk in self.vit:
            x = blk(x)
        x = self.norm(x)

        # 8. Strip CLS token if present (like momentum encoder)
        fused_tokens = x[:, 1:, :] if self.cls_token is not None else x  # (B, 2P, D)

        # Patch indices split (subtract offset for obj so they're w.r.t. obj_pc)
        tool_patch_idx = patch_idx[:, :P, :]        # (B, P, K) — into tool_pc
        obj_patch_idx  = patch_idx[:, P:, :] - N    # (B, P, K) — into obj_pc

        return SDFEncodeResult(
            fused_tokens=fused_tokens,
            tool_patch_idx=tool_patch_idx,
            obj_patch_idx=obj_patch_idx,
        )

    # Convenience wrapper so the module can be called directly
    def forward(
        self,
        tool_pc: torch.Tensor,
        obj_pc:  torch.Tensor,
    ) -> SDFEncodeResult:
        return self.encode(tool_pc, obj_pc)

    # ── Weight loading ───────────────────────────────────────────────────────

    def _load_weights(self, path: str) -> None:
        """Load pretrained encoder weights (e.g. from SDF pretraining)."""
        print(f"[SDFPointCloudEncoder] Loading weights from {path}")
        ckpt = torch.load(path, map_location="cpu")
        # Support several checkpoint formats
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        # Strip any 'encoder.' prefix if weights come from SDFSegmentor
        sd = {k.removeprefix("encoder."): v for k, v in sd.items()
              if not k.startswith("tool_head") and not k.startswith("obj_head")}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        print("  Weights loaded.")
