#!/usr/bin/env python3

from typing import Optional, Dict
from dataclasses import dataclass
from rsl_rl.modules.util.config import ConfigBase

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops
import math

# Updated for flash-attn 2.x
try:
    from flash_attn import flash_attn_qkvpacked_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print(
        "[WARN] flash-attn not available in point_tokens.py, using standard attention"
    )

from torch.nn.modules.activation import MultiheadAttention
from einops import rearrange

from rsl_rl.modules.models.common import PosEncodingSine, MLP
from rsl_rl.modules.models.common import replace_layer
from rsl_rl.modules.models.rl.net.mae.fast_attn import replace_self_attention

from icecream import ic

# import nvtx


# Custom FlashMHA replacement for flash-attn 2.x
class FlashMHA(nn.Module):
    """Replacement for old FlashMHA using flash-attn 2.x API"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_dropout = attention_dropout
        self.batch_first = batch_first
        self.causal = causal

        self.Wqkv = nn.Linear(
            embed_dim, 3 * embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

    def forward(self, x, key_padding_mask=None, need_weights=False, **kwargs):
        """
        Args:
            x: (batch, seqlen, embed_dim) if batch_first else (seqlen, batch, embed_dim)
            key_padding_mask: (batch, seqlen), 1=keep, 0=drop (FlashMHA convention)
            need_weights: bool
        Returns:
            output: same shape as x
            attn_weights: None or attention weights
        """
        if not self.batch_first:
            x = x.transpose(0, 1)  # (S, B, E) -> (B, S, E)

        batch_size, seqlen, _ = x.shape

        # Generate QKV
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )

        if FLASH_ATTN_AVAILABLE and not need_weights:
            # Use flash attention
            context = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=None,
                causal=self.causal,
            )  # (batch, seqlen, nheads, headdim)

            # Apply mask if needed (zero out dropped positions)
            if key_padding_mask is not None:
                # key_padding_mask: (B, S), 1=keep, 0=drop
                mask = key_padding_mask.unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
                context = context * mask

            context = rearrange(context, "b s h d -> b s (h d)")
            output = self.out_proj(context)

            if not self.batch_first:
                output = output.transpose(0, 1)  # (B, S, E) -> (S, B, E)
            return output, None
        else:
            # Fallback to standard attention
            q, k, v = qkv.unbind(dim=2)  # Each: (B, S, H, D)

            # Transpose for attention: (B, S, H, D) -> (B, H, S, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Compute attention scores
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores = th.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, S)

            # Apply key padding mask if provided
            if key_padding_mask is not None:
                # key_padding_mask: (B, S), 1=keep, 0=drop
                # Convert to attention mask: 0=keep, -inf=drop
                attn_mask = (1 - key_padding_mask).bool()  # 0=keep, 1=drop
                attn_scores = attn_scores.masked_fill(
                    attn_mask.unsqueeze(1).unsqueeze(2), float("-inf")  # (B, 1, 1, S)
                )

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(
                attn_probs, p=self.attention_dropout, training=self.training
            )

            context = th.matmul(attn_probs, v)  # (B, H, S, D)
            context = context.transpose(1, 2)  # (B, S, H, D)
            context = rearrange(context, "b s h d -> b s (h d)")

            output = self.out_proj(context)

            if not self.batch_first:
                output = output.transpose(0, 1)  # (B, S, E) -> (S, B, E)

            if need_weights:
                return output, attn_probs
            else:
                return output, None


class FlashMHAWrapper(nn.Module):
    def __init__(self, *args, **kwds):
        super().__init__()
        self._attn = FlashMHA(*args, **kwds)

    def forward(
        self,
        query: th.Tensor,
        key: th.Tensor,
        value: th.Tensor,
        key_padding_mask: Optional[th.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[th.Tensor] = None,
        average_attn_weights: bool = True,
    ):
        # self, x, key_padding_mask=None, need_weights=False
        # source: (query, key, value)
        # we expect query == key == value
        # assert (query == key)
        # assert (key == value)

        # target: x
        # (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)

        with th.cuda.amp.autocast(True, th.float16):
            x, _ = self._attn(
                query, key_padding_mask=key_padding_mask, need_weights=need_weights
            )
        x = x.to(dtype=query.dtype)
        return x, None


class FlashSelfAttention(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self._num_head = num_head
        self.attention = FlashMHA()

    def forward(
        self,
        query: th.Tensor,
        key: th.Tensor,
        value: th.Tensor,
        key_padding_mask: Optional[th.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[th.Tensor] = None,
        average_attn_weights: bool = True,
    ):
        assert query == key
        assert key == value
        # source:
        # B, S, D when bf=True

        # target:
        # (B, S, 3, H, D)
        qkv = einops.repeat(query, "b s d -> b s three h d", three=3, h=self.num_head)
        return self.attention(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights
        )


class HilbertCode(nn.Module):
    def __init__(self, m: int = 10):
        super().__init__()
        self.m: int = m
        self.register_buffer(
            "PHM",
            th.as_tensor(
                [
                    [0, 1, 3, 2, 7, 6, 4, 5],
                    [4, 5, 7, 6, 3, 2, 0, 1],
                    [6, 7, 5, 4, 1, 0, 2, 3],
                    [2, 3, 1, 0, 5, 4, 6, 7],
                    [0, 1, 7, 6, 3, 2, 4, 5],
                    [4, 5, 3, 2, 7, 6, 0, 1],
                    [2, 3, 5, 4, 1, 0, 6, 7],
                    [6, 7, 1, 0, 5, 4, 2, 3],
                    [0, 7, 3, 4, 1, 6, 2, 5],
                    [2, 5, 1, 6, 3, 4, 0, 7],
                    [4, 3, 5, 2, 7, 0, 6, 1],
                    [4, 3, 7, 0, 5, 2, 6, 1],
                ],
                dtype=th.long,
            ).reshape(-1, 2, 2, 2),
        )
        self.register_buffer(
            "PNM",
            th.as_tensor(
                [
                    [8, 4, 3, 4, 10, 5, 3, 5],
                    [2, 4, 11, 4, 2, 5, 9, 5],
                    [1, 7, 8, 8, 1, 6, 10, 10],
                    [11, 11, 0, 7, 9, 9, 0, 6],
                    [8, 0, 11, 1, 6, 0, 6, 1],
                    [10, 10, 9, 9, 0, 7, 1, 7],
                    [10, 10, 9, 9, 4, 2, 4, 3],
                    [2, 8, 3, 11, 2, 5, 3, 5],
                    [0, 2, 9, 9, 4, 7, 4, 7],
                    [1, 3, 8, 8, 1, 3, 5, 6],
                    [11, 11, 0, 2, 5, 6, 0, 2],
                    [10, 10, 1, 3, 4, 7, 4, 7],
                ],
                dtype=th.long,
            ).reshape(-1, 2, 2, 2),
        )

    def forward(self, p: th.Tensor) -> th.Tensor:
        p = p.to(th.long)
        a: th.Tensor = th.zeros(p.shape[:-1], dtype=th.int32, device=p.device)
        T: th.Tensor = th.zeros(p.shape[:-1], dtype=th.long, device=p.device)
        for q in range(self.m):
            x = p.bitwise_right_shift(self.m - q - 1).bitwise_and_(1)
            i, j, k = th.unbind(x, dim=-1)
            a.bitwise_left_shift_(3).bitwise_or_(self.PHM[T, i, j, k])
            T = self.PNM[T, i, j, k]

        # for i in range(self.m):
        # for i in range(1):
        #    d: int = self.m - i - 1
        #    x = (p >> d) & 1
        #    a = (a << 3) | self.PHM[T,
        #                            x[..., 0],
        #                            x[..., 1],
        #                            x[..., 2]]
        #    T = self.PNM[T,
        #                 x[..., 0],
        #                 x[..., 1],
        #                 x[..., 2]]
        return a


def morton_code(p: th.Tensor) -> th.Tensor:
    """
    (...,3) -> (...)
    """
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]

    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249

    y = (y | (y << 16)) & 0x030000FF
    y = (y | (y << 8)) & 0x0300F00F
    y = (y | (y << 4)) & 0x030C30C3
    y = (y | (y << 2)) & 0x09249249

    z = (z | (z << 16)) & 0x030000FF
    z = (z | (z << 8)) & 0x0300F00F
    z = (z | (z << 4)) & 0x030C30C3
    z = (z | (z << 2)) & 0x09249249

    return x | (y << 1) | (z << 2)


def spatial_sort(x: th.Tensor, code_fn=morton_code) -> th.Tensor:
    _HMAX: int = (1 << 10) - 1
    # 10-bit x 3 ==> 30-bit < 32 (or 31) bits,
    # so this is correct.
    bmin = th.min(x, dim=-2, keepdim=True).values
    bmax = th.max(x, dim=-2, keepdim=True).values

    # Convert to fixed-point representation
    p = th.floor(_HMAX * (x - bmin) / (bmax - bmin)).to(dtype=th.int32)
    z = code_fn(p)
    i = th.argsort(z, dim=-1)

    out = th.gather(x, -2, i[..., None].expand(x.shape))
    return out


class SpatialSort(nn.Module):
    def __init__(self, code: nn.Module):
        super().__init__()
        self.code = code

    def forward(
        self, x: th.Tensor, aux: Optional[Dict[str, th.Tensor]] = None
    ) -> th.Tensor:
        with th.no_grad():
            _HMAX: int = (1 << 10) - 1
            # 10-bit x 3 ==> 30-bit < 32 (or 31) bits,
            # so this is correct.
            bmin = th.min(x, dim=-2, keepdim=True).values
            bmax = th.max(x, dim=-2, keepdim=True).values
            # Convert to fixed-point representation
            p = th.floor(_HMAX * (x - bmin) / (bmax - bmin)).to(dtype=th.int32)
            z = self.code(p)
            i = th.argsort(z, dim=-1)
            # i = th.randint(z.shape[-1], size=z.shape,
            #                device=z.device,
            #                dtype=th.long)
        out = th.take_along_dim(x, i[..., None], -2)
        if aux is not None:
            aux["sort_index"] = i
        # out = th.gather(x, -2, i[..., None].expand(x.shape))
        return out


class PatchEncoder(nn.Module):
    """Mini pointnet-style encoder"""

    def __init__(self, dim_out: int):
        super().__init__()
        self.dim_out = dim_out
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, dim_out, 1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        B, P, 3 -> B, C
        """
        f = self.first_conv(x.swapaxes(-2, -1))  # -> B,F,P
        f_g = th.max(f, dim=2, keepdim=True)[0]  # -> B,F,1
        f = th.cat([f_g.expand(f.shape), f], dim=1)  # -> B, 2F, P
        f = self.second_conv(f)
        f_g = th.max(f, dim=2, keepdim=False)[0]  # -> B,F,1
        return f_g


class MLPPatchEncoder(nn.Module):
    def __init__(self, patch_size: int, dim_out: int):
        super().__init__()
        dims = (patch_size * 3, 64, dim_out)
        self.model = MLP(
            dims, act_cls=nn.ELU, activate_output=False, use_bn=False, use_ln=True
        )
        self.dim_out = dim_out

    # @nvtx.annotate("MLPPatchEncoder.forward")
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        B,P,3 -> B,C
        """
        x = einops.rearrange(x, "... p d -> ... (p d)")
        x = self.model(x)
        return x


class NoOpPatchEncoder(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.dim_out = patch_size * 3

    # @nvtx.annotate("NoOpPatchEncoder.forward")
    def forward(self, x: th.Tensor) -> th.Tensor:
        return einops.rearrange(x, "... p d -> ... (p d)")
        return x


class PointTokenizer(nn.Module):
    @dataclass
    class Config(ConfigBase):
        use_hilbert: bool = True
        patch_size: int = 16
        patch_feature_dim: int = 64
        output_dim: int = 256
        use_viewpoint: bool = True
        transformer_depth: int = 2
        transformer_num_head: int = 8
        pe_type: str = "sine"
        p_drop: float = 0.0
        version: int = 2
        fast_attn: bool = True

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        if cfg.use_hilbert:
            # self.hilbert_code = HilbertCode()
            self.hilbert_code = th.jit.script(HilbertCode())
        code_fn = self.hilbert_code if cfg.use_hilbert else morton_code
        # self.spatial_sort = th.jit.script(SpatialSort(code_fn))
        # self.spatial_sort = SpatialSort(code_fn)

        # self.patch_encoder = PatchEncoder(
        #     cfg.patch_feature_dim)
        self.patch_encoder = MLPPatchEncoder(cfg.patch_size, cfg.patch_feature_dim)
        # self.patch_encoder = NoOpPatchEncoder(
        #     cfg.patch_size)
        feat_dim: int = self.patch_encoder.dim_out

        if cfg.pe_type == "sine":
            self.pos_enc = PosEncodingSine(3, feat_dim - 3)
        elif cfg.pe_type == "linear":
            self.pos_enc = nn.Linear(3, feat_dim)
        else:
            raise ValueError(f"unknown pe type = {cfg.pe_type}")

        d_in = feat_dim
        if cfg.use_viewpoint:
            d_in += 3
        self.projector = nn.Linear(d_in, cfg.output_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                cfg.output_dim,
                cfg.transformer_num_head,
                cfg.output_dim,
                dropout=cfg.p_drop,
                batch_first=True,
            ),
            cfg.transformer_depth,
        )

        if cfg.fast_attn:

            def _new_layer(old_layer: MultiheadAttention):
                assert old_layer.bias_k is None
                assert old_layer.bias_v is None

                attention = FlashMHAWrapper(
                    old_layer.embed_dim,
                    old_layer.num_heads,
                    (old_layer.in_proj_bias is not None),
                    old_layer.batch_first,
                    old_layer.dropout,
                    False,
                    old_layer.out_proj.weight.device,
                    old_layer.out_proj.weight.dtype,
                ).to(old_layer.out_proj.weight.device)

                # COPY WEIGHTS
                with th.no_grad():
                    attention._attn.Wqkv.weight.copy_(old_layer.in_proj_weight)
                    attention._attn.Wqkv.bias.copy_(old_layer.in_proj_bias)
                    attention._attn.out_proj.weight.copy_(old_layer.out_proj.weight)

                # [1] copy train/eval flag.
                attention.train(old_layer.training)
                # [2] copy `requires_grad` flag.
                attention.requires_grad_(old_layer.out_proj.weight.requires_grad)
                return attention

            replace_layer(self.transformer, MultiheadAttention, _new_layer)

        # Genuinely have zero clue
        self.register_parameter(
            "emb_token", nn.Parameter(th.rand(feat_dim), requires_grad=True)
        )
        nn.init.normal_(self.emb_token, std=0.02)

    # @nvtx.annotate("PointTokenizer.forward")
    def forward(
        self,
        x: th.Tensor,
        v: Optional[th.Tensor] = None,
        aux: Optional[Dict[str, th.Tensor]] = None,
    ):
        cfg = self.cfg
        s0 = x.shape
        x = x.reshape(-1, *x.shape[-2:])

        # First, spatial organization.
        # with nvtx.annotate("spatial_sort"):
        with th.no_grad():
            if False:
                x = self.spatial_sort(x)
            else:
                code_fn = self.hilbert_code if cfg.use_hilbert else morton_code
                x = spatial_sort(x, code_fn)  # ..., N, 3

        # with nvtx.annotate("patch_encode"):
        s = x.shape
        x = x.reshape(-1, cfg.patch_size, 3)
        c = x.mean(dim=-2)
        f = self.patch_encoder(x - c[..., None, :])
        c_pe = self.pos_enc(c)
        f = f + c_pe

        # with nvtx.annotate("add_embedding"):
        num_patch = s[-2] // cfg.patch_size
        f = f.reshape(*s[:-2], num_patch, -1)
        if cfg.version == 1:
            # last
            z = self.emb_token[None, None].expand(f.shape[0], 1, -1)
            f = th.cat([f, z], dim=-2)
        else:
            # first
            z = self.emb_token[None, None].expand(f.shape[0], 1, -1)
            f = th.cat([z, f], dim=-2)

        # with nvtx.annotate("add_viewpoint"):
        if cfg.use_viewpoint and v is not None:
            v = v.unsqueeze(dim=-2).expand(*s[:-2], num_patch + 1, -1)
            f = th.cat([f, v], dim=-1)

        # with nvtx.annotate("projector"):
        # TODO(ycho): consider quantization here.
        f = self.projector(f)

        # with nvtx.annotate("transformer"):
        f = self.transformer(f)

        # .mean(dim=-2)
        # print('f', f[:, 0])  # 4,256
        if cfg.version == 1:
            f = f[..., -1, :]
        else:
            # legacy OR version 2
            f = f[..., 0, :]

        f = f.reshape(*s0[:-2], *f.shape[1:])
        return f


class PointTokenizerLegacy(PointTokenizer):
    @dataclass
    class Config(PointTokenizer.Config):
        v2: bool = False


def test_morton_sort():
    x = th.randn((4, 1024, 3))
    y = spatial_sort(x)
    print("y", y.shape)


def test_tokenizer():
    x = th.randn((4, 512, 3), device="cuda:1")
    v = th.randn((4, 3), device="cuda:1")
    model = PointTokenizer(
        PointTokenizer.Config(use_viewpoint=False, fast_attn=True)
    ).to(device="cuda:1")
    print(model)

    print(model)
    y = model(x)
    print(y.shape)

    x = th.randn((4, 512, 3), device="cpu")
    v = th.randn((4, 3), device="cpu")
    model = PointTokenizer(PointTokenizer.Config(use_viewpoint=True, fast_attn=False))
    print(model)
    y = model(x, v)
    print("y", y.shape)


def show_spatial_sort():
    from util.vis import AutoWindow
    import numpy as np
    import pickle
    from util.torch_util import dcn

    spatial_sort = SpatialSort(HilbertCode())

    with open("/input/ACRONYM/cloud.pkl", "rb") as fp:
        d = pickle.load(fp)

    NP = 16  # num patches
    # v = next(iter(d.values()))
    vs = list(d.values())
    v = vs[np.random.choice(len(vs))]
    P = v.shape[-2] // NP  # patch size
    # x = th.randn((NP * P, 3))
    x = th.as_tensor(v)
    # print('x', x.shape)
    # x = th.meshgrid(
    #     th.linspace(-1, 1, 32),
    #     th.linspace(-1, 1, 32),
    #     th.linspace(-1, 1, 32),
    #     indexing='ij')
    # x = th.stack(x, dim=-1).reshape(-1, 3)
    y = spatial_sort(x).detach().cpu().numpy()
    # x = x[2]
    win = AutoWindow()
    vis = win.vis
    vis.add_cloud("cloud", x.detach().cpu().numpy())
    print("y", y.shape)
    x = y.max() - y.min()
    for i, c in enumerate(y.reshape(NP, P, 3)):
        print("i", i, c.shape)
        vis.add_cloud(
            f"cloud-{i:02d}", c + dcn(x).item(), color=np.random.uniform(size=3)
        )
        print("add")
        win.wait()


def xxx():
    from util.vis import AutoWindow
    import numpy as np
    import pickle
    from util.torch_util import dcn
    from pytorch3d.ops.sample_farthest_points import sample_farthest_points
    from pytorch3d.ops.knn import knn_points, knn_gather

    with open("/input/ACRONYM/cloud.pkl", "rb") as fp:
        d = pickle.load(fp)
    NP = 16  # num patches
    # v = next(iter(d.values()))
    vs = list(d.values())
    v = vs[np.random.choice(len(vs))]
    P = v.shape[-2] // NP  # patch size
    # x = th.randn((NP * P, 3))
    x = th.as_tensor(v)

    c, _ = sample_farthest_points(x[None], K=x.shape[-2] // P)
    _, nn_idx, p = knn_points(c, x[None], K=64, return_nn=True)
    p = p[0]
    win = AutoWindow()
    vis = win.vis
    vis.add_cloud("cloud", dcn(x), color=(1, 0, 0))
    vis.add_cloud("fps-patch", dcn(p.reshape(-1, 3)) + (0, 0, 1), color=(0, 0, 1))
    win.wait()


def main():
    # show_spatial_sort()
    xxx()

    # test_tokenizer()
    # encoder = PointnetSimpleEncoder(
    #     PointnetSimpleEncoder.Config())
    # x = th.randn((8, 512, 3))
    # z = encoder(x)
    # print(z)


if __name__ == "__main__":
    main()
