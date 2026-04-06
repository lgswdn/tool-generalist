#!/usr/bin/env python3

from typing import Dict, Optional

import torch as th
import torch.nn as nn
import einops

from dataclasses import dataclass, fields, replace
from typing import Tuple
from icecream import ic

from rsl_rl.modules.models.rl.net.base import FeatureBase
from rsl_rl.modules.models.common import (
    attention,
    MultiHeadAttentionV2,
    MultiHeadLinear,
)


class StateDependentCrossFeatNet(nn.Module, FeatureBase):
    """
    State-dependet cross attention, where we derive the query
    from a linear projection of the current task specification.
    """

    @dataclass(init=False)
    class StateDependentCrossFeatNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (8, 256)
        dim_out: int = 128
        query_keys: Optional[Tuple[str, ...]] = None
        num_query: int = 16
        ctx_dim: int = 0
        emb_dim: int = 128
        cat_query: bool = False
        cat_ctx: bool = True

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)

    Config = StateDependentCrossFeatNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        in_dim: int = cfg.dim_in[-1]
        self.to_q = MultiHeadLinear(
            cfg.ctx_dim, cfg.num_query, cfg.emb_dim, unbind=False
        )
        self.to_kv = MultiHeadLinear(in_dim, 2, cfg.emb_dim, unbind=True)
        self._attn = None
        if cfg.cat_ctx and cfg.cat_query:
            raise ValueError("both cat_ctx and cat_query cannot be true!")

    def forward(
        self, x: th.Tensor, ctx: Dict[str, th.Tensor], mask: Optional[th.Tensor] = None
    ) -> th.Tensor:
        # x should look like (..., NUM_KEY, EMBED_DIM)
        # mask should look like (..., NUM_KEY)
        cfg = self.cfg
        if cfg.query_keys is not None:
            # build ctx->query
            ctx_cat = th.cat([ctx[k] for k in cfg.query_keys], dim=-1)
        else:
            ctx_cat = ctx  # Assume ctx is already a tensor if query_keys is None
        query = self.to_q(ctx_cat)
        query = query.broadcast_to(*x.shape[:-2], *query.shape[-2:])

        # build x->key,value
        k, v = self.to_kv(x)

        # apply attention
        aux = {}
        out = attention(query, k, v, aux=aux, key_padding_mask=mask)
        self._attn = aux["attn"]
        out = out.reshape(*x.shape[:-2], -1)
        if cfg.cat_query:
            return th.cat([out, query.reshape(*x.shape[:-2], -1)], -1)
        elif cfg.cat_ctx:
            return th.cat([out, ctx_cat], -1)
        else:
            return out


def main():
    batch_size: int = 7
    num_query: int = 4
    x = th.randn((batch_size, 8, 256))
    ctx = {"a": th.randn((batch_size, 32)), "b": th.randn((batch_size, 16))}
    model = StateDependentCrossFeatNet(
        StateDependentCrossFeatNet.Config(
            dim_in=(8, 256),
            dim_out=128,
            keys=["a", "b"],
            num_query=num_query,
            ctx_dim=32 + 16,
        )
    )
    y = model(x, ctx=ctx)
    print(y.shape)
    ic(model)


if __name__ == "__main__":
    main()
