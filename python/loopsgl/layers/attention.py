from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from loopsgl.core import get_global_ctx
from loopsgl.distributed import get_tp_info
from loopsgl.utils import div_even
from loopsgl.utils.breakable_graph import eager_on_graph, is_prefill_graph_warmup

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from loopsgl.layers import RMSNorm
    from loopsgl.models import RotaryConfig


@eager_on_graph()
def _prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    layer_id: int,
    core_metadata_depth: int | torch.Tensor | None = None,
) -> None:
    """Write attention over real query rows into a stable token-bucket output.

    The first core layer also prepares last-exited KV addresses for its depth.
    """
    if is_prefill_graph_warmup():
        out.zero_()
        return
    ctx = get_global_ctx()
    batch = ctx.batch
    count = batch.prefill_num_tokens
    assert count is not None
    if core_metadata_depth is not None:
        ctx.attn_backend.prepare_core_metadata(batch, core_metadata_depth)
    result = ctx.attn_backend.forward(q[:count], k[:count], v[:count], layer_id, batch)
    out[:count].copy_(result)
    out[count:].zero_()


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
            is_neox_style=rotary_config.is_neox_style,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(
        self,
        qkv: torch.Tensor,
        *,
        cache_layer_id: int | None = None,
        core_metadata_depth: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        layer_id = self.layer_id if cache_layer_id is None else cache_layer_id
        if ctx.batch.prefill_num_tokens is not None:
            o = torch.empty_like(q)
            _prefill_attention(q, k, v, o, layer_id, core_metadata_depth)
        else:
            if core_metadata_depth is not None:
                ctx.attn_backend.prepare_core_metadata(ctx.batch, core_metadata_depth)
            o = ctx.attn_backend.forward(q, k, v, layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)
