from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from loopsgl.core import ForwardBatch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData, get_metadata_buffer, prepare_paged_metadata

if TYPE_CHECKING:
    from loopsgl.models import ModelConfig


@dataclass
class TRTLLMCaptureData(BaseCaptureData):
    pass


@dataclass
class TRTLLMMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TensorRTLLMBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: TRTLLMCaptureData | None = None
        self.eager: BaseCaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.kvcache.device
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: ForwardBatch
    ) -> torch.Tensor:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        metadata = batch.attn_metadata
        assert isinstance(metadata, TRTLLMMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))

        if batch.is_prefill:
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens,
                max_q_len=metadata.max_seqlen_q,
                max_kv_len=metadata.max_seqlen_k,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                cum_seq_lens_q=metadata.cu_seqlens_q,
                cum_seq_lens_kv=metadata.cu_seqlens_k,
                kv_layout="NHD",
                batch_size=batch.size,
                out_dtype=q.dtype,
            )
        else:
            return trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens,
                max_seq_len=metadata.max_seqlen_k,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                kv_layout="NHD",
                out_dtype=q.dtype,
            )

    def prepare_metadata(self, batch: ForwardBatch) -> None:
        self.eager = get_metadata_buffer(batch, self.eager, self.page_size)
        prepare_paged_metadata(batch, self.eager, self.page_size)
        bs = batch.padded_size
        batch.attn_metadata = TRTLLMMetadata(
            cu_seqlens_k=self.eager.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=self.eager.cu_seqlens_q[: bs + 1],
            cache_seqlens=self.eager.seq_lens[:bs],
            max_seqlen_k=max(batch.seq_lens_cpu),
            max_seqlen_q=max(batch.extend_lens_cpu),
            # TRTLLM derives the row stride from width, so expose the full capacity.
            page_table=self.eager.page_table[:bs],
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = TRTLLMCaptureData.create(
            max_bs, (max_seq_len + self.page_size - 1) // self.page_size, self.kvcache.device
        )
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: ForwardBatch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = TRTLLMMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: ForwardBatch) -> None:
        bs = batch.padded_size
        assert self.capture is not None and bs in self.capture_bs
        prepare_paged_metadata(batch, self.capture, self.page_size)
        batch.attn_metadata = TRTLLMMetadata(
            cu_seqlens_k=self.capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=self.capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=self.capture.seq_lens[:bs],
            max_seqlen_k=self.capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,
            page_table=self.capture.page_table[:bs],
        )
