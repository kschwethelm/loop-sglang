from __future__ import annotations

from dataclasses import dataclass

import torch

from loopsgl.core import ForwardBatch, get_global_ctx

from .metadata import extend_set_metadata, normal_decode_set_metadata


@dataclass
class BaseCaptureData:
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor

    @classmethod
    def create(
        cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs
    ) -> BaseCaptureData:
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )


def get_metadata_buffer(
    batch: ForwardBatch, buffer: BaseCaptureData | None, page_size: int
) -> BaseCaptureData:
    """Grow eager buffers only when batch or page-table capacity increases."""
    num_pages = (max(batch.seq_lens_cpu) + page_size - 1) // page_size
    if (
        buffer is None
        or batch.padded_size > buffer.seq_lens.numel()
        or num_pages > buffer.page_table.shape[1]
    ):
        max_bs = max(batch.padded_size, buffer.seq_lens.numel() if buffer else 0)
        max_pages = max(num_pages, buffer.page_table.shape[1] if buffer else 0)
        buffer = BaseCaptureData.create(
            1 << (max_bs - 1).bit_length(),
            1 << (max_pages - 1).bit_length(),
            batch.seq_lens.device,
        )
    return buffer


def prepare_paged_metadata(batch: ForwardBatch, buffer: BaseCaptureData, page_size: int) -> None:
    """Write directly into reusable buffers on the stream executing attention."""
    bs = batch.padded_size
    max_pages = (max(batch.seq_lens_cpu) + page_size - 1) // page_size
    if batch.is_decode:
        normal_decode_set_metadata(
            buffer.seq_lens[:bs],
            buffer.cu_seqlens_k[: bs + 1],
            buffer.page_table[:bs],
            get_global_ctx().page_table,
            batch.req_pool_indices,
            max_pages,
            batch.seq_lens,
            0,
            page_size,
            buffer.cu_seqlens_q[: bs + 1],
        )
    else:
        extend_set_metadata(
            buffer.seq_lens[:bs],
            buffer.cu_seqlens_k[: bs + 1],
            buffer.cu_seqlens_q[: bs + 1],
            buffer.page_table[:bs],
            get_global_ctx().page_table,
            batch.req_pool_indices,
            max_pages,
            batch.seq_lens,
            batch.extend_lens,
            page_size,
        )
