"""Last-exited KV caching with full-depth paged storage and copies on exit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from .mha_pool import MHAKVCache


@triton.jit
def _copy_exited_rows(
    storage,
    records,
    page_table,
    table_stride: tl.constexpr,
    lane_stride: tl.constexpr,
    rows_per_depth: tl.constexpr,
    row_elements: tl.constexpr,
    steps: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    lane = tl.program_id(1).to(tl.int64)
    # Each record holds (request pool index, token position, completed depth).
    request = tl.load(records + token * 3).to(tl.int64)
    position = tl.load(records + token * 3 + 1)
    row = tl.load(page_table + request * table_stride + position).to(tl.int64)
    depth = tl.load(records + token * 3 + 2) - 1
    columns = tl.program_id(2) * block + tl.arange(0, block)
    source = storage + lane * lane_stride + (depth * rows_per_depth + row) * row_elements
    values = tl.load(source + columns, columns < row_elements, other=0)
    for target in tl.static_range(steps):
        if target > depth:
            destination = (
                storage + lane * lane_stride + (target * rows_per_depth + row) * row_elements
            )
            tl.store(destination + columns, values, columns < row_elements)


@triton.jit
def _copy_depth_metadata(
    logical_table,
    logical_stride: tl.constexpr,
    physical_table,
    physical_stride: tl.constexpr,
    logical_out,
    physical_out,
    lengths,
    query_offsets,
    depths,
    scalar_depth: tl.constexpr,
    depth_value: tl.constexpr,
    logical_pages: tl.constexpr,
    page_size: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    depth = depth_value if scalar_depth else tl.load(depths + row)
    pages = tl.cdiv(tl.load(lengths + row), page_size)
    mask = columns < pages
    logical = tl.load(logical_table + row * logical_stride + columns, mask, other=0)
    tl.store(
        physical_table + row * physical_stride + columns,
        logical + depth * logical_pages,
        mask,
    )
    start = tl.load(query_offsets + row)
    end = tl.load(query_offsets + row + 1)
    out_mask = columns < end - start
    out = tl.load(logical_out + start + columns, out_mask, other=0)
    tl.store(
        physical_out + start + columns,
        out + depth * logical_pages * page_size,
        out_mask,
    )


@dataclass
class _CopyStaging:
    host: torch.Tensor
    device: torch.Tensor
    uploaded: torch.cuda.Event


class LastExitedCopyKVCache(MHAKVCache):
    """Keep all depth slots and copy completed KV into the skipped versions.

    Each depth owns a contiguous region of ``num_pages`` pages.
    Logical pages are translated into the region of the requested depth before attention.
    Early exits copy the final computed KV into all later depths of the same token, so no
    memory is reclaimed.
    """

    def __init__(self, *, loop_steps: int, **kwargs: Any) -> None:
        self.loop_steps = loop_steps
        self.logical_pages = kwargs["num_pages"]
        self.page_size = kwargs["page_size"]
        kwargs["num_pages"] *= loop_steps
        super().__init__(**kwargs)
        self.rows_per_depth = self.logical_pages * self.page_size
        self._copy_staging: list[_CopyStaging] = []
        for depth in range(1, loop_steps):
            self._kv_buffer[:, :, depth * self.logical_pages].zero_()

    def copy_from_rows(self, page_table: torch.Tensor, records: list[tuple[int, int, int]]) -> None:
        """Copy each exited token's final KV into its skipped depths.

        ``records`` holds ``(req_pool_idx, position, completed_depth)`` per exited token.
        The copy runs on the current stream, which must be ordered after the core writes
        and before any later attention or page reuse.
        """
        count = len(records)
        if not count:
            return
        staging = next(
            (
                item
                for item in self._copy_staging
                if len(item.host) >= count and item.uploaded.query()
            ),
            None,
        )
        if staging is None:
            capacity = max(256, triton.next_power_of_2(count))
            staging = _CopyStaging(
                torch.empty((capacity, 3), dtype=torch.int64, pin_memory=True),
                torch.empty((capacity, 3), dtype=torch.int64, device=self.device),
                torch.cuda.Event(),
            )
            self._copy_staging.append(staging)
        staging.host[:count].copy_(torch.tensor(records, dtype=torch.int64))
        staging.device[:count].copy_(staging.host[:count], non_blocking=True)
        # Host rows can be reused after upload; device rows are ordered on this stream.
        staging.uploaded.record()
        row_elements = self._storage_shape[1] * self._storage_shape[2]
        _copy_exited_rows[(count, 2 * self.num_layers, triton.cdiv(row_elements, 1024))](
            self._kv_buffer,
            staging.device,
            page_table,
            page_table.stride(0),
            self._kv_buffer.stride(1),
            self.rows_per_depth,
            row_elements,
            self.loop_steps,
            1024,
        )

    def prepare_core_metadata(
        self, metadata: Any, logical_out: torch.Tensor, depths: int | torch.Tensor
    ) -> None:
        """Translate logical pages into the depth region for each request row."""
        scalar = isinstance(depths, int)
        _copy_depth_metadata[
            (
                metadata.cache_seqlens.numel(),
                triton.cdiv(max(metadata.page_table.shape[1], metadata.max_seqlen_q), 256),
            )
        ](
            metadata.page_table,
            metadata.page_table.stride(0),
            metadata.core_page_table[0],
            metadata.core_page_table.stride(1),
            logical_out,
            metadata.core_out_loc[0],
            metadata.cache_seqlens,
            metadata.cu_seqlens_q,
            depths,
            scalar,
            depths if scalar else 0,
            self.logical_pages,
            self.page_size,
            256,
        )
