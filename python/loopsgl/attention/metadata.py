# Portions adapted from SGLang's kernels/ops/attention/metadata.py and
# kernels/ops/kvcache/kv_indices.py (v0.5.20).
# Copyright 2023-2024 SGLang Team.
# Licensed under the Apache License, Version 2.0.
# See LICENSES/sglang.txt in the repository root.

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_indices_ptr,
    req_to_token_ptr_stride,
):
    """Gather token-granular KV locations into FlashInfer's flat CSR indices."""
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    kv_indices_offset = tl.load(kv_indptr + pid)
    kv_len = tl.load(page_kernel_lens_ptr + pid).to(tl.int32)
    for i in range(tl.cdiv(kv_len, BLOCK_SIZE)):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_len
        data = tl.load(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


def create_flashinfer_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
) -> None:
    """Fill an existing indices buffer without Python row slicing or concatenation."""
    assert req_to_token.stride(1) == 1
    create_flashinfer_kv_indices_triton[(req_pool_indices.numel(),)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        kv_indptr,
        kv_indices,
        req_to_token.stride(0),
    )


@triton.jit
def _fused_metadata_kernel_general(
    seq_lens,
    extend_lens,
    req_to_token,
    req_to_token_stride: tl.constexpr,
    req_pool_indices,
    cache_seqlens,
    cu_seqlens_k,
    cu_seqlens_q,
    page_table,
    page_table_stride: tl.constexpr,
    batch_size,
    max_seq_pages,
    page_size: tl.constexpr,
    seq_len_delta: tl.constexpr,
    IS_DECODE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    # SGLang fuses the prefix sum and row gathering in a single launch.
    if pid_b == 0 and pid_c == 0:
        acc_k = 0
        acc_q = 0
        for idx in range(batch_size):
            seq = tl.load(seq_lens + idx).to(tl.int32) + seq_len_delta
            query = 1 if IS_DECODE else tl.load(extend_lens + idx).to(tl.int32)
            tl.store(cache_seqlens + idx, seq)
            tl.store(cu_seqlens_k + idx, acc_k)
            tl.store(cu_seqlens_q + idx, acc_q)
            acc_k += seq
            acc_q += query
        tl.store(cu_seqlens_k + batch_size, acc_k)
        tl.store(cu_seqlens_q + batch_size, acc_q)

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32) + seq_len_delta
    num_live_pages = tl.minimum(tl.cdiv(seq_len, page_size), max_seq_pages)
    col_start = pid_c * BLOCK_COLS
    if col_start >= num_live_pages:
        return
    row_idx = tl.load(req_pool_indices + pid_b).to(tl.int64)
    cols = col_start + tl.arange(0, BLOCK_COLS)
    mask = cols < num_live_pages
    locations = tl.load(
        req_to_token + row_idx * req_to_token_stride + cols * page_size,
        mask=mask,
        other=0,
        cache_modifier=".cg",
    )
    tl.store(
        page_table + pid_b * page_table_stride + cols,
        locations // page_size,
        mask=mask,
        cache_modifier=".cg",
    )


def normal_decode_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    max_seq_pages: int,
    seq_lens: torch.Tensor,
    seq_len_delta: int,
    page_size: int,
    cu_seqlens_q: torch.Tensor,
) -> None:
    """Fill decode metadata in place, leaving page-table tails untouched.

    Attention consumers must bound each row by cache_seqlens_int32.
    Output buffers can be eager buffers or the buffers used by captured graphs.
    """
    _set_metadata(
        cache_seqlens_int32,
        cu_seqlens_k,
        cu_seqlens_q,
        page_table,
        req_to_token,
        req_pool_indices,
        max_seq_pages,
        seq_lens,
        seq_lens,
        seq_len_delta,
        page_size,
        is_decode=True,
    )


def extend_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    max_seq_pages: int,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    page_size: int,
) -> None:
    """Prepare variable-length prefill metadata, including cached prefixes."""
    _set_metadata(
        cache_seqlens_int32,
        cu_seqlens_k,
        cu_seqlens_q,
        page_table,
        req_to_token,
        req_pool_indices,
        max_seq_pages,
        seq_lens,
        extend_lens,
        0,
        page_size,
        is_decode=False,
    )


def _set_metadata(
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    max_seq_pages: int,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    seq_len_delta: int,
    page_size: int,
    *,
    is_decode: bool,
) -> None:
    assert page_size > 0 and 0 < max_seq_pages <= page_table.shape[1]
    assert req_to_token.stride(1) == page_table.stride(1) == 1
    assert seq_lens.is_contiguous() and req_pool_indices.is_contiguous()
    batch_size = cache_seqlens.numel()
    _fused_metadata_kernel_general[(batch_size, triton.cdiv(max_seq_pages, 256))](
        seq_lens,
        extend_lens,
        req_to_token,
        req_to_token.stride(0),
        req_pool_indices,
        cache_seqlens,
        cu_seqlens_k,
        cu_seqlens_q,
        page_table,
        page_table.stride(0),
        batch_size,
        max_seq_pages,
        page_size,
        seq_len_delta,
        IS_DECODE=is_decode,
        BLOCK_COLS=256,
        num_warps=8,
        num_stages=3,
    )


@triton.jit
def _last_exited_metadata_kernel(
    logical_table,
    table_stride: tl.constexpr,
    physical_table,
    physical_group_stride: tl.constexpr,
    physical_stride: tl.constexpr,
    version_map,
    version_stride: tl.constexpr,
    version_group_stride: tl.constexpr,
    seq_lens,
    cu_seqlens_q,
    logical_out,
    physical_out,
    out_group_stride: tl.constexpr,
    depths,
    SCALAR_DEPTH: tl.constexpr,
    DEPTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(2)
    physical_table += group * physical_group_stride
    physical_out += group * out_group_stride
    version_map += group * version_group_stride
    row = tl.program_id(0)
    block = tl.program_id(1)
    depth = DEPTH if SCALAR_DEPTH else tl.load(depths + row)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    seq_len = tl.load(seq_lens + row)
    mask = cols < seq_len
    handles = tl.load(logical_table + row * table_stride + cols, mask, other=0)
    physical = tl.load(version_map + depth * version_stride + handles, mask, other=0)
    tl.store(physical_table + row * physical_stride + cols, physical, mask)
    start = tl.load(cu_seqlens_q + row)
    end = tl.load(cu_seqlens_q + row + 1)
    out_mask = cols < end - start
    out_handles = tl.load(logical_out + start + cols, out_mask, other=0)
    out_physical = tl.load(version_map + depth * version_stride + out_handles, out_mask, other=0)
    tl.store(physical_out + start + cols, out_physical, out_mask)


def prepare_last_exited_metadata(
    logical_table: torch.Tensor,
    physical_table: torch.Tensor,
    version_map: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    logical_out: torch.Tensor,
    physical_out: torch.Tensor,
    depths: int | torch.Tensor,
) -> None:
    """Resolve logical handles for each layer group at the requested depth."""
    _last_exited_metadata_kernel[
        (seq_lens.numel(), triton.cdiv(logical_table.shape[1], 256), version_map.shape[1])
    ](
        logical_table,
        logical_table.stride(0),
        physical_table,
        physical_table.stride(0),
        physical_table.stride(1),
        version_map,
        version_map.stride(0),
        version_map.stride(1),
        seq_lens,
        cu_seqlens_q,
        logical_out,
        physical_out,
        physical_out.stride(0),
        depths if isinstance(depths, torch.Tensor) else logical_out,
        SCALAR_DEPTH=isinstance(depths, int),
        DEPTH=depths if isinstance(depths, int) else 0,
        BLOCK=256,
    )
