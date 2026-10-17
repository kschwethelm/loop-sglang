# Request-table assignment adapted from SGLang's srt/mem_cache/allocation.py.
# Copyright 2023-2024 SGLang Team.
# Licensed under the Apache License, Version 2.0.
# See LICENSES/sglang.txt in the repository root.

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from loopsgl.core import ForwardBatch


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid).to(tl.int64) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)
    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)
    for _ in range(tl.cdiv(kv_end - kv_start, BLOCK_SIZE)):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


def assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
) -> None:
    """Assign allocated token locations using one offset range per request."""
    assert req_to_token.stride(1) == 1
    if batch_size:
        assign_req_to_token_pool[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.stride(0),
            triton.next_power_of_2(batch_size),
        )


@triton.jit
def _prepare_batch_inputs(
    seq_lens,
    extend_lens,
    req_pool_indices,
    page_table,
    page_table_stride: tl.constexpr,
    positions,
    req_mapping,
    out_loc,
    IS_DECODE: tl.constexpr,
    BLOCK_REQS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    req = tl.program_id(0)
    query_len = tl.load(extend_lens + req)
    token_start = tl.program_id(1) * BLOCK_TOKENS
    if token_start >= query_len:
        return
    if IS_DECODE:
        offset = req
    else:
        reqs = tl.arange(0, BLOCK_REQS)
        lengths = tl.load(extend_lens + reqs, mask=(reqs < req), other=0)
        offset = tl.sum(lengths)
    tokens = token_start + tl.arange(0, BLOCK_TOKENS)
    mask = tokens < query_len
    row = tl.load(req_pool_indices + req).to(tl.int64)
    pos = tl.load(seq_lens + req) - query_len + tokens
    loc = tl.load(page_table + row * page_table_stride + pos, mask=mask, other=0)
    tl.store(positions + offset + tokens, pos, mask=mask)
    tl.store(req_mapping + offset + tokens, row, mask=mask)
    tl.store(out_loc + offset + tokens, loc, mask=mask)


def prepare_batch_inputs(
    batch: ForwardBatch, page_table: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build flattened query positions and KV locations from GPU batch state."""
    num_tokens = sum(batch.extend_lens_cpu)
    positions = torch.empty(num_tokens, dtype=torch.int32, device=page_table.device)
    req_mapping = torch.empty_like(positions)
    out_loc = torch.empty_like(positions)
    _prepare_batch_inputs[(batch.padded_size, triton.cdiv(max(batch.extend_lens_cpu), 256))](
        batch.seq_lens,
        batch.extend_lens,
        batch.req_pool_indices,
        page_table,
        page_table.stride(0),
        positions,
        req_mapping,
        out_loc,
        IS_DECODE=batch.is_decode,
        BLOCK_REQS=triton.next_power_of_2(batch.padded_size),
        BLOCK_TOKENS=256,
    )
    batch.positions = positions
    batch.out_loc = out_loc
    return req_mapping, positions
