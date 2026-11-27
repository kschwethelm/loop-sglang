from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _max_pair(left, left_index, right, right_index):
    # Match PyTorch GreaterOrNan: the first NaN wins, then the first maximum.
    left_nan = left != left
    right_nan = right != right
    choose_left = tl.where(
        left_nan,
        (~right_nan) | (left_index < right_index),
        (left > right) | ((left == right) & (left_index < right_index)),
    )
    return tl.where(choose_left, left, right), tl.where(choose_left, left_index, right_index)


@triton.jit
def _partial_argmax(
    logits,
    partial_values,
    partial_indices,
    VOCAB: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(logits + row * VOCAB + cols, cols < VOCAB, -float("inf")).to(tl.float32)
    indices = tl.where(cols < VOCAB, cols, 2147483647)
    value, index = tl.reduce((values, indices), 0, _max_pair)
    tl.store(partial_values + row * CHUNKS + chunk, value)
    tl.store(partial_indices + row * CHUNKS + chunk, index)


@triton.jit
def _finish_argmax_prelude(
    values,
    indices,
    output,
    slots,
    embedding,
    hidden,
    cdf,
    CHUNKS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_RED: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_RED)
    value = tl.load(values + row * CHUNKS + cols, cols < CHUNKS, -float("inf"))
    index = tl.load(indices + row * CHUNKS + cols, cols < CHUNKS, 2147483647)
    _, winner = tl.reduce((value, index), 0, _max_pair)
    tl.store(output + row, winner)
    slot = tl.load(slots + row).to(tl.int64)
    components = tl.arange(0, BLOCK_HIDDEN)
    state = tl.load(embedding + winner.to(tl.int64) * HIDDEN + components, components < HIDDEN, 0)
    tl.store(hidden + slot * HIDDEN + components, state, components < HIDDEN)
    tl.store(cdf + slot, 0.0)


def try_greedy_argmax_prelude(
    logits: torch.Tensor,
    output: torch.Tensor,
    slots: torch.Tensor,
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    cdf: torch.Tensor,
) -> bool:
    """Write greedy tokens and exact lookup/zero state for independent CUDA rows.

    The caller guarantees an embedding-only prelude and unique real and padding slots.
    Return False without writes when layouts or types require the ordinary model path.
    """
    if not (
        logits.is_cuda
        and logits.ndim == embedding.ndim == hidden.ndim == 2
        and logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and embedding.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and hidden.dtype == embedding.dtype
        and 0 < logits.shape[1] == embedding.shape[0] <= 2147483647
        and hidden.shape[1] == embedding.shape[1] > 0
        and cdf.ndim == slots.ndim == output.ndim == 1
        and cdf.dtype == torch.float32
        and cdf.shape[0] == hidden.shape[0]
        and slots.dtype in (torch.int32, torch.int64)
        and output.dtype == torch.int32
        and output.shape == slots.shape == (logits.shape[0],)
        and all(
            tensor.device == logits.device and tensor.is_contiguous()
            for tensor in (logits, output, slots, embedding, hidden, cdf)
        )
        and len(
            {
                tensor.untyped_storage().data_ptr()
                for tensor in (logits, output, slots, embedding, hidden, cdf)
            }
        )
        == 6
    ):
        return False
    rows, vocab = logits.shape
    if rows:
        chunks = triton.cdiv(vocab, 4096)
        partial_values = torch.empty((rows, chunks), dtype=torch.float32, device=logits.device)
        partial_indices = torch.empty((rows, chunks), dtype=torch.int32, device=logits.device)
        _partial_argmax[(rows, chunks)](
            logits, partial_values, partial_indices, vocab, chunks, 4096, num_warps=4
        )
        _finish_argmax_prelude[(rows,)](
            partial_values,
            partial_indices,
            output,
            slots,
            embedding,
            hidden,
            cdf,
            chunks,
            embedding.shape[1],
            triton.next_power_of_2(chunks),
            triton.next_power_of_2(embedding.shape[1]),
            num_warps=4,
        )
    return True
