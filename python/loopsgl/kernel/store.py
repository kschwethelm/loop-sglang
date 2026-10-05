from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang's KV-store operation.
# https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/kernels/ops/kvcache/kvcache.py
# See LICENSES/sglang.txt in the repository root for the license text.
import functools
from typing import TYPE_CHECKING

import torch
from loopsgl.utils import init_logger

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
logger = init_logger(__name__)


@functools.cache
def _jit_store_module(
    k_row_bytes: int,
    v_row_bytes: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(k_row_bytes, v_row_bytes, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


@functools.cache
def can_use_store_cache(k_row_bytes: int, v_row_bytes: int = 0) -> bool:
    """Cache kernel availability for the key and value row widths."""
    v_row_bytes = v_row_bytes or k_row_bytes
    if k_row_bytes <= 0 or v_row_bytes <= 0 or k_row_bytes % 4 or v_row_bytes % 4:
        return False
    try:
        _jit_store_module(k_row_bytes, v_row_bytes)
        return True
    except Exception as error:
        logger.warning(
            f"KV store kernel unavailable for K/V rows of "
            f"{k_row_bytes}/{v_row_bytes} bytes: {error}"
        )
        return False


def _has_supported_layout(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    for cache, value in ((k_cache, k), (v_cache, v)):
        row_bytes = cache.shape[1] * cache.element_size()
        if row_bytes == 0 or row_bytes % 4:
            return False
        alignment = 16 if row_bytes % 512 == 0 else (8 if row_bytes % 256 == 0 else 4)
        if not all(
            tensor.is_cuda
            and tensor.device == k_cache.device
            and tensor.dtype == k_cache.dtype
            and tensor.shape[1] == cache.shape[1]
            and tensor.stride(1) == 1
            and tensor.data_ptr() % alignment == 0
            and tensor.stride(0) * tensor.element_size() % alignment == 0
            for tensor in (cache, value)
        ):
            return False
    return (
        k.shape[0] == v.shape[0] == indices.numel()
        and k_cache.shape[0] == v_cache.shape[0]
        and indices.device == k_cache.device
        and indices.dtype in (torch.int32, torch.int64)
        and indices.ndim == 1
        and indices.stride(0) > 0
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    row_bytes: int = 0,
    v_row_bytes: int = 0,
    num_split: int = 0,
    size_limit: int = 0,
    reserved_skip_index: int = 0,
) -> None:
    """Store K/V rows, preserving the reserved padding slot (0 by default).

    Row widths are inferred in bytes when unspecified.
    A nonpositive num_split selects one, two, or four warps per row automatically.
    A negative reserved_skip_index disables padding-slot preservation.
    """
    if indices.numel() == 0:
        return
    try:
        k_cache_flat = k_cache.view(k_cache.shape[0], -1)
        v_cache_flat = v_cache.view(v_cache.shape[0], -1)
        k_flat = k.view(k.shape[0], -1)
        v_flat = v.view(v.shape[0], -1)
    except RuntimeError:
        pass
    else:
        if _has_supported_layout(k_cache_flat, v_cache_flat, indices, k_flat, v_flat):
            row_bytes = row_bytes or k_flat.shape[1] * k_flat.element_size()
            v_row_bytes = v_row_bytes or v_flat.shape[1] * v_flat.element_size()
            if can_use_store_cache(row_bytes, v_row_bytes):
                module = _jit_store_module(row_bytes, v_row_bytes)
                if num_split <= 0:
                    if row_bytes % 2048 == 0 and v_row_bytes % 2048 == 0:
                        num_split = 4
                    elif row_bytes % 1024 == 0 and v_row_bytes % 1024 == 0:
                        num_split = 2
                    else:
                        num_split = 1
                module.launch(
                    k_cache_flat,
                    v_cache_flat,
                    indices,
                    k_flat,
                    v_flat,
                    num_split,
                    size_limit or k_cache_flat.shape[0],
                    reserved_skip_index,
                )
                return

    size_limit = size_limit or min(k_cache.shape[0], v_cache.shape[0])
    torch._assert_async(((indices >= 0) & (indices < size_limit)).all(), "Invalid KV cache slot")
    k = k.reshape(indices.numel(), *k_cache.shape[1:])
    v = v.reshape(indices.numel(), *v_cache.shape[1:])
    if reserved_skip_index >= 0:
        padding = indices == reserved_skip_index
        # Fixed-size masking keeps the fallback compatible with CUDA graph capture.
        k = torch.where(padding.view(-1, *([1] * (k.ndim - 1))), k_cache[indices], k)
        v = torch.where(padding.view(-1, *([1] * (v.ndim - 1))), v_cache[indices], v)
    k_cache[indices] = k
    v_cache[indices] = v
