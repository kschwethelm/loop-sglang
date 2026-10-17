from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from loopsgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


def create_kvcache_pool(
    model_config: ModelConfig,
    num_kv_layers: int,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    core_start: int | None = None,
    core_end: int | None = None,
    loop_steps: int | None = None,
) -> BaseKVCachePool:
    if core_start is not None:
        from .last_exited import LastExitedKVCache

        assert core_end is not None and loop_steps is not None
        return LastExitedKVCache(
            num_kv_heads=model_config.num_kv_heads,
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_kv_layers,
            head_dim=model_config.head_dim,
            device=device,
            dtype=dtype,
            core_start=core_start,
            core_end=core_end,
            loop_steps=loop_steps,
        )
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_kv_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


def create_prefix_cache(device: torch.device, disable_radix_cache: bool = False) -> BasePrefixCache:
    """Create the radix cache or its no-prefix-reuse implementation."""
    if disable_radix_cache:
        from .no_prefix_cache import NoPrefixCache

        return NoPrefixCache(device=device)
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
]
