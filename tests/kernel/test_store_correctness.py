from __future__ import annotations

import os
from unittest.mock import Mock

import pytest
import torch
from loopsgl.kernel import store as store_module
from loopsgl.kernel.store import _has_supported_layout, can_use_store_cache, store_cache

requires_cuda = pytest.mark.skipif(
    not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available(),
    reason="CUDA KV-store checks require a SLURM allocation",
)


@requires_cuda
@pytest.mark.parametrize(
    ("k_width", "v_width", "num_split", "v_offset"),
    [
        (2048, 2048, 0, 0),
        (2048, 2048, 2, 0),
        (5280, 5280, 0, 0),
        (18, 18, 1, 0),
        (256, 384, 0, 0),
        (384, 256, 0, 0),  # Interleaved prefix with a K tail.
        (256, 258, 0, 0),  # Row widths require serial copies.
        (256, 384, 0, 4),  # V pointers lack common-prefix alignment.
        (384, 640, 0, 4),  # V pointers lack tail alignment.
    ],
)
def test_store_cache_cuda(k_width: int, v_width: int, num_split: int, v_offset: int) -> None:
    """Check row splits, independent strides, padding, and graph replay."""
    k_storage = torch.full((17, 2 * k_width), -7, device="cuda", dtype=torch.bfloat16)
    v_storage = torch.full((17, 3 * v_width + v_offset), -7, device="cuda", dtype=k_storage.dtype)
    k_cache = k_storage[:, :k_width]
    v_cache = v_storage[:, v_offset : v_offset + v_width]
    k = torch.randn(5, 3 * k_width, device="cuda", dtype=k_cache.dtype)[:, :k_width]
    v = torch.randn(5, 4 * v_width + v_offset, device="cuda", dtype=v_cache.dtype)[
        :, v_offset : v_offset + v_width
    ]
    indices = torch.tensor([0, 16, 0, 11, 7], device="cuda", dtype=torch.int32)
    expected_k, expected_v = k_storage.clone(), v_storage.clone()
    valid = indices != 0
    expected_k[indices[valid], :k_width] = k[valid]
    expected_v[indices[valid], v_offset : v_offset + v_width] = v[valid]

    # A compilation failure must fail this test instead of silently using the fallback.
    assert _has_supported_layout(k_cache, v_cache, indices, k, v)
    assert can_use_store_cache(k_width * k.element_size(), v_width * v.element_size())
    store_cache(k_cache, v_cache, indices, k, v, num_split=num_split)
    torch.testing.assert_close(k_storage, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(v_storage, expected_v, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        store_cache(k_cache, v_cache, indices, k, v, num_split=num_split)
    k.add_(1)
    v.sub_(1)
    expected_k[indices[valid], :k_width] = k[valid]
    expected_v[indices[valid], v_offset : v_offset + v_width] = v[valid]
    graph.replay()
    torch.testing.assert_close(k_storage, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(v_storage, expected_v, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_store_cache_fallback(device: str) -> None:
    """Unsupported row layouts preserve untouched slots and accept empty batches."""
    k_storage = torch.full((9, 6), -7, device=device, dtype=torch.bfloat16)
    k_cache = k_storage[:, ::2]
    v_cache = torch.full((9, 5), -7, device=device, dtype=k_cache.dtype)
    k = torch.randn(3, 3, device=device, dtype=k_cache.dtype)
    v = torch.randn(3, 5, device=device, dtype=k_cache.dtype)
    indices = torch.tensor([0, 8, 4], device=device, dtype=torch.int64)
    expected_k = k_storage.clone()
    expected_v = v_cache.clone()
    expected_k[indices[1:], ::2] = k[1:]
    expected_v[indices[1:]] = v[1:]

    store_cache(k_cache, v_cache, indices, k, v)
    store_cache(k_cache, v_cache, indices[:0], k[:0], v[:0])
    torch.testing.assert_close(k_storage, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, expected_v, rtol=0, atol=0)

    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            store_cache(k_cache, v_cache, indices, k, v)
        k.add_(1)
        v.sub_(1)
        expected_k[indices[1:], ::2] = k[1:]
        expected_v[indices[1:]] = v[1:]
        graph.replay()
        torch.testing.assert_close(k_storage, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(v_cache, expected_v, rtol=0, atol=0)


def test_store_cache_eligibility_caches_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable compiler is probed once per K/V row-width pair."""
    compile_module = Mock(side_effect=RuntimeError("Compiler unavailable"))
    monkeypatch.setattr(store_module, "_jit_store_module", compile_module)
    can_use_store_cache.cache_clear()
    try:
        assert not can_use_store_cache(512, 768)
        assert not can_use_store_cache(512, 768)
        assert not can_use_store_cache(6)
        compile_module.assert_called_once_with(512, 768)
    finally:
        can_use_store_cache.cache_clear()
