from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from loopsgl.attention import selector


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Provide backend capabilities without loading GPU libraries or HF metadata."""
    module = SimpleNamespace(
        flash_attn_varlen_func_v4=object(),
        BatchPrefillWithPagedKVCacheWrapper=object(),
        BatchDecodeWithPagedKVCacheWrapper=object(),
        trtllm_batch_decode_with_kv_cache=object(),
        trtllm_batch_context_with_kv_cache=object(),
    )
    monkeypatch.setattr(selector, "importlib", SimpleNamespace(import_module=lambda name: module))
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(selector.logger, "info_rank0", lambda message: None)
    return SimpleNamespace(
        attention_backend="auto",
        page_size=1,
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(rank=0, size=1),
        model_config=SimpleNamespace(head_dim=128, num_qo_heads=2, num_kv_heads=2),
    )


@pytest.mark.parametrize(
    ("capability", "head_dim", "expected"),
    [
        ((8, 0), 128, ("fi", 1)),
        ((9, 0), 128, ("fa", 1)),
        ((9, 0), 96, ("fa", 1)),  # Huginn's head dimension is unsupported by FI FA2.
        ((10, 0), 128, ("trtllm", 64)),
        ((10, 0), 96, ("fa", 128)),
        ((11, 0), 128, ("fa", 128)),
        ((11, 0), 136, ("fa", 128)),
    ],
)
def test_auto_attention_backend(
    config: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    head_dim: int,
    expected: tuple[str, int],
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    config.model_config.head_dim = head_dim
    assert selector.select_attention_backend(config) == expected


def test_missing_attention_backend_falls_back(
    config: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_module = selector.importlib.import_module

    def unavailable_flash_attention(name: str) -> SimpleNamespace:
        if name == "sgl_kernel.flash_attn":
            raise ImportError("FlashAttention is unavailable")
        return import_module(name)

    monkeypatch.setattr(selector.importlib, "import_module", unavailable_flash_attention)
    assert selector.select_attention_backend(config) == ("fi", 1)


@pytest.mark.parametrize(
    ("backend", "capability", "head_dim", "page_size", "reason"),
    [
        ("fa,fi", (9, 0), 96, 1, "FlashInfer FA2 requires head dimensions"),
        ("trtllm", (10, 0), 128, 1, "TRTLLM requires a page size"),
        ("fa", (10, 0), 136, 1, "FA4 paged attention requires a page size"),
        ("fa", (10, 0), 136, 128, "FA4 SM10x prefill requires head dimensions"),
        ("fa", (11, 0), 192, 128, "FA4 split KV requires head dimensions"),
        ("trtllm", (11, 0), 128, 64, "TRTLLM auto decode requires an SM10x GPU"),
    ],
)
def test_explicit_attention_backend_rejects_unsupported_config(
    config: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    capability: tuple[int, int],
    head_dim: int,
    page_size: int,
    reason: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    config.attention_backend = backend
    config.model_config.head_dim = head_dim
    config.page_size = page_size
    with pytest.raises(ValueError, match=reason):
        selector.select_attention_backend(config)
