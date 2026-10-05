from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import torch
from loopsgl.utils import init_logger

if TYPE_CHECKING:
    from loopsgl.engine.config import EngineConfig

logger = init_logger(__name__)


def _unsupported_reason(
    backend: str, config: EngineConfig, capability: tuple[int, int], page_size: int
) -> str | None:
    head_dim = config.model_config.head_dim
    major, _ = capability
    cuda_version = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
    if backend == "fa":
        if major not in (8, 9, 10, 11):
            return "FlashAttention requires a supported SM8x, SM9x, SM10x, or SM11x GPU"
        if head_dim % 8 or not 0 < head_dim <= 256:
            return "FlashAttention requires head dimensions divisible by 8 and at most 256"
        if major >= 10 and page_size != 128:
            # FA4 uses TMA for page size 128; other sizes restrict head dimensions.
            return "FA4 paged attention requires a page size of 128"
        if major == 10 and head_dim > 128:
            # The pinned FA4 kernel exceeds TMEM capacity with two Q stages.
            return "FA4 SM10x prefill requires head dimensions at most 128"
        if major == 11 and head_dim > 176:
            # Automatic SplitKV rejects head dimensions rounded to 192 or greater.
            return "FA4 split KV requires head dimensions at most 176"
        minimum_cuda = (12, 8) if major >= 10 else (12, 3)
        if cuda_version < minimum_cuda:
            return f"FlashAttention requires CUDA {minimum_cuda[0]}.{minimum_cuda[1]} or newer"
    elif backend == "fi":
        if major < 8:
            return "FlashInfer requires SM80 or newer for FP16/BF16 attention"
        if head_dim not in (64, 128, 256, 512):
            return "FlashInfer FA2 requires head dimensions of 64, 128, 256, or 512"
    elif backend == "trtllm":
        if major != 10:
            # FlashInfer 0.6.6 auto decode selects TRTLLM-gen only on SM10x.
            return "TRTLLM auto decode requires an SM10x GPU"
        if head_dim not in (64, 128, 256):
            return "TRTLLM requires head dimensions of 64, 128, or 256"
        if page_size not in (16, 32, 64):
            return "TRTLLM requires a page size of 16, 32, or 64"

    try:
        if backend == "fa":
            module = importlib.import_module("sgl_kernel.flash_attn")
            if major >= 10 and module.flash_attn_varlen_func_v4 is None:
                return "FA4 is unavailable in the installed sgl-kernel"
        elif backend == "fi":
            module = importlib.import_module("flashinfer")
            if not all(
                hasattr(module, name)
                for name in (
                    "BatchPrefillWithPagedKVCacheWrapper",
                    "BatchDecodeWithPagedKVCacheWrapper",
                )
            ):
                return "FlashInfer paged attention wrappers are unavailable"
        elif backend == "trtllm":
            decode = importlib.import_module("flashinfer.decode")
            prefill = importlib.import_module("flashinfer.prefill")
            if not hasattr(decode, "trtllm_batch_decode_with_kv_cache") or not hasattr(
                prefill, "trtllm_batch_context_with_kv_cache"
            ):
                return "FlashInfer TRTLLM kernels are unavailable"
    except (ImportError, OSError, RuntimeError) as error:
        return f"backend import failed: {error}"
    return None


def select_attention_backend(config: EngineConfig) -> tuple[str, int]:
    """Resolve automatic selection and validate explicit prefill/decode backends."""
    from . import validate_attn_backend

    validate_attn_backend(config.attention_backend)
    model = config.model_config
    if config.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Attention backends require FP16 or BF16 model and KV-cache dtype.")
    if config.page_size <= 0:
        raise ValueError("Attention page size must be positive.")
    tp_size = config.tp_info.size
    if (
        model.num_qo_heads % tp_size
        or model.num_qo_heads % model.num_kv_heads
        or (model.num_kv_heads % tp_size and tp_size % model.num_kv_heads)
    ):
        raise ValueError("Query/KV head counts are incompatible with tensor parallelism.")

    capability = torch.cuda.get_device_capability(config.tp_info.rank)
    automatic = config.attention_backend == "auto"
    candidates = (
        ["trtllm", "fa", "fi"]
        if capability[0] == 10
        else (["fa", "fi"] if capability[0] in (9, 11) else ["fi", "fa"])
    )
    if not automatic:
        candidates = [config.attention_backend]

    rejected: list[str] = []
    for candidate in candidates:
        page_size = config.page_size
        if automatic and candidate == "trtllm" and page_size not in (16, 32, 64):
            page_size = 64
        elif automatic and capability[0] >= 10 and "fa" in candidate.split(","):
            page_size = 128
        reasons = [
            f"{backend}: {reason}"
            for backend in candidate.split(",")
            if (reason := _unsupported_reason(backend, config, capability, page_size))
        ]
        if not reasons:
            mode = "Auto-selected" if automatic else "Selected"
            logger.info_rank0(
                f"{mode} attention backend: {candidate} "
                f"(SM{capability[0]}{capability[1]}, head_dim={model.head_dim}, "
                f"dtype={config.dtype}, page_size={page_size})"
            )
            return candidate, page_size
        rejected.extend(reasons)
        if automatic:
            logger.info_rank0(f"Skipping attention backend {candidate}: {'; '.join(reasons)}")

    raise ValueError(f"No compatible attention backend: {'; '.join(rejected)}")
