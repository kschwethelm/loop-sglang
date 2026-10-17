from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch

from loopsgl.attention import create_attention_backend
from loopsgl.attention.selector import select_attention_backend
from loopsgl.core import Context, ForwardBatch, Req, SamplingParams, set_global_ctx
from loopsgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from loopsgl.kvcache import create_kvcache_pool
from loopsgl.layers import set_rope_device
from loopsgl.models import create_model, load_weight
from loopsgl.models.looped.base import BaseLoopedModel
from loopsgl.moe import create_moe_backend
from loopsgl.utils import div_even, init_logger, profile_range, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .prefill_graph import PrefillGraphRunner
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class ModelRunner:
    """Execute model batches and own GPU resources for one tensor-parallel rank."""

    def __init__(self, config: EngineConfig) -> None:
        if config.loop_cache_policy not in ("depth_indexed", "shared", "last_exited"):
            raise ValueError(
                "loop_cache_policy must be 'depth_indexed', 'shared', or 'last_exited'"
            )
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        _adjust_config(config)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.num_kv_layers = config.model_config.num_layers
        if isinstance(self.model.model, BaseLoopedModel):
            self.model.model.loop_cache_policy = config.loop_cache_policy
            self.num_kv_layers = self.model.model.num_kv_layers
            logger.info_rank0(f"Loop steps: {self.model.model.loop_steps}")
        if config.loop_cache_policy == "last_exited" and not isinstance(
            self.model.model, BaseLoopedModel
        ):
            raise ValueError("Last-exited KV requires a looped model")
        self.model.load_state_dict(self._load_weight_state_dict(config))
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)
        self._init_runtime_buffers(config, init_free_memory)
        self.prefill_graph_runner = PrefillGraphRunner(
            model=self.model,
            config=config,
            device=self.device,
            stream=self.stream,
            tp_cpu_group=self.tp_cpu_group,
        )

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        pool_kwargs: dict[str, Any] = {}
        if config.loop_cache_policy == "last_exited":
            assert isinstance(self.model.model, BaseLoopedModel)
            pool_kwargs = {
                "core_start": self.model.model.num_prelude_layers,
                "core_end": self.model.model.num_prelude_layers + self.model.model.num_core_layers,
                "loop_steps": self.model.model.loop_steps,
            }
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_kv_layers=self.num_kv_layers,
            num_pages=self.num_pages + 1,  # +1 for reserved padding page 0
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
            **pool_kwargs,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            uid=-1,
            origin_input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            sampling_params=SamplingParams(max_tokens=1),
        )
        self.dummy_req.device_len = 1
        self.dummy_req.kv.req_pool_idx = config.max_running_req
        self.page_table[self.dummy_req.kv.req_pool_idx].zero_()  # point to padding slot 0
        self.graph_runner = self._create_graph_runner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_runtime_buffers(self, config: EngineConfig, free_memory: int) -> None:
        """Allocate model execution state before computing the KV memory budget."""

    def _create_graph_runner(self, **kwargs: Any) -> GraphRunner:
        return GraphRunner(**kwargs)

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        logger.debug_rank0(f"Initializing distributed communication at {config.distributed_addr}")
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: v.to(self.dtype)
                for k, v in load_weight(
                    config.model_path, self.device, model_config=config.model_config
                )
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[0]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * self.num_kv_layers
        )
        if config.loop_cache_policy == "last_exited":
            assert isinstance(self.model.model, BaseLoopedModel)
            from loopsgl.kvcache.last_exited import LastExitedKVCache

            cache_per_page = LastExitedKVCache.memory_per_page(
                self.num_kv_layers,
                self.model.model.num_core_layers,
                self.model.model.loop_steps,
                cache_per_page // (2 * self.num_kv_layers),
            )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page - 1  # reserve padding page 0

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = (num_pages + 1) * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: ForwardBatch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with profile_range("cb.compute.prefill" if batch.is_prefill else "cb.compute.decode"):
            logits = self._forward_logits(batch)
            next_tokens_gpu = self._sample_logits(logits[: batch.size], args)
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
            copy_done_event = torch.cuda.Event()
            copy_done_event.record(self.stream)
            return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def _forward_logits(self, batch: ForwardBatch) -> torch.Tensor:
        """Share prefill execution between ordinary and depth-adaptive serving."""
        if self.prefill_graph_runner.bucket_for(batch) is not None:
            return self.prefill_graph_runner.replay(batch, self.attn_backend)
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                return self.graph_runner.replay(batch)
            self.attn_backend.prepare_metadata(batch)
            batch.record_metadata_ready()
            return self.model.forward()

    def _sample_logits(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        return self.sampler.sample(logits, args).to(torch.int32)

    def shutdown(self) -> None:
        self.prefill_graph_runner.destroy()
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    backend, page_size = select_attention_backend(config)
    override("attention_backend", backend)
    if page_size != config.page_size:
        override("page_size", page_size)
        logger.info_rank0(f"Auto-selected page size: {page_size}")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
