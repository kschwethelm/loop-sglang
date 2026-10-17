from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Literal

import torch

from loopsgl.core import LoopParams
from loopsgl.distributed import DistributedInfo
from loopsgl.utils import cached_load_hf_config, get_free_port, parse_model_override_args

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from loopsgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    cuda_graph_backend_prefill: Literal["breakable", "disabled"] = "breakable"
    cuda_graph_bs_prefill: List[int] | None = None
    cuda_graph_max_bs_prefill: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    nccl_port: int | None = None
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    loop_params: LoopParams = field(default_factory=LoopParams)
    json_model_override_args: str = "{}"
    loop_cache_policy: Literal["depth_indexed", "shared", "last_exited"] | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    def __post_init__(self) -> None:
        if self.loop_cache_policy is None:
            object.__setattr__(self, "loop_cache_policy", "depth_indexed")
        _ = self.model_override_args
        if self.cuda_graph_backend_prefill not in {"breakable", "disabled"}:
            raise ValueError("cuda_graph_backend_prefill must be 'breakable' or 'disabled'.")
        if self.cuda_graph_max_bs_prefill is not None and self.cuda_graph_max_bs_prefill < 1:
            raise ValueError("cuda_graph_max_bs_prefill must be positive.")
        if self.cuda_graph_bs_prefill is not None:
            sizes = self.cuda_graph_bs_prefill
            if any(
                isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes
            ):
                raise ValueError("Prefill graph buckets must be positive integers.")
            if len(set(sizes)) != len(sizes):
                raise ValueError("Prefill graph buckets must be unique.")
        port = self.nccl_port
        if port is None:
            port = get_free_port()
        if not 1 <= port <= 65535:
            raise ValueError("nccl_port must be between 1 and 65535.")
        object.__setattr__(self, "nccl_port", port)

    @cached_property
    def model_override_args(self) -> Dict[str, Any]:
        return parse_model_override_args(self.json_model_override_args)

    @cached_property
    def hf_config(self) -> PretrainedConfig:
        return cached_load_hf_config(self.model_path, self.model_override_args)

    @cached_property
    def model_config(self) -> ModelConfig:
        from loopsgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config, loop_params=self.loop_params)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.nccl_port}"
