from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, List, Literal, Tuple

import torch

if TYPE_CHECKING:
    from loopsgl.attention import BaseAttnBackend, BaseAttnMetadata
    from loopsgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from loopsgl.kvcache.base import RetractionBackup
    from loopsgl.moe import BaseMoeBackend
    from loopsgl.stopping import StopChecker


@dataclass(frozen=True)
class LoopParams:
    loop_steps: int | None = None
    exit_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.loop_steps is not None and (
            isinstance(self.loop_steps, bool)
            or not isinstance(self.loop_steps, int)
            or self.loop_steps <= 0
        ):
            raise ValueError("loop_steps must be a positive integer")


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
    stop: str | List[str] | None = None
    no_stop_trim: bool = False
    skip_special_tokens: bool = False
    replay_exit_depths: List[int] | None = None  # predetermined exit depths for reproducibility
    return_loop_steps: bool = False

    def __post_init__(self) -> None:
        self.normalize()

    def normalize(self) -> None:
        """Normalize stop strings to the representation used by the scheduler."""
        from loopsgl.stopping import normalize_stops

        self.stop = normalize_stops(self.stop)

    def verify(self) -> None:
        """Reject unsupported sampling values before dispatching model work."""
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a nonnegative finite number.")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be in (0, 1].")
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or (self.top_k != -1 and self.top_k < 1)
        ):
            raise ValueError("top_k must be -1 (disable) or a positive integer.")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer.")

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class ReqKvInfo:
    """KV ownership independent of the request's token history and sampling state."""

    req_pool_idx: int | None = None
    kv_committed_len: int = 0
    kv_allocated_len: int = 0
    reclaimed_versions: int = 0  # only for last-exited KV cache policy
    cache_handle: BaseCacheHandle | None = None
    retraction_backup: RetractionBackup | None = None

    @property
    def holds_kv(self) -> bool:
        return self.req_pool_idx is not None

    def reset_device_state(self) -> None:
        """Clear released GPU ownership while retaining a CPU retraction backup."""
        self.req_pool_idx = None
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self.reclaimed_versions = 0  # only for last-exited KV cache policy
        self.cache_handle = None


@dataclass(eq=False)
class Req:
    """Persistent input, output, and execution state across the request lifecycle."""

    uid: int
    origin_input_ids: torch.Tensor  # cpu tensor
    sampling_params: SamplingParams
    output_ids: List[int] = field(default_factory=list)
    kv: ReqKvInfo = field(default_factory=ReqKvInfo)
    fill_ids: torch.Tensor = field(init=False)
    device_len: int = field(default=0, init=False)
    inflight_middle_chunks: int = field(default=0, init=False)
    finished_reason: Literal["length", "eos", "stop", "abort"] | None = field(
        default=None, init=False
    )
    stop_checker: StopChecker | None = field(default=None, init=False)
    is_retracted: bool = field(default=False, init=False)
    retraction_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        assert self.origin_input_ids.is_cpu and self.origin_input_ids.ndim == 1
        assert len(self.origin_input_ids) > 0
        self.prepare_for_prefill()

    @property
    def origin_input_len(self) -> int:
        return len(self.origin_input_ids)

    @property
    def input_len(self) -> int:
        """Length of the current prefill or recomputation history."""
        return len(self.fill_ids)

    @property
    def total_len(self) -> int:
        return self.origin_input_len + len(self.output_ids)

    @property
    def max_device_len(self) -> int:
        return self.origin_input_len + self.sampling_params.max_tokens

    @property
    def output_len(self) -> int:
        return self.max_device_len - self.input_len

    @property
    def num_generated_tokens(self) -> int:
        return len(self.output_ids)

    def get_token_ids(self) -> torch.Tensor:
        """Materialize history only for prefill, prefix insertion, or CPU restore."""
        if not self.output_ids:
            return self.origin_input_ids
        return torch.cat(
            [
                self.origin_input_ids,
                torch.tensor(self.output_ids, dtype=self.origin_input_ids.dtype),
            ]
        )

    def prepare_for_prefill(self) -> None:
        assert not self.kv.holds_kv and self.inflight_middle_chunks == 0
        self.fill_ids = self.get_token_ids()
        self.device_len = 0

    def finished(self) -> bool:
        return self.finished_reason is not None

    def reset_for_retract(self) -> None:
        """Retain token history and output limits while dropping released GPU state."""
        assert self.device_len == self.total_len, "Drain outstanding results before retraction."
        assert self.inflight_middle_chunks == 0
        self.retraction_count += 1
        self.is_retracted = True
        self.kv.reset_device_state()

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.kv.kv_committed_len

    def complete_one(self, *, is_middle_chunk: bool) -> None:
        self.kv.kv_committed_len = self.device_len
        if not is_middle_chunk:
            self.device_len += 1

    def append_host(self, next_token: int) -> None:
        self.output_ids.append(next_token)

    @property
    def can_decode(self) -> bool:
        return not self.finished() and self.device_len >= self.input_len and self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(uid={self.uid}, req_pool_idx={self.kv.req_pool_idx}, "
            f"kv_committed_len={self.kv.kv_committed_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class ForwardBatch:
    """Execution snapshot independent of mutable scheduler request membership."""

    size: int
    phase: Literal["prefill", "decode"]
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    extend_lens: torch.Tensor
    seq_lens_cpu: Tuple[int, ...]
    extend_lens_cpu: Tuple[int, ...]
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    attn_metadata: BaseAttnMetadata = field(init=False)
    metadata_ready: torch.cuda.Event | None = field(default=None, init=False)
    prefill_num_tokens: int | None = field(
        default=None, init=False
    )  # For attention breaks in prefill graphs

    def record_metadata_ready(self) -> None:
        """Fence page-table reads before the scheduler can remap cached prefixes."""
        self.metadata_ready = torch.cuda.Event()
        self.metadata_ready.record()

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def padded_size(self) -> int:
        return len(self.seq_lens_cpu)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: ForwardBatch | None = field(default=None, init=False)

    @property
    def batch(self) -> ForwardBatch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: ForwardBatch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
