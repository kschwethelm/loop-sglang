from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from loopsgl.model_executor import EngineConfig


def _get_pid_suffix() -> str:
    import os
    import uuid

    return f".pid={os.getpid()}.{uuid.uuid4().hex}"


@dataclass(frozen=True)
class CDBConfig:
    refill: bool = True
    min_coda_batch_size: int = 1
    disable_lookahead_exit: bool = False

    def __post_init__(self) -> None:
        if self.min_coda_batch_size < 1:
            raise ValueError("min_coda_batch_size must be positive.")


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    scheduler: str | None = None
    cdb_config: CDBConfig = field(default_factory=CDBConfig)
    max_extend_tokens: int = 8192
    min_free_slots_delay: int | None = None
    watermark: float = 0.0
    retraction_mode: str = "offload"
    retraction_cpu_memory_gb: float = 16.0
    disable_radix_cache: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    def __post_init__(self) -> None:
        if self.scheduler is None:
            object.__setattr__(
                self, "scheduler", "cdb" if self.loop_params.exit_threshold is not None else "cb"
            )
        if self.loop_cache_policy is None and self.scheduler == "cdb":
            object.__setattr__(self, "loop_cache_policy", "last_exited")
        super().__post_init__()
        if self.scheduler not in {"cb", "cdb"}:
            raise ValueError("scheduler must be 'cb' or 'cdb'.")
        if self.scheduler == "cb" and self.loop_params.exit_threshold is not None:
            raise ValueError(
                "Early exit requires scheduler='cdb'; use --scheduler cdb or omit --scheduler."
            )
        if self.scheduler == "cdb" and self.loop_cache_policy not in {"shared", "last_exited"}:
            raise ValueError("CDB requires loop_cache_policy='shared' or 'last_exited'.")
        if self.loop_cache_policy == "last_exited" and self.page_size != 1:
            raise ValueError("Last-exited KV requires one-token pages.")
        if self.max_running_req < 1:
            raise ValueError("max_running_req must be positive.")
        if not 0.0 <= self.watermark < 1.0:
            raise ValueError("watermark must be in [0.0, 1.0).")
        if self.retraction_mode not in {"none", "recompute", "offload"}:
            raise ValueError("retraction_mode must be 'none', 'recompute', or 'offload'.")
        if not isfinite(self.retraction_cpu_memory_gb) or self.retraction_cpu_memory_gb <= 0:
            raise ValueError("retraction_cpu_memory_gb must be finite and positive.")
        if self.min_free_slots_delay is not None:
            threshold = min(self.min_free_slots_delay, self.max_running_req)
            object.__setattr__(self, "min_free_slots_delay", threshold if threshold > 1 else None)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/loopsgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/loopsgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/loopsgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/loopsgl_3" + self._unique_suffix
