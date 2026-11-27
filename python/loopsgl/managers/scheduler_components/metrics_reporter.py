# Portions adapted from SGLang's scheduler_components/metrics_reporter.py.
# Copyright 2023-2024 SGLang Team.
# Licensed under the Apache License, Version 2.0.
# See LICENSES/sglang.txt in the repository root.

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from loopsgl.utils import init_logger

if TYPE_CHECKING:
    from loopsgl.managers.base_scheduler import BaseScheduler


logger = init_logger(__name__)


@dataclass(frozen=True)
class PrefillStats:
    """Snapshot admission counts before overlapping forwards advance the requests."""

    log_input_tokens: int
    log_hit_tokens: int
    num_new_seqs: int
    num_running_reqs: int


@dataclass
class SchedulerMetricsReporter:
    """Log prefill batches and periodic decode batches using CPU-side metadata.

    Throughput uses wall time between reports, including intervening scheduler and model work.
    """

    scheduler: BaseScheduler

    def __post_init__(self) -> None:
        self.is_stats_logging_rank = self.scheduler.config.tp_info.is_primary()
        self.decode_log_interval = self.scheduler.config.decode_log_interval
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.forward_ct_decode = 0
        self.forward_ct_recurrent = 0
        self.num_generated_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = self.last_decode_stats_tic

    def _token_usage(self) -> tuple[int, float]:
        cache = self.scheduler.cache_manager
        total = cache.num_pages * cache.page_size
        evictable = cache.prefix_cache.size_info.evictable_size
        available = len(cache.free_slots) * cache.page_size + evictable
        num_used = total - available
        token_usage = num_used / total
        if cache.version_pool is not None:
            # Last-exited KV can exhaust physical versions before logical token handles.
            pool = cache.version_pool
            # Reusable prefixes retain full-depth versions that eviction can release.
            available_slots = pool.available_slots + evictable * pool.slots_per_token
            token_usage = max(token_usage, 1 - available_slots / (pool.num_kv_slots - 1))
        return num_used, token_usage

    def report_prefill_stats(
        self,
        prefill_stats: PrefillStats,
        can_run_cuda_graph: bool,
        num_generated_tokens: int,
    ) -> None:
        """Report each prefill chunk after its token results have been processed."""
        if not self.is_stats_logging_rank:
            return
        self.num_generated_tokens += num_generated_tokens
        now = time.perf_counter()
        gap_latency = now - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = now
        input_throughput = prefill_stats.log_input_tokens / gap_latency if gap_latency > 0 else 0.0
        _, token_usage = self._token_usage()
        logger.info_rank0(
            f"Prefill batch, #new-seq: {prefill_stats.num_new_seqs}, "
            f"#new-token: {prefill_stats.log_input_tokens}, "
            f"#cached-token: {prefill_stats.log_hit_tokens}, "
            f"token usage: {token_usage:.2f}, "
            f"#running-req: {prefill_stats.num_running_reqs}, "
            f"#queue-req: {len(self.scheduler.prefill_manager.pending_list)}, "
            f"cuda graph: {can_run_cuda_graph}, "
            f"input throughput (token/s): {input_throughput:.2f}"
        )

    def report_decode_stats(
        self,
        batch_size: int,
        can_run_cuda_graph: bool,
        num_generated_tokens: int = 0,
        *,
        phase: Literal["decode", "recurrent", "coda"] = "decode",
    ) -> None:
        """Sample actual batch widths every log interval, independently for core and output forwards.

        Generated-token throughput includes completed prefill and decode outputs.
        Recurrent forwards report their width without contributing output tokens or resetting throughput.
        """
        if not self.is_stats_logging_rank:
            return
        if phase == "recurrent":
            self.forward_ct_recurrent += 1
            if self.forward_ct_recurrent % self.decode_log_interval:
                return
            prefix = "Recurrent batch"
        else:
            self.num_generated_tokens += num_generated_tokens
            self.forward_ct_decode += 1
            if self.forward_ct_decode % self.decode_log_interval:
                return
            prefix = "Decode batch" if phase == "decode" else "Coda batch"

        num_running_reqs = (
            batch_size if phase == "decode" else self.scheduler.table_manager.allocated_size
        )
        num_used, token_usage = self._token_usage()
        msg = f"{prefix}, #running-req: {num_running_reqs}, "
        if phase != "decode":
            msg += f"#batch-token: {batch_size}, "
        msg += (
            f"#token: {num_used}, token usage: {token_usage:.2f}, "
            f"cuda graph: {can_run_cuda_graph}, "
        )
        if phase != "recurrent":
            now = time.perf_counter()
            gap_latency = now - self.last_decode_stats_tic
            gen_throughput = self.num_generated_tokens / gap_latency if gap_latency > 0 else 0.0
            self.last_decode_stats_tic = now
            self.num_generated_tokens = 0
            msg += f"gen throughput (token/s): {gen_throughput:.2f}, "
        msg += f"#queue-req: {len(self.scheduler.prefill_manager.pending_list)}"
        logger.info_rank0(msg)
