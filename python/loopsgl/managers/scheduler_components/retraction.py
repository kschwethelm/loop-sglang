# Backup and restore boundaries adapted from SGLang's mem_cache/common.py
# and disaggregation/decode.py.
# Copyright 2023-2024 SGLang Team.
# Licensed under the Apache License, Version 2.0.
# See LICENSES/sglang.txt in the repository root.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from loopsgl.kvcache.base import RetractionBackup

if TYPE_CHECKING:
    from loopsgl.core import Req
    from loopsgl.kvcache import BaseKVCachePool
    from loopsgl.managers.scheduler_components.cache import CacheManager
    from loopsgl.managers.scheduler_components.decode import DecodeManager
    from loopsgl.managers.scheduler_components.table import TableManager


@dataclass
class RetractionManager:
    kv_cache: BaseKVCachePool
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    max_cpu_bytes: int
    watermark: float = 0.0
    min_free_slots_delay: int | None = None
    retracted_queue: List[Req] = field(default_factory=list)
    cpu_bytes: int = 0
    peak_cpu_bytes: int = 0
    num_offloaded_reqs: int = 0
    offloaded_kv_bytes: int = 0
    restored_kv_bytes: int = 0

    def retraction_backup(self, req: Req) -> None:
        """Preserve all valid KV before the request's GPU resources are released."""
        assert req.kv.retraction_backup is None
        assert req.kv.kv_committed_len == req.device_len - 1
        assert req.device_len == req.total_len, "Drain outstanding results before offloading."
        needed_bytes = self.kv_cache.cpu_copy_size(req.kv.kv_committed_len)
        if needed_bytes > self.max_cpu_bytes - self.cpu_bytes:
            raise RuntimeError(
                f"CPU KV backup budget exhausted for request {req.uid}: "
                f"{needed_bytes} bytes required, {self.cpu_bytes} in use, "
                f"{self.max_cpu_bytes} limit. Increase --retraction-cpu-memory-gb."
            )
        indices = self.cache_manager.page_table[req.kv.req_pool_idx, : req.kv.kv_committed_len]
        backup = RetractionBackup(self.kv_cache.get_cpu_copy(indices), req.kv.kv_committed_len)
        assert backup.nbytes == needed_bytes
        req.kv.retraction_backup = backup
        self.retracted_queue.append(req)
        self.cpu_bytes += backup.nbytes
        self.peak_cpu_bytes = max(self.peak_cpu_bytes, self.cpu_bytes)
        self.num_offloaded_reqs += 1
        self.offloaded_kv_bytes += backup.nbytes

    def retraction_restore(self, req: Req) -> None:
        """Restore private KV pages and the pending decode token without re-prefill."""
        backup = req.kv.retraction_backup
        assert backup is not None and req.is_retracted and not req.kv.holds_kv
        assert self.table_manager.available_size > 0
        # Token-only prefix matching cannot guarantee the exact looped-model KV snapshot.
        handle = self.cache_manager.prefix_cache.match_prefix(req.origin_input_ids[:0]).cuda_handle
        self.cache_manager.lock(handle)
        req.kv.req_pool_idx = self.table_manager.allocate()
        req.kv.cache_handle = handle
        try:
            self.cache_manager.allocate_for_restore(req, backup.cached_len, backup.cpu_tensors)
            self.table_manager.token_pool[req.kv.req_pool_idx, : req.device_len].copy_(
                req.get_token_ids().pin_memory(), non_blocking=True
            )
            indices = self.cache_manager.page_table[req.kv.req_pool_idx, : req.kv.kv_committed_len]
            self.kv_cache.load_cpu_copy(backup.cpu_tensors, indices)
        except Exception:
            self.cache_manager.release_req(req)
            self.table_manager.free(req.kv.req_pool_idx)
            req.kv.reset_device_state()
            raise
        self.restored_kv_bytes += backup.nbytes
        self.retraction_discard(req)
        req.is_retracted = False
        self.decode_manager.filter_reqs([req])

    def resume_retracted_reqs(self) -> List[Req]:
        """Readmit offloaded requests when their snapshot and next decode step fit."""
        if (
            self.min_free_slots_delay is not None
            and self.min_free_slots_delay > 1
            and self.decode_manager.runnable
            and self.table_manager.available_size < self.min_free_slots_delay
        ):
            return []
        resumed: List[Req] = []
        while self.retracted_queue and self.table_manager.available_size > 0:
            req = self.retracted_queue[0]
            page_size = self.cache_manager.page_size
            reserved_size = (
                self.cache_manager.new_pages_required(self.decode_manager.running_reqs) * page_size
            )
            watermark = self.watermark if self.table_manager.allocated_size > 0 else 0.0
            if not self.cache_manager.restore_fits(req, reserved_size, watermark):
                if self.table_manager.allocated_size == 0:
                    self.cache_manager.check_restore_capacity(req, reserved_size, watermark)
                break
            self.retraction_restore(req)
            resumed.append(req)
        return resumed

    def retraction_discard(self, req: Req) -> None:
        """Drop a backup after restore or cancellation and release its CPU budget."""
        backup = req.kv.retraction_backup
        if backup is not None:
            self.retracted_queue.remove(req)
            self.cpu_bytes -= backup.nbytes
            req.kv.retraction_backup = None

    def abort_req(self, uid: int) -> bool:
        for req in self.retracted_queue:
            if req.uid == uid:
                req.finished_reason = "abort"
                self.retraction_discard(req)
                return True
        return False

    def clear(self) -> None:
        for req in list(self.retracted_queue):
            self.retraction_discard(req)

    def reset_metrics(self) -> None:
        if self.runnable:
            raise RuntimeError("Cannot reset retraction metrics while CPU backups are in use.")
        self.peak_cpu_bytes = 0
        self.num_offloaded_reqs = 0
        self.offloaded_kv_bytes = 0
        self.restored_kv_bytes = 0

    @property
    def runnable(self) -> bool:
        return len(self.retracted_queue) > 0
