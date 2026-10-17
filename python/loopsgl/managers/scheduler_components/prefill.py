from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

from loopsgl.core import Req
from loopsgl.managers.schedule_batch import ScheduleBatch
from loopsgl.utils import div_ceil, init_logger

if TYPE_CHECKING:
    from loopsgl.kvcache import BaseCacheHandle
    from loopsgl.managers.io_struct import TokenizedGenerateReqInput
    from loopsgl.managers.scheduler_components.cache import CacheManager
    from loopsgl.managers.scheduler_components.decode import DecodeManager
    from loopsgl.managers.scheduler_components.table import TableManager

logger = init_logger(__name__)


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    watermark_size: int
    has_running_reqs: bool
    reserve_decode: bool = False

    def _decode_headroom(self, req: Req) -> int:
        return (
            int(
                self.reserve_decode
                and req.output_len > 1
                and req.input_len % self.cache_manager.page_size == 0
            )
            * self.cache_manager.page_size
        )

    def _try_allocate_one(self, req: Req) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        page_size = self.cache_manager.page_size
        needed_pages = div_ceil(req.input_len, page_size) - div_ceil(cached_len, page_size)
        watermark_size = self.watermark_size if self.has_running_reqs else 0
        decode_size = self._decode_headroom(req) if self.has_running_reqs else 0
        required_size = needed_pages * page_size + self.reserved_size + watermark_size + decode_size
        if self.has_running_reqs and required_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if not self.has_running_reqs:
            try:
                self.cache_manager.check_capacity(needed_pages)
            except RuntimeError:
                self.cache_manager.unlock(handle)
                raise
        if required_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.fill_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        req: Req,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        remain_len = req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        self.token_budget -= chunk_size
        # Budget the full remaining prompt so later admissions cannot strand a chunked prefill.
        page_size = self.cache_manager.page_size
        self.reserved_size += (
            div_ceil(req.input_len, page_size) - div_ceil(cached_len, page_size)
        ) * page_size + self._decode_headroom(req)
        self.has_running_reqs = True
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(req.fill_ids[_slice].pin_memory(), non_blocking=True)
        req.kv.req_pool_idx = table_idx
        req.kv.cache_handle = cache_handle
        req.kv.kv_committed_len = cached_len
        req.kv.kv_allocated_len = max(req.kv.kv_allocated_len, cached_len)
        req.device_len = cached_len + chunk_size
        if is_chunked:
            req.inflight_middle_chunks += 1
        return req

    def try_add_one(self, req: Req) -> Req | None:
        if self.token_budget <= 0:
            return None

        if req.kv.holds_kv:
            assert req.kv.cache_handle is not None and req.kv.req_pool_idx is not None
            return self._add_one_req(
                req=req,
                cache_handle=req.kv.cache_handle,
                table_idx=req.kv.req_pool_idx,
                cached_len=req.kv.kv_committed_len,
            )

        if resource := self._try_allocate_one(req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                req=req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    min_free_slots_delay: int | None = None
    watermark: float = 0.0
    reserve_decode: bool = False
    pending_list: List[Req] = field(default_factory=list)

    def add_one_req(self, req: TokenizedGenerateReqInput) -> None:
        self.pending_list.append(Req(req.uid, req.input_ids, req.sampling_params))

    def add_retracted_req(self, req: Req) -> None:
        assert req.is_retracted and not req.kv.holds_kv
        req.prepare_for_prefill()
        self.pending_list.append(req)

    def schedule_next_batch(
        self,
        prefill_budget: int,
        *,
        reserved_size: int = 0,
        allow_new_requests: bool = True,
    ) -> ScheduleBatch | None:
        if len(self.pending_list) == 0:
            return None
        # Chunked prefills stay at the front and already own a request slot.
        if (
            self.min_free_slots_delay is not None
            and self.min_free_slots_delay > 1
            and self.decode_manager.runnable
            and self.table_manager.available_size < self.min_free_slots_delay
            and not self.pending_list[0].kv.holds_kv
        ):
            return None

        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=reserved_size,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            watermark_size=int(self.watermark * self.cache_manager.num_pages)
            * self.cache_manager.page_size,
            has_running_reqs=self.table_manager.allocated_size > 0,
            reserve_decode=self.reserve_decode,
        )
        reqs: List[Req] = []
        chunked_list: List[Req] = []
        for pending_req in self.pending_list:
            if not allow_new_requests and not pending_req.kv.holds_kv:
                break
            if req := adder.try_add_one(pending_req):
                if req.device_len < req.input_len:
                    chunked_list.append(req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return ScheduleBatch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
