from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Tuple

import torch

from loopsgl.core import Req
from loopsgl.kernel.batch import assign_req_to_token_pool_func
from loopsgl.kvcache import BaseCacheHandle, BaseKVCachePool, MatchResult, create_prefix_cache
from loopsgl.kvcache.last_exited import LastExitedKVCache
from loopsgl.model_executor.sample import make_device_tensor
from loopsgl.utils import div_ceil


class CacheManager:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        disable_radix_cache: bool = False,
        kv_cache: BaseKVCachePool | None = None,
    ) -> None:
        # Page 0 is reserved for padding; each free slot is a page-aligned start.
        device = page_table.device
        self.free_slots = (
            torch.arange(1, num_pages + 1, dtype=torch.int32, device=device) * page_size
        )
        self.prefix_cache = create_prefix_cache(
            device=device, disable_radix_cache=disable_radix_cache
        )
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self.version_pool = kv_cache if isinstance(kv_cache, LastExitedKVCache) else None

    def match_req(self, req: Req) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.fill_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.get_available_size()

    def _watermark_size(self, watermark: float) -> int:
        """Count reserved physical slots, or tokens for a fixed-layout KV pool."""
        if self.version_pool is not None:
            return int(watermark * (self.version_pool.num_kv_slots - 1))
        return int(watermark * self.num_pages) * self.page_size

    def get_available_size(self, watermark: float = 0.0) -> int:
        """Count full-depth token capacity after reserving physical KV headroom."""
        logical = self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size
        watermark_size = self._watermark_size(watermark)
        if self.version_pool is None:
            return logical - watermark_size
        return min(
            logical,
            (self.version_pool.available_slots - watermark_size)
            // self.version_pool.slots_per_token
            + self.prefix_cache.size_info.evictable_size,
        )

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def check_capacity(self, needed_pages: int, *, num_running_reqs: int = 0) -> None:
        free_pages = len(self.free_slots)
        evictable_pages = self.prefix_cache.size_info.evictable_size // self.page_size
        available_pages = self.available_size // self.page_size
        if needed_pages > available_pages:
            raise RuntimeError(
                f"KV cache exhausted: requested {needed_pages} pages, "
                f"{available_pages} available ({free_pages} free, {evictable_pages} evictable), "
                f"{self.num_pages} total, {num_running_reqs} running requests."
            )

    def new_pages_required(self, reqs: Iterable[Req]) -> int:
        """Count page growth for the next forward, using the allocation boundaries."""
        return sum(
            max(
                0,
                div_ceil(req.device_len, self.page_size)
                - div_ceil(req.kv.kv_allocated_len, self.page_size),
            )
            for req in reqs
        )

    def check_decode_mem(self, reqs: Iterable[Req]) -> bool:
        return self.new_pages_required(reqs) * self.page_size <= self.available_size

    def release_req(self, req: Req) -> None:
        """Release request-owned pages and prefix locks without inserting a prefix."""
        handle = req.kv.cache_handle
        assert handle is not None
        allocated_len = req.kv.kv_allocated_len
        self._free(
            self.page_table[req.kv.req_pool_idx, handle.cached_len : allocated_len],
            reclaimed_versions=req.kv.reclaimed_versions,
        )
        req.kv.reclaimed_versions = 0
        self.unlock(handle)

    def allocate_paged(self, reqs: List[Req], *, num_running_reqs: int = 0) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.kv.kv_allocated_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.kv.req_pool_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(
                self._allocate(needed_pages, num_running_reqs=num_running_reqs)
            )
            if self.version_pool is not None:
                self.version_pool.reserve_versions(allocated)
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
        for req in reqs:
            req.kv.kv_allocated_len = max(
                req.kv.kv_allocated_len, div_ceil(req.device_len, self.page_size) * self.page_size
            )

    def allocate_for_restore(
        self, req: Req, cached_len: int, backup: torch.Tensor | None = None
    ) -> None:
        """Allocate private pages for restored KV; the next decode allocates its own growth."""
        assert req.kv.kv_committed_len == 0 and cached_len < req.device_len
        num_pages = div_ceil(cached_len, self.page_size)
        physical_slots = None
        if self.version_pool is not None:
            assert backup is not None
            physical_slots = self.version_pool.restore_slots(backup, cached_len)
        allocated = self._page_to_token(self._allocate(num_pages, physical_slots=physical_slots))
        if self.version_pool is not None:
            req.kv.reclaimed_versions = self.version_pool.reserve_restore_versions(
                allocated, backup
            )
        try:
            _write_page_table(
                self.page_table, allocated, [(req.kv.req_pool_idx, 0, num_pages)], self.page_size
            )
        except Exception:
            self._free(allocated, reclaimed_versions=req.kv.reclaimed_versions)
            req.kv.reclaimed_versions = 0
            raise
        req.kv.kv_committed_len = cached_len
        req.kv.kv_allocated_len = num_pages * self.page_size

    def cache_req(self, req: Req, *, finished: bool, insert_limit: int | None = None) -> None:
        # The old handle protects a tree-owned prefix; the request owns the remaining pages.
        # Insert only committed tokens and free duplicate pages already inserted by other requests.
        # Completion also frees the private tail through the page-aligned allocation boundary.
        insert_len = req.kv.kv_committed_len
        if insert_limit is not None:
            insert_len = min(insert_len, insert_limit)
            insert_len = insert_len // self.page_size * self.page_size
        insert_ids = req.get_token_ids()[:insert_len]
        assert len(insert_ids) == insert_len
        page_indices = self.page_table[req.kv.req_pool_idx, :insert_len]
        old_handle = req.kv.cache_handle
        assert old_handle is not None
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        duplicate_indices = page_indices[old_handle.cached_len : cached_len]
        if not finished and cached_len > old_handle.cached_len:
            # Deferred frees must retain the old indices before the page table is remapped.
            # In-flight batches retain their own page-table snapshots.
            self._free(duplicate_indices.clone())
            page_indices[old_handle.cached_len : cached_len].copy_(
                new_handle.get_matched_indices()[old_handle.cached_len : cached_len]
            )
        else:
            self._free(duplicate_indices)
        if finished:  # this tail part should be freed
            self._free(
                self.page_table[
                    req.kv.req_pool_idx, new_handle.cached_len : req.kv.kv_allocated_len
                ],
                reclaimed_versions=req.kv.reclaimed_versions,
            )
            req.kv.reclaimed_versions = 0
        else:  # keep the tail part, update the handle
            req.kv.cache_handle = new_handle
            self.lock(new_handle)

    def clear(self) -> None:
        """Evict all prefixes and return their pages to the pool when no requests own KV."""
        if (
            len(self.free_slots) * self.page_size + self.prefix_cache.size_info.evictable_size
            != self.num_pages * self.page_size
        ):
            raise RuntimeError("Cannot clear the KV cache while request pages are in use.")
        self._free(self.prefix_cache.evict(self.prefix_cache.size_info.evictable_size))
        self.check_integrity()

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.version_pool is not None:
            pool = self.version_pool
            expected = pool.num_kv_slots - 1 - cache_pages * pool.slots_per_token
            if pool.available_slots != expected or int(pool._free_count.item()) != expected:
                raise RuntimeError("Shared KV slot accounting failed.")
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)
        assert torch.all(self.free_slots >= self.page_size), "Padding page cannot be allocated."

    def finalize_last_exited(self, reqs: List[Req], depths: List[int]) -> None:
        """Publish completed decode depths before freed rows can be reused."""
        pool = self.version_pool
        if pool is None:
            return
        if len(reqs) != len(depths) or any(
            depth < 1 or depth > pool.loop_steps for depth in depths
        ):
            raise ValueError("Invalid completed depth metadata.")
        early = [
            (req, depth) for req, depth in zip(reqs, depths, strict=True) if depth < pool.loop_steps
        ]
        if not early:
            return
        metadata_host = torch.tensor(
            [(req.kv.req_pool_idx, req.device_len - 1, depth) for req, depth in early],
            dtype=torch.int64,
            pin_memory=self.device.type == "cuda",
        )
        metadata = metadata_host.to(self.device, non_blocking=True)
        handles = self.page_table[metadata[:, 0], metadata[:, 1]]
        released = sum(pool.loop_steps - depth for _, depth in early)
        pool.alias_versions(handles, metadata[:, 2], released)
        for req, depth in early:
            req.kv.reclaimed_versions += pool.loop_steps - depth

    def restore_fits(self, req: Req, reserved_size: int, watermark: float) -> bool:
        """Check independent logical and physical budgets without token rounding."""
        evictable = self.prefix_cache.size_info.evictable_size
        logical_available = len(self.free_slots) * self.page_size + evictable
        logical_needed = div_ceil(req.device_len, self.page_size) * self.page_size
        watermark_size = self._watermark_size(watermark)
        pool = self.version_pool
        if pool is None:
            return logical_needed + reserved_size + watermark_size <= logical_available
        if logical_needed + reserved_size > logical_available:
            return False
        backup = req.kv.retraction_backup
        assert backup is not None
        slots = pool.restore_slots(backup.cpu_tensors, backup.cached_len)
        physical_needed = slots + pool.slots_per_token * (1 + reserved_size) + watermark_size
        physical_available = pool.available_slots + pool.slots_per_token * evictable
        return physical_needed <= physical_available

    def check_restore_capacity(self, req: Req, reserved_size: int, watermark: float) -> None:
        """Reject an unrestorable snapshot when no running request can free space."""
        if self.restore_fits(req, reserved_size, watermark):
            return
        watermark_size = self._watermark_size(watermark)
        pool = self.version_pool
        if pool is None:
            needed_pages = div_ceil(req.device_len + reserved_size + watermark_size, self.page_size)
            self.check_capacity(needed_pages)
            return
        backup = req.kv.retraction_backup
        assert backup is not None
        slots = pool.restore_slots(backup.cpu_tensors, backup.cached_len)
        evictable = self.prefix_cache.size_info.evictable_size
        physical_needed = slots + pool.slots_per_token * (1 + reserved_size) + watermark_size
        physical_available = pool.available_slots + pool.slots_per_token * evictable
        logical_needed = req.device_len + reserved_size
        logical_available = len(self.free_slots) + evictable
        raise RuntimeError(
            f"KV cache exhausted restoring request {req.uid}: "
            f"{physical_needed} KV slots needed, {physical_available} available; "
            f"{logical_needed} logical handles needed, {logical_available} available."
        )

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor, *, reclaimed_versions: int = 0) -> None:
            lazy_free_list.append((indices[:: self.page_size], reclaimed_versions))

        lazy_free_list: List[Tuple[torch.Tensor, int]] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            if lazy_free_list:
                if self.version_pool is not None:
                    indices = torch.cat([item[0] for item in lazy_free_list])
                    pool = self.version_pool
                    reclaimed = sum(item[1] for item in lazy_free_list)
                    released = len(indices) * pool.slots_per_token - pool.core_groups * reclaimed
                    self.version_pool.free_versions(indices, released)
                self.free_slots = torch.cat(
                    [self.free_slots] + [item[0] for item in lazy_free_list]
                )

    def _allocate(
        self, needed_pages: int, *, num_running_reqs: int = 0, physical_slots: int | None = None
    ) -> torch.Tensor:
        pool = self.version_pool
        if pool is None:
            self.check_capacity(needed_pages, num_running_reqs=num_running_reqs)
        evictable = self.prefix_cache.size_info.evictable_size // self.page_size
        if needed_pages > len(self.free_slots) + evictable:
            raise RuntimeError("KV cache has insufficient logical token handles.")
        needed_slots = (
            needed_pages * pool.slots_per_token
            if pool is not None and physical_slots is None
            else physical_slots
        )
        deficit = max(0, needed_pages - len(self.free_slots))
        if pool is not None:
            assert needed_slots is not None
            deficit = max(
                deficit, div_ceil(max(0, needed_slots - pool.available_slots), pool.slots_per_token)
            )
        if deficit > evictable:
            self.check_capacity(needed_pages, num_running_reqs=num_running_reqs)
            raise RuntimeError("KV cache has insufficient physical KV slots.")
        if deficit:
            evicted = self.prefix_cache.evict(deficit * self.page_size)
            # Eviction must make capacity available to this allocation immediately.
            CacheManager._free(self, evicted)
        if len(self.free_slots) < needed_pages:
            raise RuntimeError("KV cache eviction did not free enough logical handles.")
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor, *, reclaimed_versions: int = 0) -> None:
        if len(indices) > 0:
            pool = self.version_pool
            if pool is not None:
                pool.free_versions(
                    indices,
                    len(indices) * pool.slots_per_token - pool.core_groups * reclaimed_versions,
                )
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    assert sum((end - start) * page_size for _, start, end in allocation_info) == len(allocated)
    metadata = make_device_tensor(
        [
            [row for row, _, _ in allocation_info],
            [start * page_size for _, start, _ in allocation_info],
            [end * page_size for _, _, end in allocation_info],
        ],
        torch.int32,
        page_table.device,
    )
    assign_req_to_token_pool_func(
        metadata[0], page_table, metadata[1], metadata[2], allocated, len(allocation_info)
    )
