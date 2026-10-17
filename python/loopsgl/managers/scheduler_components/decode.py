# Portions adapted from SGLang's schedule_batch.py.
# Copyright 2023-2024 SGLang Team.
# Licensed under the Apache License, Version 2.0.
# See LICENSES/sglang.txt in the repository root.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, List, Set

from loopsgl.core import Req
from loopsgl.managers.schedule_batch import ScheduleBatch

if TYPE_CHECKING:
    from loopsgl.managers.scheduler_components.cache import CacheManager
    from loopsgl.managers.scheduler_components.table import TableManager
    from loopsgl.model_executor.sample import Sampler


@dataclass
class DecodeManager:
    running_reqs: Set[Req] = field(default_factory=set)
    running_batch: ScheduleBatch | None = field(default=None, init=False)
    _removed_reqs: Set[Req] = field(default_factory=set, init=False)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)
        if self.running_batch is not None and req in self.running_batch.reqs:
            # A restore can reuse the same Req with a different pool row before scheduling.
            self._removed_reqs.add(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.remove_req(req)
                return req
        return None

    def schedule_next_batch(self, sampler: Sampler) -> ScheduleBatch | None:
        if self.running_batch is not None:
            self.running_batch.filter_batch(
                [
                    i
                    for i, req in enumerate(self.running_batch.reqs)
                    if req in self.running_reqs and req not in self._removed_reqs
                ]
            )
        self._removed_reqs.clear()
        if not self.runnable:
            self.running_batch = None
            return None
        existing = set(self.running_batch.reqs) if self.running_batch is not None else set()
        new_reqs = sorted(self.running_reqs - existing, key=lambda req: req.uid)
        if new_reqs:
            new_batch = ScheduleBatch(new_reqs, "decode")
            new_batch.prepare(sampler)
            if self.running_batch is None or not self.running_batch.size:
                self.running_batch = new_batch
            else:
                self.running_batch.merge_batch(new_batch, sampler.vocab_size)
        return self.running_batch

    def retract_decode(
        self,
        cache_manager: CacheManager,
        table_manager: TableManager,
        *,
        retraction_backup: Callable[[Req], None] | None = None,
    ) -> List[Req]:
        """Retract short-output, long-input requests until the next decode fits."""
        if cache_manager.check_decode_mem(self.running_reqs):
            return []
        reqs = self._get_decode_retraction_order(self.running_reqs)
        retracted: List[Req] = []
        while not cache_manager.check_decode_mem(reqs):
            if len(reqs) <= 1:
                uid = reqs[0].uid if reqs else None
                raise RuntimeError(
                    f"Request {uid} cannot advance after retracting all other decode requests. "
                    f"The next step needs {cache_manager.new_pages_required(reqs)} KV pages, "
                    f"{cache_manager.available_size // cache_manager.page_size} available, "
                    f"{cache_manager.num_pages} total."
                )
            req = reqs.pop()
            if retraction_backup is not None:
                retraction_backup(req)
            self.remove_req(req)
            cache_manager.release_req(req)
            table_manager.free(req.kv.req_pool_idx)
            req.reset_for_retract()
            retracted.append(req)
        return retracted

    @staticmethod
    def _get_decode_retraction_order(reqs: Iterable[Req]) -> List[Req]:
        # SGLang's length policy pops victims from the end of this order.
        ordered = sorted(reqs, key=lambda req: req.uid)
        ordered.sort(
            key=lambda req: (req.num_generated_tokens, -req.origin_input_len), reverse=True
        )
        return ordered

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
