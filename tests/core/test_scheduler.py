from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import loopsgl.distributed.info as distributed_info
import pytest
import torch
from loopsgl.core import Req, SamplingParams
from loopsgl.distributed import DistributedInfo
from loopsgl.managers.cdb_scheduler import CDBDecodeManager, CDBDecodeToken, CDBScheduler
from loopsgl.managers.io_struct import AbortReq


@pytest.mark.parametrize(("refill", "direct"), [(True, False), (False, True)])
def test_cdb_scheduling_lifecycle(
    refill: bool, direct: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep exit routing, cohort admission, cancellation, and retraction at legal token boundaries."""
    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 1))
    scheduler = CDBScheduler.__new__(CDBScheduler)
    scheduler.core_ready_tokens, scheduler.coda_ready_tokens, scheduler.next_cohort_tokens = (
        [],
        [],
        [],
    )
    scheduler.pending_gate = scheduler.pending_result = None
    scheduler.refill, scheduler.disable_lookahead_exit = refill, direct
    scheduler.cohort_frozen = True
    scheduler.draining_for_memory = scheduler.recovery_deferred = False
    scheduler.deferred_recompute = []
    scheduler.max_depth, scheduler.min_coda_batch_size = 4, 1
    scheduler.metadata_ready = None
    scheduler.retraction_mode, scheduler.num_retracted_reqs = "offload", 0
    scheduler.stream = Mock()
    scheduler.model_runner = SimpleNamespace(stream=Mock())
    scheduler.cache_manager = Mock(version_pool=None)
    scheduler.prefill_manager = Mock(pending_list=[])
    scheduler.prefill_manager.schedule_next_batch.return_value = None
    scheduler.retraction_manager = Mock(runnable=False)
    scheduler.retraction_manager.abort_req.return_value = False
    scheduler.prefill_manager.abort_req.return_value = None
    scheduler.table_manager = Mock(allocated_size=2)
    scheduler.send_result = Mock()
    scheduler.decode_manager = CDBDecodeManager()

    def request(uid: int) -> Req:
        req = Req(
            uid,
            torch.tensor([1, 3], dtype=torch.int32),
            SamplingParams(max_tokens=4, replay_exit_depths=[2, 3, 4]),
        )
        req.kv.req_pool_idx, req.device_len = uid, 3
        req.output_ids = [8]
        return req

    req = request(0)
    scheduler.decode_manager.running_reqs.add(req)
    token = CDBDecodeToken(req, 0, 2)
    scheduler._enqueue_new([token])
    assert scheduler.next_cohort_tokens == ([] if refill else [token])
    assert scheduler.core_ready_tokens == ([token] if refill else [])
    scheduler.core_ready_tokens, scheduler.next_cohort_tokens = [], []

    # A depth-two replay exits at depth two under either gate-consumption timing.
    scheduler.core_ready_tokens = [CDBDecodeToken(req, 0, 2, depth=2)]
    gate_token = CDBDecodeToken(req, 0, 2, depth=2 if direct else 1)
    scheduler._consume_gate(([gate_token], None))
    assert scheduler.get_next_batch_to_run().phase == "coda"
    assert scheduler.coda_ready_tokens[0].depth == 2

    scheduler._process_one_msg(AbortReq(req.uid))
    replacement = request(0)
    scheduler.decode_manager.running_reqs.add(replacement)
    scheduler.core_ready_tokens = [CDBDecodeToken(replacement, 0, 2, depth=1)]
    scheduler._consume_gate(([gate_token], None))
    assert not scheduler.coda_ready_tokens
    assert scheduler.core_ready_tokens[0].req is replacement

    victim = request(1)
    scheduler.decode_manager.running_reqs.add(victim)
    scheduler.cache_manager.check_decode_mem.side_effect = lambda reqs: len(list(reqs)) <= 1
    scheduler._prelude_successors(())
    assert scheduler.draining_for_memory
    retract = Mock(wraps=scheduler.decode_manager.retract_decode)
    monkeypatch.setattr(scheduler.decode_manager, "retract_decode", retract)
    scheduler._recover_boundary()
    retract.assert_not_called()
    scheduler.core_ready_tokens.clear()
    scheduler.pending_gate = ([gate_token], None)
    scheduler._recover_boundary()
    retract.assert_not_called()
    scheduler.pending_gate, scheduler.pending_result = None, object()
    scheduler._recover_boundary()
    retract.assert_not_called()
    scheduler.pending_result = None
    resumed = Mock()
    monkeypatch.setattr(scheduler, "_prelude_successors", resumed)
    scheduler._recover_boundary()
    retract.assert_called_once()
    assert scheduler.num_retracted_reqs == 1
    assert victim.is_retracted and not victim.kv.holds_kv
    assert scheduler.decode_manager.running_reqs == {replacement}
    scheduler.retraction_manager.retraction_backup.assert_called_once_with(victim)
    assert not scheduler.draining_for_memory

    # Results such as EOS can resolve pressure before the drained boundary is reached.
    retract.reset_mock()
    scheduler.draining_for_memory = True
    scheduler.cache_manager.check_decode_mem.side_effect = None
    scheduler.cache_manager.check_decode_mem.return_value = True
    scheduler._recover_boundary()
    retract.assert_not_called()
    assert not scheduler.draining_for_memory
