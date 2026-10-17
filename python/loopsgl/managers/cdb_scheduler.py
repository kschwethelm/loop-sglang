from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable, Literal, NamedTuple, NoReturn, Sequence

import torch

from loopsgl.core import Req
from loopsgl.env import ENV
from loopsgl.managers.base_scheduler import BaseScheduler, ForwardInput
from loopsgl.managers.io_struct import AbortReq, BaseReq, TokenizedGenerateReqInput
from loopsgl.managers.schedule_batch import ScheduleBatch
from loopsgl.managers.scheduler_components.batch_result_processor import PendingBatchResult
from loopsgl.managers.scheduler_components.config import SchedulerConfig
from loopsgl.managers.scheduler_components.decode import DecodeManager
from loopsgl.model_executor.sample import make_device_tensor
from loopsgl.utils import init_logger, profile_range

if TYPE_CHECKING:
    from loopsgl.model_executor import ForwardOutput
    from loopsgl.model_executor.cdb_model_runner import CDBModelRunner, GateOutput


logger = init_logger(__name__)


class CDBDecodeManager(DecodeManager):
    """Track resident decode requests across CDB stages until completion."""

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {
            req
            for req in self.running_reqs.union(reqs)
            if not req.finished() and req.kv.holds_kv and req.device_len > req.input_len
        }


@dataclass(frozen=True)
class CDBDecodeToken:
    """Immutable identity for one decode position across all model stages."""

    req: Req
    req_pool_idx: int
    position: int
    depth: int = 0

    def advance(self) -> CDBDecodeToken:
        return replace(self, depth=self.depth + 1)

    @property
    def identity(self) -> tuple[Req, int, int]:
        return self.req, self.req_pool_idx, self.position

    @property
    def is_live(self) -> bool:
        return (
            not self.req.finished()
            and self.req.kv.req_pool_idx == self.req_pool_idx
            and self.req.device_len == self.position + 1
        )


class CDBBatchPlan(NamedTuple):
    phase: Literal["prefill", "core", "coda"]
    batch: ScheduleBatch | None = None
    tokens: tuple[CDBDecodeToken, ...] = ()


class CDBScheduler(BaseScheduler):
    """Schedule staged decoding with optional one-step exit lookahead."""

    batch_phases = ("prefill", "prelude", "recurrent", "coda")

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__(config)
        self.core_ready_tokens: list[CDBDecodeToken] = []
        self.coda_ready_tokens: list[CDBDecodeToken] = []
        self.next_cohort_tokens: list[CDBDecodeToken] = []
        self.pending_gate: tuple[list[CDBDecodeToken], GateOutput | None] | None = None
        self._stage_cache: (
            tuple[tuple[tuple[Req, int, int], ...], ScheduleBatch, ForwardInput] | None
        ) = None
        self.row_write_ready: torch.cuda.Event | None = None
        self.cohort_frozen = False
        self.draining_for_memory = False
        self.recovery_deferred = False
        self.deferred_recompute: list[Req] = []
        self.max_depth = self.model_runner.model.model.loop_steps
        self._recompute_warning_logged = False
        if self.model_runner.model.model.loop_params.exit_threshold is not None:
            self._warn_recompute()
        self.refill = config.cdb_config.refill
        self.disable_lookahead_exit = config.cdb_config.disable_lookahead_exit
        self.min_coda_batch_size = (
            min(config.cdb_config.min_coda_batch_size, config.max_running_req)
            if self.refill
            else config.max_running_req
        )

    def _warn_recompute(self) -> None:
        if (
            self.retraction_mode == "recompute"
            and self.max_depth > 1
            and not self._recompute_warning_logged
        ):
            logger.warning_rank0(
                "CDB recomputation rebuilds KV at the configured full loop depth without "
                "replaying prior early-exit depths. Rebuilt KV and subsequent outputs may "
                "differ from uninterrupted decoding. Use --retraction-mode offload to "
                "preserve cached KV."
            )
            self._recompute_warning_logged = True

    def _create_model_runner(self, config: SchedulerConfig) -> CDBModelRunner:
        from loopsgl.model_executor.cdb_model_runner import CDBModelRunner

        return CDBModelRunner(config)

    def _create_decode_manager(self) -> CDBDecodeManager:
        return CDBDecodeManager()

    def _prefix_insert_limit(self, req: Req) -> int:
        return req.origin_input_len

    def _stage_input(
        self, tokens: Sequence[CDBDecodeToken], *, prepare_sampling: bool = False
    ) -> ForwardInput:
        identities = tuple(token.identity for token in tokens)
        if self._stage_cache is None or self._stage_cache[0] != identities:
            batch = ScheduleBatch([token.req for token in tokens], "decode")
            data = self._prepare_batch(batch, prepare_sampling=False)
            self._stage_cache = identities, batch, data
        _, batch, data = self._stage_cache
        if prepare_sampling:
            data = data._replace(sample_args=self.model_runner.sampler.prepare(batch))
        return data

    def _get_state_slots(self, data: ForwardInput) -> torch.Tensor:
        # Stage snapshots may outlive metadata processing on the scheduler stream.
        batch = data.batch
        for tensor in (
            batch.req_pool_indices,
            batch.seq_lens,
            batch.extend_lens,
            batch.positions,
            batch.out_loc,
            *data.input_tuple,
            *data.write_tuple,
            data.sample_args.temperatures,
            data.sample_args.top_k,
            data.sample_args.top_p,
        ):
            if tensor is not None:
                tensor.record_stream(self.model_runner.stream)
        return batch.req_pool_indices[: batch.size].to(torch.int64)

    def _prelude_successors(self, reqs: tuple[Req, ...]) -> None:
        # New tokens can follow a restore or a remapped prefix at the same position.
        self._stage_cache = None
        self.decode_manager.filter_reqs(reqs)
        if self.draining_for_memory:
            return
        # Reserve successors together; pressure pauses refill until active tokens finish.
        successors = [req for req in self.decode_manager.running_reqs if req.can_decode]
        if not self.cache_manager.check_decode_mem(successors):
            self.draining_for_memory = True
            return
        tokens = [
            CDBDecodeToken(req, req.kv.req_pool_idx, req.device_len - 1)
            for req in reqs
            if req.can_decode
        ]
        if tokens:
            data = self._stage_input(tokens)
            self.model_runner.stream.wait_stream(self.stream)
            with profile_range("cdb.compute.prelude"), self.model_runner_stream_ctx:
                data.batch.input_ids = self.token_pool[data.input_tuple]
                self.model_runner.forward_prelude(data.batch, self._get_state_slots(data))
                self.batch_size_stats["prelude"].record(data.batch)
                self.metadata_ready = data.batch.metadata_ready
            self._enqueue_new(tokens)

    def _enqueue_new(self, tokens: list[CDBDecodeToken]) -> None:
        if self.refill or not self.cohort_frozen:
            self.core_ready_tokens.extend(tokens)
        else:
            self.next_cohort_tokens.extend(tokens)

    def _consume_gate(self, pending: tuple[list[CDBDecodeToken], GateOutput | None] | None) -> None:
        if pending is None:
            return
        tokens, output = pending
        current_tokens = {token.identity: token for token in self.core_ready_tokens}
        if not any(
            token.identity in current_tokens and current_tokens[token.identity].is_live
            for token in tokens
        ):
            if output is not None:
                output.release()
            return
        decisions = [False] * len(tokens)
        if output is not None:
            with profile_range("cdb.gate_wait"):
                output.copy_done_event.synchronize()
            decisions = [bool(value) for value in output.gates_cpu.tolist()]
            output.release()
        for index, token in enumerate(tokens):
            depths = token.req.sampling_params.replay_exit_depths
            if depths is not None:
                # Replay depths describe final states, independent of gate timing.
                output_index = token.position - token.req.origin_input_len
                delay = 0 if self.disable_lookahead_exit else 1
                decisions[index] = token.depth >= depths[output_index] - delay
        exited: list[CDBDecodeToken] = []
        for token, exit_requested in zip(tokens, decisions):
            # A delayed result can refer to a token that already reached coda or was cancelled.
            current = current_tokens.get(token.identity)
            if current is None or not current.is_live:
                continue
            exit_ready = self.disable_lookahead_exit or current.depth > token.depth
            if current.depth >= self.max_depth or (exit_requested and exit_ready):
                exited.append(current)
        if exited:
            exited_ids = {token.identity for token in exited}
            self.core_ready_tokens = [
                token for token in self.core_ready_tokens if token.identity not in exited_ids
            ]
            self.coda_ready_tokens.extend(exited)

    def _run_core(self, tokens: Sequence[CDBDecodeToken]) -> None:
        data = self._stage_input(tokens)
        depths = make_device_tensor([token.depth for token in tokens], torch.int32, self.device)
        depths.record_stream(self.model_runner.stream)
        self.model_runner.stream.wait_stream(self.stream)
        with profile_range("cdb.compute.core"), self.model_runner_stream_ctx:
            gates = self.model_runner.forward_recurrent(
                data.batch, self._get_state_slots(data), depths
            )
            self.batch_size_stats["recurrent"].record(data.batch)
            if self.model_runner.model.model.loop_params.exit_threshold is not None:
                assert gates is not None
                output = self.model_runner.stage_gate_readback(gates)
            else:
                output = None
            self.metadata_ready = data.batch.metadata_ready
        advanced = {token: token.advance() for token in tokens}
        self.core_ready_tokens = [advanced.get(token, token) for token in self.core_ready_tokens]
        replaying = any(
            token.req.sampling_params.replay_exit_depths is not None for token in tokens
        )
        current_gate = (
            (list(advanced.values()), output) if output is not None or replaying else None
        )
        if self.disable_lookahead_exit:
            # Resolve this step's routing before another core can overwrite its state.
            self._consume_gate(current_gate)
        else:
            previous = self.pending_gate
            self.pending_gate = current_gate
            self._consume_gate(previous)
        # Full depth is a hard cap independent of the asynchronously consumed exit signal.
        completed = [token for token in self.core_ready_tokens if token.depth >= self.max_depth]
        self.core_ready_tokens = [
            token for token in self.core_ready_tokens if token.depth < self.max_depth
        ]
        self.coda_ready_tokens.extend(completed)
        self.cohort_frozen = not self.refill

    def _run_coda(self, tokens: Sequence[CDBDecodeToken]) -> None:
        if self.cache_manager.version_pool is not None:
            # Finalize only after the last core write and before successor allocation.
            self.stream.wait_stream(self.model_runner.stream)
            self.cache_manager.finalize_last_exited(
                [token.req for token in tokens], [token.depth for token in tokens]
            )
        data = self._stage_input(tokens, prepare_sampling=True)
        for token in tokens:
            self.coda_ready_tokens.remove(token)
        self.model_runner.stream.wait_stream(self.stream)
        with profile_range("cdb.compute.coda"), self.model_runner_stream_ctx:
            gpu_tokens = self.model_runner.forward_coda(
                data.batch, self._get_state_slots(data), data.sample_args
            )
            self.batch_size_stats["coda"].record(data.batch)
            self.metadata_ready = data.batch.metadata_ready
            for req in data.reqs:
                req.complete_one(is_middle_chunk=False)
            self.token_pool[data.write_tuple] = gpu_tokens
            output = self.model_runner.token_readback(gpu_tokens)
            self._record_row_write()
        self._prelude_successors(data.reqs)
        self._queue_output(data, output, loop_steps=tuple(token.depth for token in tokens))
        self.recovery_deferred = False
        for req in self.deferred_recompute:
            self.prefill_manager.add_retracted_req(req)
        self.deferred_recompute.clear()

    def _run_prefill(self, batch: ScheduleBatch) -> None:
        self._stage_cache = None
        data = self._prepare_batch(batch)
        self.model_runner.stream.wait_stream(self.stream)
        with self.model_runner_stream_ctx:
            output = self._forward_prefill(data)
        completed = tuple(
            req for req, middle in zip(data.reqs, data.is_middle_chunks) if not middle
        )
        if completed:
            # Canonicalize prompt KV before any decode phase can observe its private tail.
            # Prefix reuse must remain fixed throughout the successor token's recurrence.
            self.stream.wait_stream(self.model_runner.stream)
            with self.cache_manager.lazy_free_region():
                for req in completed:
                    self.cache_manager.cache_req(
                        req, finished=False, insert_limit=self._prefix_insert_limit(req)
                    )
        self._prelude_successors(completed)
        self._queue_output(data, output)

    def _discard_dead_tokens(self) -> None:
        self.core_ready_tokens = [token for token in self.core_ready_tokens if token.is_live]
        self.coda_ready_tokens = [token for token in self.coda_ready_tokens if token.is_live]
        self.next_cohort_tokens = [token for token in self.next_cohort_tokens if token.is_live]

    def _process_one_msg(self, msg: BaseReq) -> None:
        if (
            isinstance(msg, TokenizedGenerateReqInput)
            and msg.sampling_params.replay_exit_depths is not None
        ):
            if any(depth < self.max_depth for depth in msg.sampling_params.replay_exit_depths):
                self._warn_recompute()
        if isinstance(msg, AbortReq):
            self._stage_cache = None
            for req in list(self.deferred_recompute):
                if req.uid == msg.uid:
                    req.finished_reason = "abort"
                    self.deferred_recompute.remove(req)
                    return
            # Every stage can retain a request row even when no sampled token is pending.
            self.stream.wait_stream(self.model_runner.stream)
        super()._process_one_msg(msg)
        self._discard_dead_tokens()
        if isinstance(msg, AbortReq) and not (
            self.core_ready_tokens or self.coda_ready_tokens or self.next_cohort_tokens
        ):
            # Cancellation can remove the request whose progress deferred victim readmission.
            self.recovery_deferred = False
            for req in self.deferred_recompute:
                self.prefill_manager.add_retracted_req(req)
            self.deferred_recompute.clear()

    def _forward_prefill(self, data: ForwardInput) -> ForwardOutput:
        # Shared lifecycle advances only token-producing prefill and coda forwards.
        batch = data.batch
        self._get_state_slots(data)
        batch.input_ids = self.token_pool[data.input_tuple]
        output = self.model_runner.forward_prefill(batch, data.sample_args)
        self.batch_size_stats["prefill"].record(
            batch, padded_tokens=self.model_runner.prefill_graph_runner.bucket_for(batch)
        )
        self.metadata_ready = batch.metadata_ready
        for req, middle in zip(data.reqs, data.is_middle_chunks):
            if req.is_retracted:
                self.num_recomputed_tokens += req.extend_len
            req.complete_one(is_middle_chunk=middle)
            if not middle:
                req.is_retracted = False
        self.token_pool[data.write_tuple] = output.next_tokens_gpu
        self._record_row_write()
        self.decode_manager.filter_reqs(data.reqs)
        return output

    def _drained_boundary(self) -> bool:
        return not (
            self.core_ready_tokens
            or self.coda_ready_tokens
            or self.next_cohort_tokens
            or self.pending_gate
            or self.pending_result
        )

    def _recover_boundary(self) -> None:
        if not self._drained_boundary():
            return
        self._stage_cache = None
        self.model_runner.stream.synchronize()
        reqs = sorted(self.decode_manager.running_reqs, key=lambda req: req.uid)
        if self.retraction_mode == "none":
            self.cache_manager.allocate_paged(
                reqs, num_running_reqs=self.table_manager.allocated_size
            )
        elif reqs and not self.cache_manager.check_decode_mem(reqs):
            victims = self.decode_manager.retract_decode(
                self.cache_manager,
                self.table_manager,
                retraction_backup=(
                    self.retraction_manager.retraction_backup
                    if self.retraction_mode == "offload"
                    else None
                ),
            )
            self.num_retracted_reqs += len(victims)
            self.recovery_deferred = bool(victims)
            if self.retraction_mode == "recompute":
                self.deferred_recompute.extend(victims)
        self.draining_for_memory = False
        represented = {
            token.req
            for token in self.core_ready_tokens + self.coda_ready_tokens + self.next_cohort_tokens
        }
        waiting = tuple(
            req
            for req in sorted(self.decode_manager.running_reqs, key=lambda req: req.uid)
            if req.can_decode and req not in represented
        )
        self._prelude_successors(waiting)

    def get_next_batch_to_run(self) -> CDBBatchPlan | None:
        self._discard_dead_tokens()
        if not self.core_ready_tokens and self.pending_gate is not None:
            pending, self.pending_gate = self.pending_gate, None
            self._consume_gate(pending)
        if not self.core_ready_tokens and not self.coda_ready_tokens and self.pending_gate is None:
            self.cohort_frozen = False
            self.core_ready_tokens.extend(self.next_cohort_tokens)
            self.next_cohort_tokens.clear()
            if self.pending_result is None:
                self._recover_boundary()
        if len(self.coda_ready_tokens) >= self.min_coda_batch_size:
            return CDBBatchPlan("coda", tokens=tuple(self.coda_ready_tokens))
        if not self.cohort_frozen and not self.draining_for_memory:
            owned_prompt = any(req.kv.holds_kv for req in self.prefill_manager.pending_list)
            if not self.recovery_deferred and not owned_prompt and self.retraction_manager.runnable:
                # Restore before fresh admission without draining other requests' token stages.
                self.stream.wait_stream(self.model_runner.stream)
                resumed = self.retraction_manager.resume_retracted_reqs()
                if resumed:
                    self._prelude_successors(tuple(resumed))
            reserved = (
                self.cache_manager.new_pages_required(self.decode_manager.running_reqs)
                * self.cache_manager.page_size
                if self.retraction_mode != "none"
                else 0
            )
            batch = self.prefill_manager.schedule_next_batch(
                self.prefill_budget,
                reserved_size=reserved,
                allow_new_requests=not (self.retraction_manager.runnable or self.recovery_deferred),
            )
            if batch is not None:
                return CDBBatchPlan("prefill", batch=batch)
        if self.core_ready_tokens:
            return CDBBatchPlan("core", tokens=tuple(self.core_ready_tokens))
        if self.coda_ready_tokens:
            return CDBBatchPlan("coda", tokens=tuple(self.coda_ready_tokens))
        return None

    def run_batch(self, plan: CDBBatchPlan) -> None:
        if plan.phase == "prefill":
            assert plan.batch is not None
            self._run_prefill(plan.batch)
        elif plan.phase == "core":
            self._run_core(plan.tokens)
        else:
            self._run_coda(plan.tokens)

    def _event_loop_iteration(self) -> None:
        self.ingest_requests()
        if self.draining_for_memory:
            # EOS and stop results can release enough KV to avoid retraction at the boundary.
            pending, self.pending_result = self.pending_result, None
            self.process_batch_result(pending)
        plan = self.get_next_batch_to_run()
        previous, self.pending_result = self.pending_result, None
        if plan is not None:
            self.run_batch(plan)
        self.process_batch_result(previous)
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            pending, self.pending_result = self.pending_result, None
            self.model_runner.stream.synchronize()
            self.process_batch_result(pending)

    @torch.inference_mode()
    def event_loop(self) -> NoReturn:
        while True:
            self._event_loop_iteration()

    def shutdown(self) -> None:
        output, self.pending_result = self.pending_result, None
        self.process_batch_result(output)
        if self.pending_gate is not None:
            pending, self.pending_gate = self.pending_gate, None
            self._consume_gate(pending)
        super().shutdown()

    def _record_row_write(self) -> None:
        self.row_write_ready = torch.cuda.Event()
        self.row_write_ready.record(self.model_runner.stream)

    def _free_req_resources(self, req: Req) -> None:
        self._stage_cache = None
        self.stream.wait_stream(self.model_runner.stream)
        super()._free_req_resources(req)

    def is_fully_idle(self) -> bool:
        return super().is_fully_idle() and self._drained_boundary() and not self.deferred_recompute

    def reset_generation(self) -> None:
        super().reset_generation()
        self._stage_cache = None
        self.cohort_frozen = False
        self.draining_for_memory = False
        self.recovery_deferred = False

    def _queue_output(
        self,
        data: ForwardInput,
        output: ForwardOutput,
        *,
        loop_steps: tuple[int, ...] | None = None,
    ) -> None:
        if self.pending_result is not None:
            raise RuntimeError("CDB token result pipeline exceeded its one-result bound.")
        self.pending_result = PendingBatchResult(
            forward_input=data,
            output=output,
            row_write_ready=self.row_write_ready,
            loop_steps=loop_steps,
        )

    def process_batch_result(self, result: PendingBatchResult | None) -> None:
        if result is None:
            return
        if result.row_write_ready is not None:
            self.stream.wait_event(result.row_write_ready)
        if self.metadata_ready is not None:
            self.stream.wait_event(self.metadata_ready)
        super().process_batch_result(result)
        self._discard_dead_tokens()
