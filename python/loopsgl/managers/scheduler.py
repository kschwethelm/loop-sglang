from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import torch

from loopsgl.env import ENV
from loopsgl.managers.base_scheduler import BaseScheduler, ForwardInput
from loopsgl.managers.scheduler_components.batch_result_processor import PendingBatchResult
from loopsgl.managers.scheduler_components.config import SchedulerConfig
from loopsgl.managers.scheduler_components.decode import DecodeManager
from loopsgl.utils import init_logger

if TYPE_CHECKING:
    import multiprocessing as mp

    from loopsgl.model_executor import ForwardOutput, ModelRunner

logger = init_logger(__name__)


class Scheduler(BaseScheduler):
    batch_phases = ("prefill", "decode")

    def _create_model_runner(self, config: SchedulerConfig) -> ModelRunner:
        from loopsgl.model_executor import ModelRunner

        return ModelRunner(config)

    def _create_decode_manager(self) -> DecodeManager:
        return DecodeManager()

    def _event_loop_overlap_iteration(self) -> None:
        """Overlap the current forward with processing of the previous token result."""
        self.ingest_requests()
        last_result = self.pending_result
        if self.retraction_mode != "none" and not self.cache_manager.check_decode_mem(
            self.decode_manager.running_reqs
        ):
            # Pressure recovery must see every generated token before releasing or rebuilding KV.
            self.model_runner.stream.synchronize()
            self.process_batch_result(last_result)
            last_result = None
            self.pending_result = None

        forward_input = self.get_next_batch_to_run()
        self.pending_result = None
        if forward_input is not None:
            with self.model_runner_stream_ctx:
                self.model_runner.stream.wait_stream(self.stream)
                self.pending_result = PendingBatchResult(
                    forward_input=forward_input, output=self.run_batch(forward_input)
                )
        self.process_batch_result(last_result)

    def _event_loop_normal_iteration(self) -> None:
        self.ingest_requests()
        forward_input = self.get_next_batch_to_run()
        if forward_input is not None:
            self.pending_result = PendingBatchResult(
                forward_input=forward_input, output=self.run_batch(forward_input)
            )
            self.process_batch_result(self.pending_result)
            self.pending_result = None

    @torch.inference_mode()
    def event_loop(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.model_runner_stream_ctx:
                self.model_runner.stream.wait_stream(self.stream)
                while True:
                    self._event_loop_normal_iteration()
        else:
            assert torch.cuda.current_stream() == self.stream
            while True:
                self._event_loop_overlap_iteration()

    def get_next_batch_to_run(self) -> ForwardInput | None:
        retracted = []
        if self.retraction_mode != "none":
            old_available = self.cache_manager.available_size
            retracted = self.decode_manager.retract_decode(
                self.cache_manager,
                self.table_manager,
                retraction_backup=(
                    self.retraction_manager.retraction_backup
                    if self.retraction_mode == "offload"
                    else None
                ),
            )
            if self.retraction_mode == "recompute":
                for req in retracted:
                    self.prefill_manager.add_retracted_req(req)
            self.num_retracted_reqs += len(retracted)
            if retracted:
                logger.warning_rank0(
                    "KV cache pool is full. Retract requests. "
                    f"#retracted_reqs: {len(retracted)}, "
                    f"#new_tokens_gained: {self.cache_manager.available_size - old_available}"
                )

        chunked_prefill = any(req.kv.holds_kv for req in self.prefill_manager.pending_list)
        if not retracted and not chunked_prefill and self.retraction_manager.runnable:
            # CPU restores write KV and token rows that an outstanding forward may still use.
            self.stream.wait_stream(self.model_runner.stream)
            self.retraction_manager.resume_retracted_reqs()

        # Protect actual next-step allocations while admitting prefills.
        reserved_size = (
            self.cache_manager.new_pages_required(self.decode_manager.running_reqs)
            * self.cache_manager.page_size
            if self.retraction_mode != "none"
            else 0
        )
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(
                self.prefill_budget,
                reserved_size=reserved_size,
                allow_new_requests=not self.retraction_manager.runnable,
            )
            if not retracted
            else None
        ) or self.decode_manager.schedule_next_batch(self.model_runner.sampler)
        if batch is None:
            return None
        forward_input = self._prepare_batch(batch)
        if not batch.is_prefill:
            batch.prepare_for_decode()
        return forward_input

    def run_batch(self, forward_input: ForwardInput) -> ForwardOutput:
        batch = forward_input.batch
        # Scheduler tensors are read asynchronously on the model runner stream.
        for tensor in (
            batch.req_pool_indices,
            batch.seq_lens,
            batch.extend_lens,
            batch.positions,
            batch.out_loc,
            *forward_input.input_tuple,
            *forward_input.write_tuple,
            forward_input.sample_args.temperatures,
            forward_input.sample_args.top_k,
            forward_input.sample_args.top_p,
        ):
            if tensor is not None:
                tensor.record_stream(self.model_runner.stream)
        batch.input_ids = self.token_pool[forward_input.input_tuple]
        recomputed_tokens = (
            sum(req.extend_len for req in forward_input.reqs if req.is_retracted)
            if batch.is_prefill
            else 0
        )
        forward_output = self.model_runner.forward_batch(batch, forward_input.sample_args)
        if batch.is_prefill:
            self.batch_size_stats["prefill"].record(
                batch, padded_tokens=self.model_runner.prefill_graph_runner.bucket_for(batch)
            )
        else:
            self.batch_size_stats["decode"].record(batch)
        self.metadata_ready = batch.metadata_ready
        for req, is_middle_chunk in zip(forward_input.reqs, forward_input.is_middle_chunks):
            req.complete_one(is_middle_chunk=is_middle_chunk)
        self.num_recomputed_tokens += recomputed_tokens
        if batch.is_prefill:
            for req, is_middle_chunk in zip(forward_input.reqs, forward_input.is_middle_chunks):
                if not is_middle_chunk:
                    req.is_retracted = False
        self.token_pool[forward_input.write_tuple] = forward_output.next_tokens_gpu
        # Completion must also fence token-row writes before the row can be reused.
        forward_output.copy_done_event.record(self.model_runner.stream)
        self.decode_manager.filter_reqs(forward_input.reqs)
        return forward_output


def create_scheduler(config: SchedulerConfig) -> BaseScheduler:
    """Construct the configured CB or continuous-depth batching scheduler."""
    if config.scheduler == "cb":
        return Scheduler(config)
    if config.scheduler == "cdb":
        from loopsgl.managers.cdb_scheduler import CDBScheduler

        return CDBScheduler(config)
    raise ValueError(f"Unknown scheduler: {config.scheduler}")


def run_scheduler_process(
    config: SchedulerConfig, ack_queue: mp.Queue[dict], *, silent_output: bool = False
) -> None:
    """Initialize one TP worker and serve requests until the owning runtime exits."""
    import logging
    import traceback

    ready_sent = False
    try:
        with torch.inference_mode():
            scheduler = create_scheduler(config)
            scheduler.sync_all_ranks()
            ack_queue.put(scheduler.get_server_info())
            ready_sent = True
            if silent_output:
                logging.disable(logging.INFO)
            try:
                scheduler.event_loop()
            except KeyboardInterrupt:
                scheduler.shutdown()
    except Exception:
        if not ready_sent:
            ack_queue.put({"error": traceback.format_exc()})
        raise
