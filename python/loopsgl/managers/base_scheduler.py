from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import TYPE_CHECKING, ClassVar, List, NamedTuple, NoReturn, Tuple, TypeAlias

import torch

from loopsgl.core import ForwardBatch, Req
from loopsgl.kernel.batch import prepare_batch_inputs
from loopsgl.layers.logits_processor import LogitsMetadata
from loopsgl.managers.io_struct import (
    AbortReq,
    BaseReq,
    BatchTokenizedGenerateReqInput,
    ExitReq,
    FlushCacheReqInput,
    FlushProfileReq,
    GetInternalStateReq,
    ResetMetricsReq,
    SchedulerControlOutput,
    SchedulerControlReq,
    TokenIDOutput,
    TokenizedGenerateReqInput,
)
from loopsgl.managers.schedule_batch import ScheduleBatch
from loopsgl.managers.scheduler_components.batch_result_processor import (
    PendingBatchResult,
    SchedulerBatchResultProcessor,
)
from loopsgl.managers.scheduler_components.cache import CacheManager
from loopsgl.managers.scheduler_components.config import SchedulerConfig
from loopsgl.managers.scheduler_components.decode import DecodeManager
from loopsgl.managers.scheduler_components.io import SchedulerIOMixin
from loopsgl.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
)
from loopsgl.managers.scheduler_components.prefill import PrefillManager
from loopsgl.managers.scheduler_components.retraction import RetractionManager
from loopsgl.managers.scheduler_components.table import TableManager
from loopsgl.model_executor.cuda_event_timer import CudaEventTimer
from loopsgl.utils import init_logger, load_eos_token_ids, load_tokenizer

if TYPE_CHECKING:
    from loopsgl.model_executor import BatchSamplingArgs, ModelRunner


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: ForwardBatch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)
    reqs: Tuple[Req, ...]
    is_middle_chunks: Tuple[bool, ...]
    prefill_stats: PrefillStats | None = None
    logits_metadata: LogitsMetadata | None = None


class BaseScheduler(SchedulerIOMixin, ABC):
    batch_phases: ClassVar[tuple[str, ...]]

    def __init__(self, config: SchedulerConfig) -> None:
        """Initialize shared execution resources, request managers, and scheduler I/O."""
        self.config = config
        self.model_runner = self._create_model_runner(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.model_runner.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.model_runner_stream_ctx = torch.cuda.stream(self.model_runner.stream)
        self.metadata_ready: torch.cuda.Event | None = None
        self.pending_result: PendingBatchResult | None = None
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.model_runner.page_table)
        self.cache_manager = CacheManager(
            self.model_runner.num_pages,
            config.page_size,
            self.model_runner.page_table,
            disable_radix_cache=config.disable_radix_cache,
            kv_cache=self.model_runner.kv_cache,
        )
        self.decode_manager = self._create_decode_manager()
        self.retraction_mode = config.retraction_mode
        self.num_retracted_reqs = 0
        self.num_recomputed_tokens = 0
        self.metrics_reporter = SchedulerMetricsReporter(self)
        self.cuda_event_timer = CudaEventTimer(
            self.model_runner.stream, enabled=config.enable_cuda_event_timing
        )
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            min_free_slots_delay=config.min_free_slots_delay,
            watermark=config.watermark,
            reserve_decode=self.retraction_mode != "none",
            chunk_budget=config.prefill_chunk_budget,
        )
        self.retraction_manager = RetractionManager(
            kv_cache=self.model_runner.kv_cache,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            decode_manager=self.decode_manager,
            max_cpu_bytes=int(config.retraction_cpu_memory_gb * 1024**3),
            watermark=config.watermark,
            min_free_slots_delay=config.min_free_slots_delay,
        )
        # some alias for easy access
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(
            config.model_path, self.tokenizer, hf_config=config.hf_config
        )
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_prefill_tokens

        self.pending_control_requests: List[SchedulerControlReq] = []

        # Initialize the I/O mixin
        super().__init__(config, self.model_runner.tp_cpu_group)
        self.batch_result_processor = SchedulerBatchResultProcessor(
            tokenizer=self.tokenizer,
            eos_token_ids=self.eos_token_ids,
            cache_manager=self.cache_manager,
            decode_manager=self.decode_manager,
            free_req_resources=self._free_req_resources,
            prefix_insert_limit=self._prefix_insert_limit,
            loop_steps=config.model_config.loop_steps,
        )

    def is_fully_idle(self) -> bool:
        """Require all resident, queued, and pending-result work to have drained."""
        return not (
            self.table_manager.allocated_size
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self.retraction_manager.runnable
            or self.pending_result is not None
        )

    def ingest_requests(self) -> None:
        """Receive and dispatch this iteration's input, blocking only when fully idle."""
        self._process_control_requests()
        blocking = self.is_fully_idle()
        if blocking:
            self.on_idle()
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)
        self._process_control_requests()

    def on_idle(self) -> None:
        """Check resource accounting before waiting for new requests."""
        if not self.is_fully_idle():
            return
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def shutdown(self) -> None:
        """Synchronize outstanding device work and release retraction and model runner resources."""
        torch.cuda.synchronize(self.device)
        self.retraction_manager.clear()
        self.sync_all_ranks()
        self.model_runner.shutdown()

    def process_batch_result(self, result: PendingBatchResult | None) -> None:
        """Wait for a token-producing forward and apply its results to the requests."""
        if result is None:
            return
        result.output.copy_done_event.synchronize()
        if result.forward_input.batch.is_prefill and self.metadata_ready is not None:
            # A lookahead forward must snapshot rows before prefix insertion remaps them.
            torch.cuda.current_stream(self.device).wait_event(self.metadata_ready)
        reply = self.batch_result_processor.process_batch_result(
            result.forward_input, result.output, result.loop_steps
        )
        self.send_result(reply)
        batch = result.forward_input.batch
        if batch.is_prefill:
            if result.forward_input.prefill_stats is not None:
                self.metrics_reporter.report_prefill_stats(
                    result.forward_input.prefill_stats,
                    can_run_cuda_graph=self.model_runner.prefill_graph_runner.bucket_for(batch)
                    is not None,
                    num_generated_tokens=len(reply),
                )
        else:
            self.metrics_reporter.report_decode_stats(
                batch.size,
                can_run_cuda_graph=self.model_runner.graph_runner.can_use_cuda_graph(batch),
                num_generated_tokens=len(reply),
                phase="coda" if self.config.scheduler == "cdb" else "decode",
            )

    def _process_one_msg(self, msg: BaseReq) -> None:
        """Validate and dispatch a request, cancellation, batch, or shutdown message."""
        if isinstance(msg, SchedulerControlReq):
            self.pending_control_requests.append(msg)
        elif isinstance(msg, BatchTokenizedGenerateReqInput):
            for msg in msg.batch:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitReq):
            raise KeyboardInterrupt
        elif isinstance(msg, TokenizedGenerateReqInput):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.model_runner.max_seq_len
            max_output_len = max_seq_len - input_len
            params = msg.sampling_params
            exact_budget = (
                msg.return_logprob
                or params.replay_exit_depths is not None
                or params.trace_decode_token_ids is not None
            )
            if exact_budget and (max_output_len <= 0 or params.max_new_tokens > max_output_len):
                raise ValueError(
                    f"Request {msg.uid}: prompt length {input_len} plus max_new_tokens "
                    f"{params.max_new_tokens} must fit within context length {max_seq_len} "
                    "for scoring, forced decoding, or exit replay."
                )
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_new_tokens > max_output_len:
                msg.sampling_params.max_new_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_new_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortReq):
            self.send_result(
                [
                    TokenIDOutput(
                        uid=msg.uid,
                        next_token=0,
                        finished=True,
                        skip_token=True,
                        finish_reason={
                            "type": "abort",
                            "message": "Aborted",
                            "status_code": None,
                            "err_type": None,
                        },
                    )
                ]
            )
            logger.debug_rank0("Aborting request %d", msg.uid)
            if self.retraction_manager.abort_req(msg.uid):
                return
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                req_to_free.finished_reason = "abort"
                if req_to_free.kv.holds_kv:
                    # Outstanding forwards can still write the pages and token row.
                    self.stream.wait_stream(self.model_runner.stream)
                    self.cache_manager.release_req(req_to_free)
                    assert req_to_free.kv.req_pool_idx is not None
                    self.table_manager.free(req_to_free.kv.req_pool_idx)
                    req_to_free.kv.reset_device_state()
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        """Cache the reusable prefix and release a completed request's device resources."""
        assert req.kv.req_pool_idx is not None
        if req.device_len > req.total_len:
            # Early stopping can leave a lookahead decode reading the row and writing KV.
            self.stream.wait_stream(self.model_runner.stream)
        self.table_manager.free(req.kv.req_pool_idx)
        self.cache_manager.cache_req(
            req, finished=True, insert_limit=self._prefix_insert_limit(req)
        )
        req.kv.reset_device_state()

    def _prepare_batch(
        self, batch: ScheduleBatch, *, prepare_sampling: bool = True
    ) -> ForwardInput:
        """Allocate KV and snapshot batch metadata without advancing decode lengths."""
        from loopsgl.model_executor.sample import make_device_tensor

        self.cache_manager.allocate_paged(
            batch.reqs, num_running_reqs=self.table_manager.allocated_size
        )
        batch.prepare(self.model_runner.sampler, prepare_sampling=prepare_sampling)
        forward_batch = batch.get_forward_batch()
        self.model_runner.graph_runner.pad_batch(forward_batch)
        input_mapping = prepare_batch_inputs(forward_batch, self.model_runner.page_table)
        if batch.is_prefill:
            write_positions = make_device_tensor(
                [req.device_len if req.can_decode else -1 for req in batch.reqs],
                torch.int32,
                self.device,
            )
        else:
            write_positions = forward_batch.seq_lens[: batch.size]
        forward_input = ForwardInput(
            batch=forward_batch,
            sample_args=batch.sampling_info,
            input_tuple=input_mapping,
            write_tuple=(batch.req_pool_indices, write_positions),
            reqs=tuple(batch.reqs),
            is_middle_chunks=tuple(
                batch.is_prefill and req.device_len < req.input_len for req in batch.reqs
            ),
            prefill_stats=batch.prefill_stats,
        )
        if forward_input.sample_args.needs_logprobs_or_trace:
            return self._prepare_logprobs(forward_input)
        return forward_input

    def _prepare_logprobs(self, data: ForwardInput) -> ForwardInput:
        """Bind score and trace rows to one immutable execution position snapshot."""
        args = data.sample_args
        positions = tuple(
            (-1 if middle or position >= req.max_device_len else position)
            for position, middle, req in zip(
                data.batch.seq_lens_cpu, data.is_middle_chunks, data.reqs
            )
        )
        flags = tuple(req.return_logprob for req in data.reqs)
        lengths = tuple(req.origin_input_len for req in data.reqs) if data.batch.is_prefill else ()
        starts = (
            tuple(
                req.logprob_start_len if req.logprob_start_len >= 0 else req.origin_input_len
                for req in data.reqs
            )
            if data.batch.is_prefill
            else ()
        )
        offsets = [0]
        for enabled, length, start in zip(flags, lengths, starts):
            offsets.append(offsets[-1] + (max(length - start - 1, 0) if enabled else 0))
        if not data.batch.is_prefill:
            offsets = [0] * (len(data.reqs) + 1)
        metadata = LogitsMetadata(
            return_logprob_rows_cpu=flags,
            top_logprobs_nums=tuple(req.top_logprobs_num for req in data.reqs),
            extend_seq_lens_cpu=lengths,
            extend_logprob_start_lens_cpu=starts,
            input_logprob_offsets_cpu=tuple(offsets),
            output_positions_cpu=positions,
            prompt_lens_cpu=tuple(req.origin_input_len for req in data.reqs),
            input_ids_cpu=(
                tuple(
                    (
                        tuple(req.origin_input_ids.tolist())
                        if req.return_logprob and 0 <= req.logprob_start_len < req.origin_input_len
                        else ()
                    )
                    for req in data.reqs
                )
                if data.batch.is_prefill
                else ()
            ),
        )
        snapshot = args.snapshot(metadata, self.device)
        if snapshot.forced_token_ids is not None:
            snapshot.forced_token_ids.record_stream(self.model_runner.stream)
        return data._replace(sample_args=snapshot, logits_metadata=metadata)

    @abstractmethod
    def _create_model_runner(self, config: SchedulerConfig) -> ModelRunner:
        """Construct the model runner used by this scheduler."""

    @abstractmethod
    def _create_decode_manager(self) -> DecodeManager:
        """Construct the manager that tracks this scheduler's decode requests."""

    @abstractmethod
    def event_loop(self) -> NoReturn:
        """Run the scheduler's request and execution loop until interrupted."""

    def _prefix_insert_limit(self, req: Req) -> int | None:
        """Return the maximum token prefix eligible for caching, or None for no limit."""
        return None

    def reset_generation(self) -> None:
        """Reset generation counters and decode bookkeeping after all request work has drained."""
        self.model_runner.stream.synchronize()
        if not self.is_fully_idle():
            raise RuntimeError("Cannot reset generation while scheduler work remains.")
        self.decode_manager.running_reqs.clear()
        self.decode_manager.running_batch = None
        self.decode_manager._removed_reqs.clear()
        self.retraction_manager.reset_metrics()
        self.num_retracted_reqs = 0
        self.num_recomputed_tokens = 0
        self.metrics_reporter.reset_metrics()
        self.cuda_event_timer.reset()

    def get_server_info(self) -> dict:
        """Return model limits and scheduler counters without exposing worker GPU objects."""
        info = {
            "model_path": self.config.model_path,
            "dtype": str(self.config.dtype),
            "scheduler": self.config.scheduler,
            "max_seq_len": self.model_runner.max_seq_len,
            "num_pages": self.model_runner.num_pages,
            "tp_size": self.config.tp_info.size,
            "loop_params": asdict(self.config.model_config.loop_params),
            "num_retracted_reqs": self.num_retracted_reqs,
            "num_recomputed_tokens": self.num_recomputed_tokens,
        }
        if self.cuda_event_timer.enabled:
            info["cuda_event_timing"] = self.cuda_event_timer.snapshot(self.batch_phases)
        return info

    def _process_control_requests(self) -> None:
        """Apply control requests after all token work and pending results have drained."""
        if not self.pending_control_requests or not self.is_fully_idle():
            return
        self.model_runner.stream.synchronize()
        pending, self.pending_control_requests = self.pending_control_requests, []
        for msg in pending:
            try:
                if isinstance(msg, FlushCacheReqInput):
                    self.cache_manager.clear()
                elif isinstance(msg, ResetMetricsReq):
                    self.reset_generation()
                elif isinstance(msg, FlushProfileReq):
                    torch.cuda.cudart().cudaProfilerStop()
                    self.sync_all_ranks()
                elif not isinstance(msg, GetInternalStateReq):
                    raise ValueError(f"Unknown scheduler control request: {type(msg)}")
                self.stream.synchronize()
                reply = SchedulerControlOutput(uid=msg.uid, result=self.get_server_info())
            except Exception as exc:
                reply = SchedulerControlOutput(uid=msg.uid, result={}, error=str(exc))
            self.send_control_result(reply)
