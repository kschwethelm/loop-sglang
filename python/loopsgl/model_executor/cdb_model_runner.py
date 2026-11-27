from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch

from loopsgl.core import ForwardBatch
from loopsgl.distributed import get_tp_info
from loopsgl.kernel import indexing
from loopsgl.kernel.argmax import try_greedy_argmax_prelude
from loopsgl.layers.logits_processor import LogitsProcessorOutput
from loopsgl.models.looped.base import BaseLoopedForCausalLM
from loopsgl.utils import profile_range

from .config import EngineConfig
from .graph import _determine_cuda_graph_bs
from .model_runner import ForwardOutput, ModelRunner
from .sample import BatchSamplingArgs


def _gather_state(pool: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Gather short contiguous CUDA state rows with the existing vector-copy kernel."""
    # PyTorch 2.9 uses its small-index CUDA path for at most 16 rows.
    if (
        pool.is_cuda
        and pool.ndim == 2
        and pool.is_contiguous()
        and pool.data_ptr() % 16 == 0
        and slots.device == pool.device
        and slots.ndim == 1
        and slots.dtype in (torch.int32, torch.int64)
        and slots.is_contiguous()
        and 0 < slots.numel() <= 16
        and pool.shape[1] > 0
        and pool.shape[1] * pool.element_size() % 4 == 0
    ):
        return indexing(pool, slots)
    return pool.index_select(0, slots)


@dataclass
class GateOutput:
    gates_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event
    _release: Callable[[], None]
    _released: bool = False

    def release(self) -> None:
        """Release the snapshot; its next writer waits for the outstanding copy."""
        if not self._released:
            self._release()
            self._released = True


class CDBModelRunner(ModelRunner):
    """Execute decode phases using persistent request-row recurrent state."""

    def _init_runtime_buffers(self, config: EngineConfig, free_memory: int) -> None:
        if not isinstance(self.model, BaseLoopedForCausalLM):
            raise ValueError("Continuous-depth batching requires a looped model")
        if config.loop_cache_policy not in ("shared", "last_exited"):
            raise ValueError(
                "Continuous-depth batching requires shared or last-exited recurrent KV"
            )
        sizes = _determine_cuda_graph_bs(
            config.cuda_graph_bs, config.cuda_graph_max_bs, free_memory
        )
        self.max_state_batch = max(config.max_running_req, max(sizes, default=0))
        self.padding_state_start = config.max_running_req
        capacity = config.max_running_req + self.max_state_batch
        self.state_pool = {
            name: torch.zeros((capacity, *shape), dtype=dtype, device=self.device)
            for name, (shape, dtype) in self.model.model.cdb_state_spec().items()
            if name != "exit" or self.model.model.loop_params.exit_threshold is not None
        }
        self.padding_state_slots = torch.arange(
            self.padding_state_start, capacity, dtype=torch.int64, device=self.device
        )
        self._zero_depths = torch.zeros(self.max_state_batch, dtype=torch.int32, device=self.device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        gate_slots = 3 if self.model.model.loop_params.exit_threshold is not None else 0
        self._gate_hosts = [
            torch.empty(self.max_state_batch, dtype=torch.bool, pin_memory=True)
            for _ in range(gate_slots)
        ]
        self._gate_snapshots = [
            torch.empty(self.max_state_batch, dtype=torch.bool, device=self.device)
            for _ in self._gate_hosts
        ]
        self._gate_events = [torch.cuda.Event() for _ in self._gate_hosts]
        self._gate_leased = [False] * len(self._gate_hosts)

    def _create_graph_runner(self, **kwargs: Any) -> Any:
        from .cdb_graph import CDBGraphRunner

        return CDBGraphRunner(model_runner=self, **kwargs)

    def forward_prefill(self, batch: ForwardBatch, args: BatchSamplingArgs) -> ForwardOutput:
        if batch.is_decode:
            raise ValueError("CDB prefill requires an extend batch")
        with profile_range("cdb.compute.prefill"):
            if not args.needs_logprobs_or_trace:
                logits = self._forward_logits(batch)
                return self.token_readback(self._sample_logits(logits[: batch.size], args))
            logits, scores = self._forward_with_logprobs(batch, args)
            tokens = self._sample_with_trace(logits[: batch.size], args)
            scores = self.sampler.compute_logprobs(logits[: batch.size], tokens, args, scores)
            return self.token_readback(tokens, scores)

    def _write_prelude_state(self, input_ids: torch.Tensor, slots: torch.Tensor) -> None:
        """Initialize recurrent state for the supplied token IDs and request rows."""
        model = self.model.model
        hidden, injection = model.forward_prelude(input_ids)
        self.state_pool["hidden"].index_copy_(0, slots, hidden)
        if injection is not None:
            self.state_pool["injection"].index_copy_(0, slots, injection)
        if model.loop_params.exit_threshold is not None:
            self.state_pool["exit"].index_copy_(0, slots, model.init_exit_state(hidden))

    def _write_greedy_prelude(
        self, logits: torch.Tensor, slots: torch.Tensor, tokens: torch.Tensor
    ) -> bool:
        """Write greedy tokens and successor state, returning whether writes were fused."""
        embedding = self.model.model.get_fused_prelude_embedding()
        exit_pool = self.state_pool.get("exit")
        if (
            embedding is not None
            and "injection" not in self.state_pool
            and exit_pool is not None
            and try_greedy_argmax_prelude(
                logits, tokens, slots, embedding, self.state_pool["hidden"], exit_pool
            )
        ):
            return True
        tokens.copy_(torch.argmax(logits, dim=-1))
        self._write_prelude_state(tokens, slots)
        return False

    def _execute_phase(
        self, phase: Literal["prelude", "core", "coda"], slots: torch.Tensor, depths: torch.Tensor
    ) -> torch.Tensor | None:
        assert isinstance(self.model, BaseLoopedForCausalLM)
        model = self.model.model
        if phase == "prelude":
            self._write_prelude_state(self.ctx.batch.input_ids, slots)
            return None
        hidden = _gather_state(self.state_pool["hidden"], slots)
        if phase == "coda":
            return self.model.forward_coda(hidden)
        injection_pool = self.state_pool.get("injection")
        injection = None if injection_pool is None else _gather_state(injection_pool, slots)
        # Read the old gate state before updating hidden, including aliasing Huginn states.
        exit_pool = self.state_pool.get("exit")
        exit_state = None if exit_pool is None else _gather_state(exit_pool, slots)
        hidden = model.core_step(hidden, depths, injection)
        gates = None
        if model.loop_params.exit_threshold is not None:
            assert exit_state is not None
            gates, exit_state = model.exit_signal(hidden, exit_state, depths)
            self.state_pool["exit"].index_copy_(0, slots, exit_state)
        self.state_pool["hidden"].index_copy_(0, slots, hidden)
        return gates

    def _forward_phase(
        self,
        phase: Literal["prelude", "core", "coda", "coda_prelude"],
        batch: ForwardBatch,
        slots: torch.Tensor,
        depths: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        assert torch.cuda.current_stream() == self.stream
        if not batch.is_decode:
            raise ValueError("CDB phases require decode attention metadata")
        slots = slots.to(device=self.device)
        if slots.numel() != batch.size:
            raise ValueError("State slots must match the real phase batch size")
        if depths is None:
            depths = self._zero_depths[: batch.size]
        else:
            depths = depths.to(device=self.device, dtype=torch.int32)
        if depths.numel() != batch.size:
            raise ValueError("Depths must match the real phase batch size")
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                return self.graph_runner.replay_phase(phase, batch, slots, depths)
            if phase == "coda_prelude":
                raise ValueError("Fused coda and prelude require a captured CUDA graph")
            slots = slots.to(dtype=torch.int64)
            if self.model.model.phase_uses_attention(phase):
                self.attn_backend.prepare_metadata(batch)
            batch.record_metadata_ready()
            return self._execute_phase(phase, slots, depths)

    def forward_prelude(self, batch: ForwardBatch, slots: torch.Tensor) -> None:
        self._forward_phase("prelude", batch, slots)

    def forward_recurrent(
        self, batch: ForwardBatch, slots: torch.Tensor, depths: torch.Tensor
    ) -> torch.Tensor | None:
        gates = self._forward_phase("core", batch, slots, depths)
        return None if gates is None else gates[: batch.size]

    def forward_coda(
        self, batch: ForwardBatch, slots: torch.Tensor, args: BatchSamplingArgs
    ) -> torch.Tensor:
        logits = self._forward_phase("coda", batch, slots)
        assert logits is not None
        return self._sample_logits(logits[: batch.size], args)

    def forward_coda_with_logprobs(
        self, batch: ForwardBatch, slots: torch.Tensor, args: BatchSamplingArgs
    ) -> tuple[torch.Tensor, LogitsProcessorOutput | None]:
        logits = self._forward_phase("coda", batch, slots)
        assert logits is not None
        logits = logits[: batch.size]
        tokens = self._sample_with_trace(logits, args)
        return tokens, self.sampler.compute_logprobs(logits, tokens, args)

    def supports_fused_boundary(self) -> bool:
        """Require validated TP1 execution and deterministic attention-free boundaries."""
        model = self.model.model
        return (
            get_tp_info().size == 1
            and model.prelude_is_stateless
            and not model.phase_uses_attention("prelude")
            and not model.phase_uses_attention("coda")
        )

    def can_fuse_boundary(self, batch: ForwardBatch, args: BatchSamplingArgs) -> bool:
        """Use the captured boundary only for greedy requests and supported model phases."""
        return (
            not args.needs_logprobs_or_trace
            and args.temperatures is None
            and self.graph_runner.can_use_cuda_graph(batch)
            and self.supports_fused_boundary()
        )

    def forward_coda_prelude(self, batch: ForwardBatch, slots: torch.Tensor) -> torch.Tensor:
        """Sample greedily and initialize successor state in one captured boundary."""
        output = self._forward_phase("coda_prelude", batch, slots)
        assert output is not None
        # Readback can outlive the next replay's persistent token buffer.
        return output[: batch.size].clone()

    def stage_gate_readback(self, gates: torch.Tensor) -> GateOutput:
        try:
            index = self._gate_leased.index(False)
        except ValueError as exc:
            raise RuntimeError(
                "Consume and release a gate snapshot before launching another core"
            ) from exc
        self._gate_leased[index] = True
        size = gates.numel()
        if size > self.max_state_batch:
            self._gate_leased[index] = False
            raise ValueError("Gate batch exceeds persistent snapshot capacity")
        snapshot = self._gate_snapshots[index][:size]
        # An obsolete result can release its lease before its D2H copy finishes.
        self.stream.wait_event(self._gate_events[index])
        snapshot.copy_(gates)
        host = self._gate_hosts[index][:size]
        ready = torch.cuda.Event()
        ready.record(self.stream)
        event = self._gate_events[index]
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(ready)
            host.copy_(snapshot, non_blocking=True)
            event.record(self.copy_stream)

        def release() -> None:
            self._gate_leased[index] = False

        return GateOutput(host, event, release)

    def token_readback(
        self, tokens: torch.Tensor, scores: LogitsProcessorOutput | None = None
    ) -> ForwardOutput:
        host = torch.empty(tokens.shape, dtype=tokens.dtype, device="cpu", pin_memory=True)
        ready = torch.cuda.Event()
        ready.record(self.stream)
        event = torch.cuda.Event()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(ready)
            host.copy_(tokens, non_blocking=True)
            tokens.record_stream(self.copy_stream)
            host_scores = scores.copy_to_cpu() if scores is not None else None
            event.record(self.copy_stream)
        return ForwardOutput(tokens, host, event, host_scores)
