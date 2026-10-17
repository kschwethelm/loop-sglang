from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch

from loopsgl.core import ForwardBatch
from loopsgl.models.looped.base import BaseLoopedForCausalLM
from loopsgl.utils import profile_range

from .config import EngineConfig
from .graph import _determine_cuda_graph_bs
from .model_runner import ForwardOutput, ModelRunner
from .sample import BatchSamplingArgs


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
            logits = self._forward_logits(batch)
            return self.token_readback(self._sample_logits(logits[: batch.size], args))

    def _execute_phase(
        self, phase: Literal["prelude", "core", "coda"], slots: torch.Tensor, depths: torch.Tensor
    ) -> torch.Tensor | None:
        assert isinstance(self.model, BaseLoopedForCausalLM)
        model = self.model.model
        if phase == "prelude":
            hidden, injection = model.forward_prelude(self.ctx.batch.input_ids)
            self.state_pool["hidden"].index_copy_(0, slots, hidden)
            if injection is not None:
                self.state_pool["injection"].index_copy_(0, slots, injection)
            if model.loop_params.exit_threshold is not None:
                self.state_pool["exit"].index_copy_(0, slots, model.init_exit_state(hidden))
            return None
        hidden = self.state_pool["hidden"].index_select(0, slots)
        if phase == "coda":
            return self.model.forward_coda(hidden)
        injection_pool = self.state_pool.get("injection")
        injection = None if injection_pool is None else injection_pool.index_select(0, slots)
        # Read the old gate state before updating hidden, including aliasing Huginn states.
        exit_pool = self.state_pool.get("exit")
        exit_state = None if exit_pool is None else exit_pool.index_select(0, slots)
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
        phase: Literal["prelude", "core", "coda"],
        batch: ForwardBatch,
        slots: torch.Tensor,
        depths: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        assert torch.cuda.current_stream() == self.stream
        if not batch.is_decode:
            raise ValueError("CDB phases require decode attention metadata")
        slots = slots.to(device=self.device, dtype=torch.int64)
        if slots.numel() != batch.size:
            raise ValueError("State slots must match the real phase batch size")
        if depths is None:
            depths = torch.zeros(batch.size, dtype=torch.int32, device=self.device)
        else:
            depths = depths.to(device=self.device, dtype=torch.int32)
        if depths.numel() != batch.size:
            raise ValueError("Depths must match the real phase batch size")
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                return self.graph_runner.replay_phase(phase, batch, slots, depths)
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

    def token_readback(self, tokens: torch.Tensor) -> ForwardOutput:
        host = torch.empty(tokens.shape, dtype=tokens.dtype, device="cpu", pin_memory=True)
        ready = torch.cuda.Event()
        ready.record(self.stream)
        event = torch.cuda.Event()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(ready)
            host.copy_(tokens, non_blocking=True)
            tokens.record_stream(self.copy_stream)
            event.record(self.copy_stream)
        return ForwardOutput(tokens, host, event)
