from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from tqdm import tqdm

from loopsgl.core import ForwardBatch, get_global_ctx
from loopsgl.distributed import get_tp_info
from loopsgl.models import BaseLLMModel
from loopsgl.utils import init_logger

from .graph import GraphCaptureBuffer, GraphRunner

if TYPE_CHECKING:
    from .cdb_model_runner import CDBModelRunner

logger = init_logger(__name__)
Phase = Literal["prelude", "core", "coda"]


class CDBGraphRunner(GraphRunner):
    """Capture phase graphs with gathered state and persistent scatter outputs."""

    def __init__(self, *, model_runner: CDBModelRunner, **kwargs: Any) -> None:
        self.model_runner = model_runner
        super().__init__(**kwargs)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel) -> None:
        self.graph_map: dict[tuple[Phase, int], torch.cuda.CUDAGraph] = {}
        if not self.max_graph_bs:
            logger.info_rank0("CDB CUDA graphs are disabled")
            return
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self.slots = self.model_runner.padding_state_slots[: self.max_graph_bs].clone()
        self.depths = torch.zeros(self.max_graph_bs, dtype=torch.int32, device=self.device)
        self.gates = (
            torch.empty(self.max_graph_bs, dtype=torch.bool, device=self.device)
            if self.model_runner.model.model.loop_params.exit_threshold is not None
            else None
        )
        torch.cuda.synchronize(self.device)
        pool = None
        for size in tqdm(
            reversed(self.graph_bs_list),
            total=len(self.graph_bs_list),
            desc="Capturing CDB decode graphs",
            unit="batch",
            disable=not get_tp_info().is_primary(),
        ):
            batch = ForwardBatch(
                size=size,
                phase="decode",
                req_pool_indices=self.padding_req_pool_indices[:size],
                seq_lens=self.padding_seq_lens[:size],
                extend_lens=self.padding_seq_lens[:size],
                seq_lens_cpu=(1,) * size,
                extend_lens_cpu=(1,) * size,
            )
            self.buffer.set_batch(batch)
            for phase in ("prelude", "core", "coda"):
                if self.model_runner.model.model.phase_uses_attention(phase):
                    self.attn_backend.prepare_for_capture(batch)
                graph = torch.cuda.CUDAGraph()
                with get_global_ctx().forward_batch(batch):
                    self._run_phase(phase, size)
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                        self._run_phase(phase, size)
                if pool is None:
                    pool = graph.pool()
                self.graph_map[phase, size] = graph
        logger.info_rank0(f"Captured CDB phase graphs for sizes: {self.graph_bs_list}")

    def _run_phase(self, phase: Phase, size: int) -> None:
        output = self.model_runner._execute_phase(phase, self.slots[:size], self.depths[:size])
        if phase == "core" and output is not None:
            assert self.gates is not None
            self.gates[:size].copy_(output)
        elif phase == "coda":
            assert output is not None
            self.buffer.logits[:size].copy_(output)

    def replay_phase(
        self,
        phase: Phase,
        batch: ForwardBatch,
        slots: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.can_use_cuda_graph(batch):
            raise ValueError("Phase batch cannot use a captured graph")
        size = batch.padded_size
        if size not in self.graph_bs_list:
            raise ValueError("Phase batch must be padded before preparing its inputs")
        # Recurrent core and coda consume persistent state instead of token IDs.
        if phase == "prelude":
            self.buffer.input_ids[:size].copy_(batch.input_ids)
        uses_attention = self.model_runner.model.model.phase_uses_attention(phase)
        if uses_attention:
            self.buffer.positions[:size].copy_(batch.positions)
            self.buffer.out_loc[:size].copy_(batch.out_loc)
        self.slots[: batch.size].copy_(slots)
        self.depths[: batch.size].copy_(depths)
        padding = size - batch.size
        if padding:
            self.slots[batch.size : size].copy_(self.model_runner.padding_state_slots[:padding])
            self.depths[batch.size : size].zero_()
        if uses_attention:
            self.attn_backend.prepare_for_replay(batch)
        batch.record_metadata_ready()
        self.graph_map[phase, size].replay()
        if phase == "core":
            return None if self.gates is None else self.gates[: batch.size]
        if phase == "coda":
            return self.buffer.logits[: batch.size]
        return None

    def replay(self, batch: ForwardBatch) -> torch.Tensor:
        raise RuntimeError("CDB decode requires an explicit model phase")
