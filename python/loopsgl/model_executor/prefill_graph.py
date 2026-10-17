# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Token-bucket prefill graphs with eager attention and logits processing.

The capture layout follows SGLang's breakable prefill runner.
Graphs are prepared with attention stubs before KV allocation, so their memory is
included in the model runner's KV budget and no capture writes reach request KV.
"""

from __future__ import annotations

import copy
import gc
from bisect import bisect_left
from functools import partial
from typing import TYPE_CHECKING, Callable

import torch
from tqdm import tqdm

from loopsgl.core import ForwardBatch, get_global_ctx
from loopsgl.distributed import get_tp_info
from loopsgl.utils import init_logger
from loopsgl.utils.breakable_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    prefill_graph_warmup,
)

if TYPE_CHECKING:
    from loopsgl.attention import BaseAttnBackend
    from loopsgl.models import BaseLLMModel

    from .config import EngineConfig

logger = init_logger(__name__)


def prefill_graph_buckets(max_num_tokens: int) -> list[int]:
    """Return SGLang's dense token schedule capped at the prefill limit."""
    if max_num_tokens < 1:
        raise ValueError("The prefill graph token limit must be positive")
    sizes = (
        list(range(4, 33, 4))
        + list(range(48, 257, 16))
        + list(range(288, 513, 32))
        + list(range(576, 1025, 64))
        + list(range(1280, 4097, 256))
        + list(range(4608, max_num_tokens + 1, 512))
    )
    return [size for size in sizes if size <= max_num_tokens]


class PrefillGraphRunner:
    """Replay the transformer body while using live request metadata in breaks."""

    def __init__(
        self,
        *,
        model: BaseLLMModel,
        config: EngineConfig,
        device: torch.device,
        stream: torch.cuda.Stream,
        tp_cpu_group: torch.distributed.ProcessGroup,
    ) -> None:
        self.model = model
        self.device = device
        self.stream = stream
        self.graphs: dict[int, BreakableCUDAGraph] = {}
        self.capture_batches: dict[int, ForwardBatch] = {}
        self.num_replays = 0
        self.buckets: list[int] = []
        if config.cuda_graph_backend_prefill == "disabled":
            return
        # Every registered model exposes the same hidden-state/logits split.
        limit = min(
            config.cuda_graph_max_bs_prefill or config.max_forward_len,
            config.max_forward_len,
        )
        self.buckets = (
            sorted(config.cuda_graph_bs_prefill)
            if config.cuda_graph_bs_prefill is not None
            else prefill_graph_buckets(limit)
        )
        if not self.buckets:
            return
        if self.buckets[-1] > limit:
            raise ValueError(
                "Prefill graph buckets exceed the configured forward or sequence limit"
            )
        max_tokens = self.buckets[-1]
        self.input_ids = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros_like(self.input_ids)
        self.output = torch.empty(
            (max_tokens, config.model_config.hidden_size), dtype=config.dtype, device=device
        )
        self.pool = torch.cuda.graph_pool_handle()
        barrier = (
            partial(torch.distributed.barrier, group=tp_cpu_group)
            if get_tp_info().size > 1
            else None
        )
        self._capture(barrier)

    def _capture(self, barrier: Callable[[], None] | None) -> None:
        logger.info_rank0(f"Capturing breakable prefill graphs for token counts: {self.buckets}")
        ctx = get_global_ctx()
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            with torch.cuda.stream(self.stream), prefill_graph_warmup():
                for count in tqdm(
                    reversed(self.buckets),
                    total=len(self.buckets),
                    desc="Capturing prefill graphs",
                    disable=not get_tp_info().is_primary(),
                ):
                    batch = ForwardBatch(
                        size=1,
                        phase="prefill",
                        req_pool_indices=torch.zeros(1, dtype=torch.int32, device=self.device),
                        seq_lens=torch.full((1,), count, dtype=torch.int32, device=self.device),
                        extend_lens=torch.full((1,), count, dtype=torch.int32, device=self.device),
                        seq_lens_cpu=(count,),
                        extend_lens_cpu=(count,),
                    )
                    batch.input_ids = self.input_ids[:count]
                    batch.positions = self.positions[:count]
                    batch.out_loc = torch.zeros(count, dtype=torch.int32, device=self.device)
                    batch.prefill_num_tokens = count
                    graph = BreakableCUDAGraph()
                    with ctx.forward_batch(batch):
                        for _ in range(2):
                            self.output[:count].copy_(self.model.model.forward(batch.input_ids))
                        self.stream.synchronize()
                        if barrier is not None:
                            barrier()
                        with BreakableCUDAGraphCapture(graph, pool=self.pool, barrier_fn=barrier):
                            self.output[:count].copy_(self.model.model.forward(batch.input_ids))
                    self.graphs[count] = graph
                    self.capture_batches[count] = batch
        except BaseException:
            self.destroy()
            raise
        finally:
            if gc_enabled:
                gc.enable()
        self.stream.synchronize()

    def bucket_for(self, batch: ForwardBatch) -> int | None:
        if not batch.is_prefill or not self.buckets:
            return None
        count = sum(batch.extend_lens_cpu)
        if count < 1:
            return None
        index = bisect_left(self.buckets, count)
        if index == len(self.buckets):
            return None
        bucket = self.buckets[index]
        # Upstream rejects replay when padding would more than double the work.
        return bucket if bucket <= 2 * count else None

    def replay(
        self,
        batch: ForwardBatch,
        attn_backend: BaseAttnBackend,
        *,
        timing_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None,
    ) -> torch.Tensor:
        bucket = self.bucket_for(batch)
        if bucket is None:
            raise ValueError("Prefill batch is not eligible for CUDA graph replay")
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("Prefill graphs must replay on the model runner stream")
        count = sum(batch.extend_lens_cpu)
        self.input_ids[:count].copy_(batch.input_ids)
        self.positions[:count].copy_(batch.positions)
        self.input_ids[count:bucket].zero_()
        self.positions[count:bucket].zero_()
        live_batch = copy.copy(batch)
        live_batch.input_ids = self.input_ids[:bucket]
        live_batch.positions = self.positions[:bucket]
        # Request axes and KV write indices retain their real, unpadded lengths.
        live_batch.prefill_num_tokens = count
        with get_global_ctx().forward_batch(live_batch):
            attn_backend.prepare_metadata(live_batch)
            live_batch.record_metadata_ready()
            batch.metadata_ready = live_batch.metadata_ready
            batch.attn_metadata = live_batch.attn_metadata
            if timing_events is not None:
                timing_events[0].record(self.stream)
            self.graphs[bucket].replay()
            self.num_replays += 1
            logits = self.model.lm_head.forward(self.output[:count])
            if timing_events is not None:
                timing_events[1].record(self.stream)
            return logits

    def destroy(self) -> None:
        """Release closures and graph pools before distributed teardown."""
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.capture_batches.clear()
