# Adapted from SGLang's layers/logits_processor.py and logprob_processor.py.
# Copyright 2023-2026 SGLang Team. Licensed under Apache-2.0.
# See LICENSES/sglang.txt.

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from .logprob_processor import compute_token_logprobs

if TYPE_CHECKING:
    from loopsgl.core import ForwardBatch
    from loopsgl.layers.embedding import ParallelLMHead


@dataclass(frozen=True)
class LogitsMetadata:
    return_logprob_rows_cpu: tuple[bool, ...]
    top_logprobs_nums: tuple[int, ...]
    extend_seq_lens_cpu: tuple[int, ...]
    extend_logprob_start_lens_cpu: tuple[int, ...]
    input_logprob_offsets_cpu: tuple[int, ...]
    output_positions_cpu: tuple[int, ...]
    prompt_lens_cpu: tuple[int, ...]
    input_ids_cpu: tuple[tuple[int, ...], ...]

    @property
    def has_input_logprobs(self) -> bool:
        return any(
            enabled and start < length
            for enabled, start, length in zip(
                self.return_logprob_rows_cpu,
                self.extend_logprob_start_lens_cpu,
                self.extend_seq_lens_cpu,
            )
        )


@dataclass
class LogitsProcessorOutput:
    next_token_logits: torch.Tensor | None = None
    next_token_logprobs: torch.Tensor | None = None
    input_token_logprobs: torch.Tensor | None = None
    next_token_top_logprobs_val: torch.Tensor | None = None
    next_token_top_logprobs_idx: torch.Tensor | None = None
    input_top_logprobs_val: torch.Tensor | None = None
    input_top_logprobs_idx: torch.Tensor | None = None

    def copy_to_cpu(self) -> LogitsProcessorOutput:
        """Copy only compact scores on the current stream before its result event."""
        result = LogitsProcessorOutput()
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name != "next_token_logits" and value is not None:
                host = torch.empty_like(value, device="cpu", pin_memory=True)
                host.copy_(value, non_blocking=True)
                if value.is_cuda:
                    value.record_stream(torch.cuda.current_stream(value.device))
                setattr(result, item.name, host)
        return result


class LogitsProcessor:
    """Project input predictors in bounded groups after the complete prefill."""

    def forward(
        self,
        hidden: torch.Tensor,
        lm_head: ParallelLMHead,
        batch: ForwardBatch,
        metadata: LogitsMetadata,
    ) -> LogitsProcessorOutput:
        final_rows = batch.attn_metadata.get_last_indices(batch.size)
        result = LogitsProcessorOutput(next_token_logits=lm_head.project(hidden[final_rows]))
        predictors: list[int] = []
        targets: list[int] = []
        cursor = 0
        for enabled, length, start, ids in zip(
            metadata.return_logprob_rows_cpu,
            metadata.extend_seq_lens_cpu,
            metadata.extend_logprob_start_lens_cpu,
            metadata.input_ids_cpu,
        ):
            if enabled:
                predictors.extend(range(cursor + start, cursor + length - 1))
                targets.extend(ids[start + 1 :])
            cursor += length
        top_count = max(metadata.top_logprobs_nums, default=0)
        result.input_token_logprobs = hidden.new_empty(len(targets), dtype=torch.float32)
        if top_count:
            result.input_top_logprobs_val = hidden.new_empty(
                (len(targets), top_count), dtype=torch.float32
            )
            result.input_top_logprobs_idx = torch.empty(
                (len(targets), top_count), dtype=torch.int64, device=hidden.device
            )
        for start in range(0, len(targets), 2048):
            stop = min(start + 2048, len(targets))
            rows = torch.tensor(
                predictors[start:stop], dtype=torch.int64, pin_memory=hidden.is_cuda
            ).to(hidden.device, non_blocking=True)
            ids = torch.tensor(
                targets[start:stop], dtype=torch.int64, pin_memory=hidden.is_cuda
            ).to(hidden.device, non_blocking=True)
            logits = lm_head.project(hidden[rows])
            scores, values, indices = compute_token_logprobs(logits, ids, top_count)
            result.input_token_logprobs[start:stop] = scores
            if values is not None:
                result.input_top_logprobs_val[start:stop] = values
                result.input_top_logprobs_idx[start:stop] = indices
        return result
