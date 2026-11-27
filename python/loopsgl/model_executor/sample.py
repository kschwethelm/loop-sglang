from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, List

import torch

from loopsgl.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from loopsgl.layers.logprob_processor import compute_token_logprobs
from loopsgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from loopsgl.managers.schedule_batch import ScheduleBatch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    return_logprob: tuple[bool, ...] | None = None
    top_logprobs_nums: tuple[int, ...] | None = None
    trace_decode_token_ids: tuple[list[int] | None, ...] | None = None
    forced_token_ids: torch.Tensor | None = None
    logits_metadata: LogitsMetadata | None = None

    @property
    def needs_logprobs_or_trace(self) -> bool:
        return self.return_logprob is not None or self.trace_decode_token_ids is not None

    def snapshot(self, metadata: LogitsMetadata, device: torch.device) -> BatchSamplingArgs:
        targets = None
        if self.trace_decode_token_ids is not None:
            values = []
            for trace, position, prompt_len in zip(
                self.trace_decode_token_ids, metadata.output_positions_cpu, metadata.prompt_lens_cpu
            ):
                index = position - prompt_len
                values.append(trace[index] if trace is not None and 0 <= index < len(trace) else -1)
            targets = make_device_tensor(values, torch.int32, device)
        return replace(self, forced_token_ids=targets, logits_metadata=metadata)

    def filter_batch(
        self,
        indices: torch.Tensor,
        all_greedy: bool,
        *,
        need_top_k: bool,
        need_top_p: bool,
        cpu_indices: list[int] | None = None,
    ) -> BatchSamplingArgs:
        flags = (
            tuple(self.return_logprob[i] for i in cpu_indices)
            if self.return_logprob is not None
            else None
        )
        if flags is not None and not any(flags):
            flags = None
        counts = (
            tuple(self.top_logprobs_nums[i] for i in cpu_indices) if flags is not None else None
        )
        traces = (
            tuple(self.trace_decode_token_ids[i] for i in cpu_indices)
            if self.trace_decode_token_ids is not None
            else None
        )
        if traces is not None and all(trace is None for trace in traces):
            traces = None
        return BatchSamplingArgs(
            (
                self.temperatures[indices]
                if self.temperatures is not None and not all_greedy
                else None
            ),
            self.top_k[indices] if self.top_k is not None and need_top_k else None,
            self.top_p[indices] if self.top_p is not None and need_top_p else None,
            return_logprob=flags,
            top_logprobs_nums=counts,
            trace_decode_token_ids=traces,
        )

    def merge_batch(
        self, other: BatchSamplingArgs, size: int, other_size: int, vocab_size: int
    ) -> BatchSamplingArgs:
        def merge(
            left: torch.Tensor | None, right: torch.Tensor | None, default: int | float
        ) -> torch.Tensor | None:
            if left is None and right is None:
                return None
            template = left if left is not None else right
            assert template is not None
            if left is None:
                left = template.new_full((size,), default)
            if right is None:
                right = template.new_full((other_size,), default)
            return torch.cat([left, right])

        def merge_rows(left: tuple | None, right: tuple | None, default: object) -> tuple | None:
            if left is None and right is None:
                return None
            return (left if left is not None else (default,) * size) + (
                right if right is not None else (default,) * other_size
            )

        return BatchSamplingArgs(
            merge(self.temperatures, other.temperatures, 1e-6),
            merge(self.top_k, other.top_k, vocab_size),
            merge(self.top_p, other.top_p, 1.0),
            return_logprob=merge_rows(self.return_logprob, other.return_logprob, False),
            top_logprobs_nums=merge_rows(self.top_logprobs_nums, other.top_logprobs_nums, 0),
            trace_decode_token_ids=merge_rows(
                self.trace_decode_token_ids, other.trace_decode_token_ids, None
            ),
        )


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: ScheduleBatch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [p.top_p for p in params]
        temperatures = (
            None
            if all(p.is_greedy for p in params)
            else make_device_tensor(ts, torch.float32, self.device)
        )
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        result = BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)
        if any(req.return_logprob for req in batch.reqs):
            result.return_logprob = tuple(req.return_logprob for req in batch.reqs)
            result.top_logprobs_nums = tuple(req.top_logprobs_num for req in batch.reqs)
        if any(p.trace_decode_token_ids is not None for p in params):
            result.trace_decode_token_ids = tuple(p.trace_decode_token_ids for p in params)
        return result

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)

    def compute_logprobs(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        args: BatchSamplingArgs,
        output: LogitsProcessorOutput | None = None,
    ) -> LogitsProcessorOutput | None:
        """Fill compact output scores from the original model distribution."""
        if args.return_logprob is None:
            return output
        metadata = args.logits_metadata
        assert metadata is not None
        if output is None:
            output = LogitsProcessorOutput()
        rows = [
            i
            for i, enabled in enumerate(metadata.return_logprob_rows_cpu)
            if enabled and metadata.output_positions_cpu[i] >= 0
        ]
        if not rows:
            return output
        indices = torch.tensor(rows, dtype=torch.int64, pin_memory=logits.is_cuda).to(
            logits.device, non_blocking=True
        )
        top_count = max(metadata.top_logprobs_nums[i] for i in rows)
        scores, values, token_ids = compute_token_logprobs(
            logits[indices], tokens[indices], top_count
        )
        output.next_token_logprobs = logits.new_zeros(logits.shape[0], dtype=torch.float32)
        output.next_token_logprobs[indices] = scores
        if values is not None:
            output.next_token_top_logprobs_val = values.new_zeros((logits.shape[0], top_count))
            output.next_token_top_logprobs_idx = token_ids.new_zeros((logits.shape[0], top_count))
            output.next_token_top_logprobs_val[indices] = values
            output.next_token_top_logprobs_idx[indices] = token_ids
        return output
