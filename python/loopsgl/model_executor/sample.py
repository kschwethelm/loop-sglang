from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from loopsgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from loopsgl.managers.schedule_batch import ScheduleBatch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None

    def filter_batch(
        self, indices: torch.Tensor, all_greedy: bool, *, need_top_k: bool, need_top_p: bool
    ) -> BatchSamplingArgs:
        return BatchSamplingArgs(
            (
                self.temperatures[indices]
                if self.temperatures is not None and not all_greedy
                else None
            ),
            self.top_k[indices] if self.top_k is not None and need_top_k else None,
            self.top_p[indices] if self.top_p is not None and need_top_p else None,
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

        return BatchSamplingArgs(
            merge(self.temperatures, other.temperatures, 1e-6),
            merge(self.top_k, other.top_k, vocab_size),
            merge(self.top_p, other.top_p, 1.0),
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
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
