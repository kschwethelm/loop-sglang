# Portions adapted from SGLang's schedule_batch.py.
# Copyright 2023-2024 SGLang Team.
# Licensed under the Apache License, Version 2.0.
# See LICENSES/sglang.txt in the repository root.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Tuple

import torch

from loopsgl.core import ForwardBatch, Req
from loopsgl.model_executor.sample import BatchSamplingArgs, make_device_tensor

if TYPE_CHECKING:
    from loopsgl.model_executor.sample import Sampler


@dataclass
class BatchSizeStats:
    """Count actual and padded input tokens over executed forwards."""

    num_batches: int = 0
    num_tokens: int = 0
    num_padded_tokens: int = 0

    def record(self, batch: ForwardBatch, *, padded_tokens: int | None = None) -> None:
        tokens = sum(batch.extend_lens_cpu) if batch.is_prefill else batch.size
        if padded_tokens is None:
            padded_tokens = tokens if batch.is_prefill else batch.padded_size
        self.num_batches += 1
        self.num_tokens += tokens
        self.num_padded_tokens += padded_tokens


@dataclass
class ScheduleBatch:
    """Persistent scheduler state with GPU tensors updated as membership changes."""

    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    req_pool_indices: torch.Tensor = field(init=False)
    seq_lens: torch.Tensor = field(init=False)
    extend_lens: torch.Tensor = field(init=False)
    seq_lens_cpu: Tuple[int, ...] = field(init=False)
    extend_lens_cpu: Tuple[int, ...] = field(init=False)
    sampling_info: BatchSamplingArgs = field(init=False)
    vocab_size: int = field(init=False)
    initialized: bool = field(default=False, init=False)

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    def prepare(self, sampler: Sampler, *, prepare_sampling: bool = True) -> None:
        if self.initialized:
            return
        assert self.reqs and all(req.kv.req_pool_idx is not None for req in self.reqs)
        self.seq_lens_cpu = tuple(req.device_len for req in self.reqs)
        self.extend_lens_cpu = tuple(req.extend_len for req in self.reqs)
        self.req_pool_indices = make_device_tensor(
            [req.kv.req_pool_idx for req in self.reqs], torch.int32, sampler.device
        )
        self.seq_lens = make_device_tensor(list(self.seq_lens_cpu), torch.int32, sampler.device)
        self.extend_lens = (
            make_device_tensor(list(self.extend_lens_cpu), torch.int32, sampler.device)
            if self.is_prefill
            else torch.ones_like(self.seq_lens)
        )
        self.sampling_info = sampler.prepare(self) if prepare_sampling else BatchSamplingArgs(None)
        self.vocab_size = sampler.vocab_size
        self.initialized = True

    def filter_batch(self, keep_indices: List[int]) -> None:
        if keep_indices == list(range(self.size)):
            return
        self.reqs = [self.reqs[i] for i in keep_indices]
        if not self.initialized:
            return
        indices = make_device_tensor(keep_indices, torch.int64, self.seq_lens.device)
        self.req_pool_indices = self.req_pool_indices[indices]
        self.seq_lens = self.seq_lens[indices]
        self.extend_lens = self.extend_lens[indices]
        self.seq_lens_cpu = tuple(self.seq_lens_cpu[i] for i in keep_indices)
        self.extend_lens_cpu = tuple(self.extend_lens_cpu[i] for i in keep_indices)
        self.sampling_info = self.sampling_info.filter_batch(
            indices,
            all(req.sampling_params.is_greedy for req in self.reqs),
            need_top_k=any(1 <= req.sampling_params.top_k != self.vocab_size for req in self.reqs),
            need_top_p=any(req.sampling_params.top_p < 1.0 for req in self.reqs),
        )

    def merge_batch(self, other: ScheduleBatch, vocab_size: int) -> None:
        assert self.initialized and other.initialized and self.phase == other.phase == "decode"
        self.sampling_info = self.sampling_info.merge_batch(
            other.sampling_info, self.size, other.size, vocab_size
        )
        self.reqs = self.reqs + other.reqs
        self.req_pool_indices = torch.cat([self.req_pool_indices, other.req_pool_indices])
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.extend_lens = torch.cat([self.extend_lens, other.extend_lens])
        self.seq_lens_cpu += other.seq_lens_cpu
        self.extend_lens_cpu += other.extend_lens_cpu
        # Keep sampling order stable when an older offloaded request returns.
        self.filter_batch(sorted(range(self.size), key=lambda i: self.reqs[i].uid))

    def get_forward_batch(self) -> ForwardBatch:
        assert self.initialized
        return ForwardBatch(
            size=self.size,
            phase=self.phase,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            extend_lens=self.extend_lens,
            seq_lens_cpu=self.seq_lens_cpu,
            extend_lens_cpu=self.extend_lens_cpu,
        )

    def prepare_for_decode(self) -> None:
        assert self.initialized and not self.is_prefill
        # Overlap can still read the submitted lengths on another stream.
        self.seq_lens = self.seq_lens + 1
        self.seq_lens_cpu = tuple(length + 1 for length in self.seq_lens_cpu)
