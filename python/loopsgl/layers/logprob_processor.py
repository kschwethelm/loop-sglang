# Adapted from SGLang's layers/logprob_processor.py.
# Copyright 2023-2026 SGLang Team. Licensed under Apache-2.0.
# See LICENSES/sglang.txt.

from __future__ import annotations

import torch


def compute_row_log_normalizer(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FP32 row maximum and log-sum-exp of shifted logits."""
    x = logits.float()
    row_max = x.amax(dim=-1)
    row_log_sum = torch.logsumexp(x - row_max[:, None], dim=-1)
    row_log_sum = torch.where(row_max.isinf(), 0.0, row_log_sum)
    return row_max, row_log_sum


def get_top_logprobs_raw(
    logits: torch.Tensor, normalizer: tuple[torch.Tensor, torch.Tensor], count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return original-distribution top scores with the original argmax first."""
    values, indices = torch.topk(logits, count, dim=-1)
    greedy = logits.argmax(dim=-1)
    # topk can choose a different maximum when multiple vocabulary entries tie.
    missing = ~(indices == greedy[:, None]).any(dim=-1)
    indices[:, -1] = torch.where(missing, greedy, indices[:, -1])
    greedy_column = (indices == greedy[:, None]).to(torch.int32).argmax(dim=-1)
    first = indices[:, 0].clone()
    indices.scatter_(1, greedy_column[:, None], first[:, None])
    indices[:, 0] = greedy
    row_max, row_log_sum = normalizer
    values = (logits.gather(1, indices).float() - row_max[:, None]) - row_log_sum[:, None]
    return values, indices


def compute_token_logprobs(
    logits: torch.Tensor, tokens: torch.Tensor, top_count: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    normalizer = compute_row_log_normalizer(logits)
    row_max, row_log_sum = normalizer
    scores = (logits.gather(1, tokens.long()[:, None]).squeeze(1).float() - row_max) - row_log_sum
    if top_count == 0:
        return scores, None, None
    values, indices = get_top_logprobs_raw(logits, normalizer, top_count)
    return scores, values, indices
