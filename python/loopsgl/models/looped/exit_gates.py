from __future__ import annotations

import torch


def cumulative_exit_probability(
    logits: torch.Tensor, previous_probability: torch.Tensor
) -> torch.Tensor:
    """Accumulate sigmoid exit hazards into cumulative probabilities in FP32.

    Logits and previous probabilities have matching shapes, with one value per token.
    """
    probability = logits.float().sigmoid()
    previous = previous_probability.float()
    return previous + (1.0 - previous) * probability


def relative_state_change(
    current: torch.Tensor, previous: torch.Tensor, *, eps: float = 1e-9
) -> torch.Tensor:
    """Return relative L2 state change, reducing the hidden dimension in FP32."""
    state = current.float()
    previous_state = previous.float()
    return (state - previous_state).norm(dim=-1) / state.norm(dim=-1).clamp_min(eps)
