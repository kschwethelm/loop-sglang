from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from loopsgl.models.config import ModelConfig
from loopsgl.models.looped.config import OuroConfig
from loopsgl.models.looped.huginn import HuginnForCausalLM


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, True),
        ({"rope_is_neox": False}, False),
        ({"rope_is_neox_style": False}, False),
        ({"rope_is_neox_style": True, "rope_is_neox": False}, True),
    ],
)
def test_rotary_style_config(fields: dict[str, bool], expected: bool) -> None:
    config = OuroConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_size=128,
        vocab_size=64,
        intermediate_size=256,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000,
        max_position_embeddings=128,
        **fields,
    )
    assert ModelConfig.from_hf(config).rotary_config.is_neox_style is expected


@pytest.mark.parametrize("std", [0.0, -0.1])
def test_huginn_nonpositive_state_init_std(std: float) -> None:
    """Nonpositive initialization variance preserves the zero recurrent state."""
    injection = torch.ones(3, 8)
    model = HuginnForCausalLM.__new__(HuginnForCausalLM)
    model.model = SimpleNamespace(
        embed_tokens=SimpleNamespace(forward=lambda input_ids: injection),
        prelude=SimpleNamespace(op_list=[]),
    )
    model._embed_scale = 2.0
    model._state_init = "like-init"
    model._state_init_std = std
    state, actual_injection = model.prelude(torch.tensor([1, 2, 3]))
    torch.testing.assert_close(state, torch.zeros_like(injection))
    torch.testing.assert_close(actual_injection, injection * 2)
