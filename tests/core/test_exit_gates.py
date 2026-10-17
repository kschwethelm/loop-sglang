from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from loopsgl.core import LoopParams
from loopsgl.layers import LinearReplicated
from loopsgl.models.looped.base import BaseLoopedModel
from loopsgl.models.looped.huginn import HuginnModel
from loopsgl.models.looped.ouro import OuroModel


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ouro_cumulative_exit_state(dtype: torch.dtype) -> None:
    """Repeated subthreshold hazards accumulate independently for reordered tokens."""
    model = OuroModel.__new__(OuroModel)
    model.min_loop_steps = 1
    model._exit_gate_type = "early_exit"
    model.loop_params = LoopParams(loop_steps=4, exit_threshold=0.5)
    model.early_exit_gate = SimpleNamespace(forward=lambda hidden: hidden[:, :1])
    hidden = torch.logit(torch.tensor([[0.3], [0.1]])).to(dtype)
    initial = model.init_exit_state(hidden)
    first_signal, first_state = model.exit_signal(hidden, initial, 0)
    assert first_signal.dtype == torch.bool
    assert not first_signal.any()
    assert first_state.dtype == torch.float32

    order = torch.tensor([1, 0])
    signal, state = model.exit_signal(hidden[order], first_state[order], 1)
    assert signal.tolist() == [False, True]
    torch.testing.assert_close(state, torch.tensor([0.19, 0.51]), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(first_state, torch.tensor([0.3, 0.1]), atol=1e-3, rtol=1e-3)
    assert not initial.any()
    assert not model.init_exit_state(hidden).any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_huginn_previous_state_and_threshold(dtype: torch.dtype) -> None:
    """Zero states stay finite, equality does not exit, and successive calls advance history."""
    model = HuginnModel.__new__(HuginnModel)
    model.loop_params = LoopParams(loop_steps=4, exit_threshold=0.5)
    previous = torch.tensor([[0, 0], [1, 0], [-1, -1], [1, 1]], dtype=dtype)
    current = torch.tensor([[0, 0], [2, 0], [1, 1], [1, 1]], dtype=dtype)
    previous_before, current_before = previous.clone(), current.clone()
    initial = model.init_exit_state(previous)
    assert initial is previous

    signal, state = model.exit_signal(current, initial, 0)
    assert signal.dtype == torch.bool
    assert signal.tolist() == [True, False, False, True]
    assert state is current
    assert state.dtype == dtype
    following = current.clone()
    next_signal, next_state = model.exit_signal(following, state, 1)
    assert next_signal.all()
    assert next_state is following
    torch.testing.assert_close(previous, previous_before)
    torch.testing.assert_close(current, current_before)


@pytest.mark.skipif(
    not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available(),
    reason="CUDA graph checks require CUDA in a SLURM allocation",
)
@pytest.mark.parametrize("model_type", [OuroModel, HuginnModel])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_exit_gate_cuda_graph(model_type: type[BaseLoopedModel], dtype: torch.dtype) -> None:
    """Captured gates replay correctly as hidden states and exit history change."""
    hidden = torch.ones(4, 128, device="cuda", dtype=dtype)
    previous = torch.zeros_like(hidden)
    model = model_type.__new__(model_type)
    model.loop_params = LoopParams(loop_steps=4, exit_threshold=0.5)
    if model_type is OuroModel:
        model.min_loop_steps = 1
        model._exit_gate_type = "early_exit"
        gate = LinearReplicated(128, 1, has_bias=True)
        gate.weight = torch.zeros(1, 128, device="cuda", dtype=dtype)
        gate.weight[:, 0] = 1
        gate.bias = torch.zeros(1, device="cuda", dtype=dtype)
        model.early_exit_gate = gate
        hidden[:, 0] = -0.8472978604  # Sigmoid probability 0.3.
    state = model.init_exit_state(previous)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.init_exit_state(previous)
            model.exit_signal(hidden, state, 0)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_initial = model.init_exit_state(previous)
        captured_signal, captured_state = model.exit_signal(hidden, state, 0)

    for step in range(2):
        expected_initial = model.init_exit_state(previous).clone()
        hidden_before, state_before = hidden.clone(), state.clone()
        expected_signal, expected_state = model.exit_signal(hidden, state, 0)
        graph.replay()
        torch.testing.assert_close(captured_initial, expected_initial)
        torch.testing.assert_close(captured_signal, expected_signal)
        torch.testing.assert_close(captured_state, expected_state)
        torch.testing.assert_close(hidden, hidden_before)
        torch.testing.assert_close(state, state_before)
        assert captured_signal.all().item() == (step == 1)
        state.copy_(captured_state)
        hidden.mul_(1.02)
