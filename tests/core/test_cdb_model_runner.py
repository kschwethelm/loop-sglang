from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from loopsgl.core import LoopParams
from loopsgl.model_executor.cdb_model_runner import CDBModelRunner
from loopsgl.models.looped.ouro import OuroModel
from torch.nn import functional as F


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("exit_threshold", [None, 0.2])
@torch.inference_mode()
def test_greedy_prelude_state(device: str, exit_threshold: float | None) -> None:
    """Greedy selection resets successor rows without changing other requests, also on replay."""
    vocab, hidden_size = 8193, 16
    embedding = (
        (torch.arange(vocab * hidden_size, device=device) % 251)
        .reshape(vocab, hidden_size)
        .to(torch.bfloat16)
    )
    model = OuroModel.__new__(OuroModel)
    model._exit_gate_type = "early_exit"
    model.loop_params = LoopParams(loop_steps=4, exit_threshold=exit_threshold)
    model.embed_tokens = SimpleNamespace(
        weight=embedding, tp_size=1, forward=lambda ids: F.embedding(ids, embedding)
    )
    runner = CDBModelRunner.__new__(CDBModelRunner)
    runner.model = SimpleNamespace(model=model)
    runner.state_pool = {
        "hidden": torch.full((6, hidden_size), -1, device=device, dtype=embedding.dtype)
    }
    if exit_threshold is not None:
        runner.state_pool["exit"] = torch.full((6,), 0.75, device=device, dtype=torch.float32)

    slots = torch.tensor([4, 0, 2], device=device)
    tokens = torch.empty(3, device=device, dtype=torch.int32)
    logits = torch.full((3, vocab), -1, device=device, dtype=embedding.dtype)
    logits[0, 4095:4097] = 1  # A tie across reduction chunks selects the first maximum.
    logits[1, -1] = 1
    logits[2, 1] = logits[2, -1] = float("nan")

    def check_state() -> None:
        expected_tokens = torch.argmax(logits, dim=-1)
        torch.testing.assert_close(tokens, expected_tokens.to(torch.int32))
        expected_hidden = torch.full_like(runner.state_pool["hidden"], -1)
        expected_hidden[slots] = F.embedding(expected_tokens, embedding)
        torch.testing.assert_close(runner.state_pool["hidden"], expected_hidden, rtol=0, atol=0)
        if exit_threshold is not None:
            expected_exit = torch.full((6,), 0.75, device=device, dtype=torch.float32)
            expected_exit[slots] = 0
            torch.testing.assert_close(runner.state_pool["exit"], expected_exit, rtol=0, atol=0)

    fused = runner._write_greedy_prelude(logits, slots, tokens)
    assert fused == (device == "cuda" and exit_threshold is not None)
    check_state()
    if device == "cpu":
        return

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            runner._write_greedy_prelude(logits, slots, tokens)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        runner._write_greedy_prelude(logits, slots, tokens)

    for _ in range(2):
        logits.copy_(logits.roll(1, dims=1))
        runner.state_pool["hidden"].fill_(-1)
        if exit_threshold is not None:
            runner.state_pool["exit"].fill_(0.75)
        graph.replay()
        check_state()
