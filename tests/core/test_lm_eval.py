from __future__ import annotations

from types import SimpleNamespace

import pytest
from loopsgl.core import LoopParams
from loopsgl.entrypoints.engine import Engine
from loopsgl.server_args import ServerArgs


@pytest.mark.parametrize(
    ("model_args", "expected"),
    [
        ("", LoopParams()),
        (",loop_steps=8", LoopParams(loop_steps=8)),
        (",exit_threshold=0.5", LoopParams(exit_threshold=0.5)),
        (",loop_steps=8,exit_threshold=0.5", LoopParams(loop_steps=8, exit_threshold=0.5)),
    ],
)
def test_lm_eval_loop_params(
    model_args: str, expected: LoopParams, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scalar evaluation arguments reach the offline scheduler as grouped loop settings."""
    pytest.importorskip("lm_eval")
    from loopsgl.evaluation import LoopSGLangLM

    configs: list[ServerArgs] = []

    def capture_config(self: Engine, server_args: ServerArgs, startup_timeout: float) -> tuple:
        configs.append(server_args)
        return (
            SimpleNamespace(tokenizer=SimpleNamespace(), shutdown=lambda: None),
            {"max_seq_len": 128},
            [],
        )

    monkeypatch.setattr(Engine, "_launch_subprocesses", capture_config)
    LoopSGLangLM.create_from_arg_string(f"pretrained=unused,loop_cache_policy=shared{model_args}")
    assert len(configs) == 1
    assert configs[0].loop_params == expected
    assert configs[0].loop_cache_policy == "shared"
