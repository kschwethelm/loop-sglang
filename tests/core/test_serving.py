from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import orjson
import pytest
from loopsgl.benchmark import serving
from loopsgl.benchmark.datasets import DatasetRow
from loopsgl.core import SamplingParams


def test_absolute_arrivals(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first arrival is immediate and dispatch overhead does not accumulate."""
    clock = [10.0]
    sleeps = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(serving, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(serving, "asyncio", SimpleNamespace(sleep=sleep))
    requests = [object(), object(), object()]
    offsets = np.random.default_rng(7).exponential(0.5, size=2).cumsum()

    async def check() -> None:
        seen = []
        async for request in serving.get_request(requests, 2, 7, 10.0):
            seen.append(request)
            if len(seen) == 1:
                clock[0] += 0.1
        assert seen == requests

    asyncio.run(check())
    assert sleeps == pytest.approx(
        [
            max(0, offsets[0] - 0.1),
            offsets[1] - max(offsets[0], 0.1),
        ]
    )


@pytest.mark.parametrize("failure", [None, "truncated", "abort", "count"])
def test_native_stream_timing_and_completion(
    failure: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First visible text controls TTFT while terminal metadata validates every stream."""
    clock = [1.0]
    finish = {"type": "abort"} if failure == "abort" else {"type": "length", "length": 4}
    final_count = 3 if failure == "count" else 4
    events = [
        (1.01, {"text": "", "meta_info": {"completion_tokens": 1}}),
        (1.04, {"text": "a", "meta_info": {"completion_tokens": 2}}),
        (
            1.10,
            {
                "text": "abc",
                "meta_info": {"completion_tokens": final_count, "finish_reason": finish},
            },
        ),
    ]

    class Response:
        status = 200

        @property
        def content(self) -> Any:
            async def chunks() -> Any:
                for timestamp, event in events:
                    clock[0] = timestamp
                    yield b"data: " + orjson.dumps(event) + b"\n"
                if failure != "truncated":
                    clock[0] = 1.12
                    yield b"data: [DONE]\n"

            return chunks()

        async def __aenter__(self) -> Response:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    class Session:
        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        def post(self, *args: Any, **kwargs: Any) -> Response:
            return Response()

    monkeypatch.setattr(serving, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(serving, "_create_bench_client_session", Session)
    request = serving.RequestFuncInput(
        [8, 9],
        "http://localhost/generate",
        2,
        4,
        "unused",
        SamplingParams(max_new_tokens=4, ignore_eos=True),
    )
    output = asyncio.run(serving.async_request_sglang_generate(request))
    assert output.success == (failure is None)
    if failure is None:
        assert output.start_time == 1.0
        assert output.ttft == pytest.approx(0.04)
        assert output.itl == pytest.approx([0.03, 0.03])
        assert output.latency == pytest.approx(0.12)
        assert output.output_len == 4
    else:
        assert output.error


def test_normalized_latency() -> None:
    """The paper metric averages request ratios with unequal generated lengths."""
    rows = [DatasetRow("a", 1, 2), DatasetRow("b", 1, 3)]
    outputs = [
        serving.RequestFuncOutput(success=True, latency=2, output_len=2),
        serving.RequestFuncOutput(success=True, latency=9, output_len=3),
    ]
    metrics, lengths = serving.calculate_metrics(
        rows, outputs, 10, SimpleNamespace(encode=lambda *args, **kwargs: [])
    )
    assert metrics.mean_normalized_latency_s_per_token == 2
    assert lengths == [2, 3]


def test_failed_point_is_not_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed stream invalidates the complete measured point at the benchmark entrypoint."""

    async def failed_request(*args: Any, **kwargs: Any) -> serving.RequestFuncOutput:
        return serving.RequestFuncOutput(error="Truncated stream")

    class Response:
        def raise_for_status(self) -> None:
            pass

        async def json(self) -> dict[str, Any]:
            return {
                "max_seq_len": 128,
                "model_path": "unused",
                "scheduler": "cb",
                "num_retracted_reqs": 0,
                "server_args": {},
            }

        async def __aenter__(self) -> Response:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    class Session:
        def get(self, *args: Any, **kwargs: Any) -> Response:
            return Response()

        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    output = tmp_path / "result.jsonl"
    args = serving.parse_args(
        [
            "--ready-check-timeout-sec",
            "0",
            "--num-prompts",
            "1",
            "--warmup-requests",
            "0",
            "--disable-tqdm",
            "--output-file",
            str(output),
        ]
    )
    monkeypatch.setattr(serving, "_create_bench_client_session", Session)
    monkeypatch.setattr(serving, "async_request_sglang_generate", failed_request)
    monkeypatch.setattr(serving, "load_tokenizer", lambda *args: None)
    monkeypatch.setattr(serving, "get_dataset", lambda *args, **kwargs: [DatasetRow([8, 9], 2, 4)])
    with pytest.raises(RuntimeError, match="1 of 1 requests failed"):
        asyncio.run(serving.benchmark(args))
    assert not output.exists()


@pytest.mark.parametrize("empty", [False, True])
def test_serving_cycles_replay_workload(
    empty: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cycled replay records its mean depth and measured retractions, excluding warmup."""
    rows = (
        []
        if empty
        else [
            DatasetRow([8, 9], 2, 3, exit_depths=[2, 3]),
            DatasetRow([10, 11], 2, 4, exit_depths=[4, 3, 2]),
        ]
    )
    seen = []
    retractions = [100]

    async def complete_request(
        request: serving.RequestFuncInput, *_: Any
    ) -> serving.RequestFuncOutput:
        seen.append(request)
        retractions[0] += 5 if len(seen) <= args.warmup_requests else 1
        return serving.RequestFuncOutput(
            success=True,
            latency=0.1,
            ttft=0.01,
            itl=[0.03] * (request.output_len - 1),
            output_len=request.output_len,
        )

    async def server_info() -> dict[str, Any]:
        return {
            "max_seq_len": 128,
            "model_path": "unused",
            "scheduler": "cdb",
            "num_retracted_reqs": retractions[0],
            "loop_params": {"loop_steps": 4},
            "server_args": {
                "max_running_req": 2,
                "cdb_config": {"disable_lookahead_exit": False},
            },
        }

    response = MagicMock()
    response.json = AsyncMock(side_effect=server_info)
    session = MagicMock()
    session.__aenter__.return_value = session
    session.get.return_value.__aenter__.return_value = response
    monkeypatch.setattr(serving, "_create_bench_client_session", lambda: session)
    monkeypatch.setattr(serving, "async_request_sglang_generate", complete_request)
    monkeypatch.setattr(
        serving, "load_tokenizer", lambda *args: SimpleNamespace(encode=lambda *args, **kwargs: [])
    )
    monkeypatch.setattr(serving, "get_dataset", lambda *args, **kwargs: rows)
    output = tmp_path / "result.jsonl"
    args = serving.parse_args(
        [
            "--ready-check-timeout-sec",
            "0",
            "--num-prompts",
            "7",
            "--disable-tqdm",
            "--dataset-name",
            "custom",
            "--custom-input-format",
            "token-ids",
            "--no-shuffle",
            "--replay-exits",
            "--extra-request-body",
            '{"temperature": 0.3, "top_k": 5, "skip_special_tokens": true}',
            "--output-file",
            str(output),
        ]
    )
    if empty:
        with pytest.raises(ValueError, match="No requests remain"):
            asyncio.run(serving.benchmark(args))
        assert not seen
        assert not output.exists()
        return

    result = asyncio.run(serving.benchmark(args))
    assert args.warmup_requests == 4
    warmup, seen = seen[:4], seen[4:]
    assert len({id(request) for request in warmup}) == 4
    assert {tuple(request.prompt) for request in warmup} == {(8, 9), (10, 11)}
    expected = [rows[index % 2] for index in range(7)]
    assert [request.prompt for request in seen] == [row.prompt for row in expected]
    assert [request.output_len for request in seen] == [row.output_len for row in expected]
    assert [request.sampling_params.replay_exit_depths for request in seen] == [
        row.exit_depths for row in expected
    ]
    assert len({id(request) for request in seen}) == 7
    assert all(
        request.sampling_params.temperature == 0.3
        and request.sampling_params.top_k == 5
        and request.sampling_params.skip_special_tokens
        for request in warmup + seen
    )
    saved = orjson.loads(output.read_bytes())
    assert saved["completed"] == 7
    assert saved["total_output"] == sum(row.output_len for row in expected)
    assert saved["mean_decode_depth"] == pytest.approx(47 / 17)
    assert saved["num_retracted_reqs"] == 7
    assert saved["server_info"]["num_retracted_reqs"] == 127
    assert result == saved


def test_single_token_metrics_are_serializable() -> None:
    """A trace with no inter-token samples still produces valid JSON metrics."""
    metrics, _ = serving.calculate_metrics(
        [DatasetRow([8], 1, 1)],
        [serving.RequestFuncOutput(success=True, latency=0.1, ttft=0.1, output_len=1)],
        0.2,
        SimpleNamespace(encode=lambda *args, **kwargs: [8]),
    )
    saved = json.loads(json.dumps(asdict(metrics), allow_nan=False))
    assert saved["max_itl_ms"] == 0
    assert saved["mean_tpot_ms"] == 0
