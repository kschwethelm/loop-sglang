from __future__ import annotations

import asyncio
import json
import socket
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator, Callable
from unittest.mock import Mock

import aiohttp
import pytest
import torch
import uvicorn
import loopsgl.distributed.info as distributed_info
from loopsgl.core import Req, SamplingParams
from loopsgl.distributed import DistributedInfo
from loopsgl.entrypoints import http_server
from loopsgl.managers.base_scheduler import BaseScheduler
from loopsgl.managers.detokenizer_manager import DetokenizerManager
from loopsgl.managers.io_struct import AbortReq, TokenIDOutput, TokenizedGenerateReqInput
from loopsgl.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from loopsgl.managers.tokenizer_manager import TokenizerManager
from loopsgl.server_args import ServerArgs


@asynccontextmanager
async def serve_app(app: object) -> AsyncIterator[str]:
    """Serve an application on an owned localhost socket and reap its task."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(1000):
            if server.started:
                break
            if task.done():
                await task
                raise RuntimeError("Test HTTP server failed to start.")
            await asyncio.sleep(0.01)
        else:
            raise TimeoutError("Test HTTP server failed to become ready.")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=10)
        sock.close()


class NativeTestManager(TokenizerManager):
    """Exercise frontend preparation with CPU result processing and detokenization."""

    def auto_create_handle_loop(self) -> None:
        self._raise_if_unavailable()

    async def _send_one_request(self, obj: TokenizedGenerateReqInput) -> None:
        req = Req(obj.uid, obj.input_ids, obj.sampling_params)
        cache = Mock()
        cache.lazy_free_region = nullcontext
        processor = SchedulerBatchResultProcessor(
            tokenizer=self.tokenizer,
            eos_token_ids={2},
            cache_manager=cache,
            decode_manager=Mock(),
            free_req_resources=Mock(),
            prefix_insert_limit=lambda req: None,
            loop_steps=4,
        )
        detokenizer = DetokenizerManager.__new__(DetokenizerManager)
        detokenizer.tokenizer = self.tokenizer
        detokenizer.decode_map = {}
        for index in range(obj.sampling_params.max_new_tokens):
            token = 2 if index == obj.sampling_params.max_new_tokens - 1 else 8
            outputs = processor.process_batch_result(
                SimpleNamespace(
                    batch=SimpleNamespace(is_prefill=False), reqs=[req], is_middle_chunks=[False]
                ),
                SimpleNamespace(next_tokens_cpu=torch.tensor([token])),
            )
            self.rid_to_state[obj.uid].out_list.extend(detokenizer.detokenize(outputs))
        self.rid_to_state[obj.uid].event.set()


@pytest.mark.parametrize("stream", [False, True])
def test_native_generation_contract(
    stream: bool, tiny_checkpoint: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native responses retain generated EOS counts through the production metadata path."""
    cuda_initialized = torch.cuda.is_initialized()
    args = ServerArgs(
        model_path=str(tiny_checkpoint("ouro")), dtype=torch.bfloat16, tp_info=DistributedInfo(0, 1)
    )
    manager = NativeTestManager(args)
    monkeypatch.setattr(http_server, "_GLOBAL_STATE", manager)

    async def check() -> None:
        try:
            async with serve_app(http_server.app) as url, aiohttp.ClientSession() as session:
                async with session.get(url + "/health") as response:
                    assert response.status == 200
                payload = {
                    "input_ids": [8, 9],
                    "sampling_params": {"max_new_tokens": 2},
                    "stream": stream,
                }
                async with session.post(url + "/generate", json=payload) as response:
                    assert response.status == 200, await response.text()
                    if stream:
                        events = [
                            line[6:]
                            for line in (await response.text()).splitlines()
                            if line.startswith("data: ")
                        ]
                        assert events[-1] == "[DONE]"
                        result = json.loads(events[-2])
                    else:
                        result = await response.json()
                assert result["meta_info"] == {
                    "id": "0",
                    "prompt_tokens": 2,
                    "completion_tokens": 2,
                    "finish_reason": {"type": "stop", "matched": 2},
                }
                assert result["text"] == "t8"
                for invalid in (
                    {"prompt": "t8", "max_tokens": 2},
                    {"text": "t8", "input_ids": [8]},
                    {"text": ""},
                    {"input_ids": []},
                    {"input_ids": [True]},
                    {"input_ids": [64]},
                    {"input_ids": [8], "sampling_params": {"max_tokens": 2}},
                ):
                    async with session.post(url + "/generate", json=invalid) as response:
                        assert response.status in (400, 422)
                assert not manager.rid_to_state
        finally:
            await manager.close()

    asyncio.run(check())
    params = SamplingParams()
    assert (
        params.max_new_tokens,
        params.temperature,
        params.skip_special_tokens,
        params.top_k,
    ) == (1024, 0.0, False, -1)
    assert torch.cuda.is_initialized() == cuda_initialized


def test_abort_preserves_completion_count(
    tiny_checkpoint: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheduler aborts retain native metadata and the last generated count, including zero."""
    from loopsgl.utils import load_tokenizer

    monkeypatch.setattr(distributed_info, "_TP_INFO", DistributedInfo(0, 1))
    manager = DetokenizerManager.__new__(DetokenizerManager)
    manager.tokenizer = load_tokenizer(str(tiny_checkpoint("ouro")))
    manager.decode_map = {}
    for uid, count in ((0, 0), (1, 1)):
        if count:
            manager.detokenize([TokenIDOutput(uid, 8, False, completion_tokens=1)])
        scheduler = SimpleNamespace(
            send_result=Mock(),
            retraction_manager=SimpleNamespace(abort_req=Mock(return_value=True)),
        )
        BaseScheduler._process_one_msg(scheduler, AbortReq(uid))
        (messages,) = scheduler.send_result.call_args.args
        (output,) = manager.detokenize(messages)
        assert output.completion_tokens == count
        assert output.finish_reason == {
            "type": "abort",
            "message": "Aborted",
            "status_code": None,
            "err_type": None,
        }
        assert output.token_ids == ([8] if count else [])


def test_native_disconnect_aborts_request(
    tiny_checkpoint: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing a native stream schedules abort work outside the cancelled response task."""
    args = ServerArgs(
        model_path=str(tiny_checkpoint("ouro")),
        dtype=torch.bfloat16,
        tp_info=DistributedInfo(0, 1),
    )
    manager = NativeTestManager(args)
    monkeypatch.setattr(http_server, "_GLOBAL_STATE", manager)

    async def check() -> None:
        aborted = asyncio.Event()

        async def hold_request(obj: TokenizedGenerateReqInput) -> None:
            pass

        async def abort_request(uid: int) -> None:
            # Scheduler IPC can suspend while the HTTP response task is being cancelled.
            await asyncio.sleep(0.01)
            manager.rid_to_state.pop(uid, None)
            aborted.set()

        monkeypatch.setattr(manager, "_send_one_request", hold_request)
        monkeypatch.setattr(manager, "abort_request", abort_request)
        try:
            async with serve_app(http_server.app) as url, aiohttp.ClientSession() as session:
                response = await session.post(
                    url + "/generate",
                    json={"input_ids": [8, 9], "stream": True},
                )
                assert response.status == 200
                response.close()
                await asyncio.wait_for(aborted.wait(), timeout=5)
                assert not manager.rid_to_state
        finally:
            await manager.close()

    asyncio.run(check())
