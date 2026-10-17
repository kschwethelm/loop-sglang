from __future__ import annotations

import asyncio
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch
from loopsgl import Engine
from loopsgl.core import SamplingParams
from loopsgl.managers.io_struct import (
    AbortReq,
    BaseReq,
    BatchTokenizedGenerateReqInput,
    GenerateReqInput,
    ResetMetricsReq,
    SchedulerControlOutput,
    SchedulerControlReq,
    TokenIDOutput,
    TokenizedGenerateReqInput,
)
from loopsgl.server_args import ServerArgs
from loopsgl.utils import ZmqPullQueue, ZmqPushQueue


def _cpu_scheduler(config: ServerArgs, ack_queue: mp.Queue, *, silent_output: bool = False) -> None:
    """Serve deterministic tokens over production IPC while retaining cancellable requests."""
    incoming = ZmqPullQueue(config.zmq_backend_addr, create=True, decoder=BaseReq.decoder)
    tokens = ZmqPushQueue(config.zmq_detokenizer_addr, create=True, encoder=BaseReq.encoder)
    replies = ZmqPushQueue(config.zmq_frontend_addr, create=False, encoder=BaseReq.encoder)
    info = {
        "max_seq_len": 128,
        "loop_params": {"loop_steps": 4},
        "num_retracted_reqs": 0,
        "batch_size_metrics": {},
    }
    pending_uids: set[int] = set()
    info["num_aborted_reqs"] = 0

    def serve() -> None:
        while True:
            message = incoming.get()
            for request in (
                message.batch if isinstance(message, BatchTokenizedGenerateReqInput) else [message]
            ):
                if isinstance(request, SchedulerControlReq):
                    if isinstance(request, ResetMetricsReq) and pending_uids:
                        continue
                    replies.put(
                        SchedulerControlOutput(request.uid, {**info, "pending": len(pending_uids)})
                    )
                elif isinstance(request, AbortReq):
                    assert request.uid in pending_uids
                    pending_uids.remove(request.uid)
                    info["num_aborted_reqs"] += 1
                    tokens.put(TokenIDOutput(request.uid, 0, True, skip_token=True))
                elif isinstance(request, TokenizedGenerateReqInput):
                    token = int(request.input_ids[0])
                    if token == 63:
                        raise RuntimeError("Injected scheduler failure: " + "x" * 100_000)
                    if request.sampling_params.max_tokens == 100:
                        pending_uids.add(request.uid)
                        continue
                    for index in range(request.sampling_params.max_tokens):
                        tokens.put(
                            TokenIDOutput(
                                request.uid, token, index + 1 == request.sampling_params.max_tokens
                            )
                        )

    from loopsgl.managers import scheduler

    scheduler.create_scheduler = lambda config: SimpleNamespace(
        sync_all_ranks=lambda: None, get_server_info=lambda: info, event_loop=serve
    )
    scheduler.run_scheduler_process(config, ack_queue, silent_output=silent_output)


class _CPUEngine(Engine):
    run_scheduler_process_func = staticmethod(_cpu_scheduler)


def test_runtime_lifecycle(
    tiny_checkpoint: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route concurrent replies, cancel requests, propagate worker failure, and reap only owned workers."""
    cuda_initialized = torch.cuda.is_initialized()
    monkeypatch.setattr("loopsgl.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT", 0.01)
    engine = _CPUEngine(str(tiny_checkpoint("ouro")), startup_timeout=30)
    context = mp.get_context("spawn")
    release = context.Event()
    unrelated = context.Process(target=release.wait)
    unrelated.start()

    async def check() -> None:
        params = {"max_new_tokens": 3, "ignore_eos": True}
        for invalid in (
            {"temperature": float("nan")},
            {"temperature": -1},
            {"top_p": 0},
            {"top_p": 1.5},
            {"top_k": 0},
            {"top_k": True},
            {"max_tokens": 1.5},
        ):
            with pytest.raises(ValueError):
                await engine.async_generate(input_ids=[8], sampling_params=invalid)
        assert not engine.tokenizer_manager.rid_to_state
        outputs = await asyncio.wait_for(
            asyncio.gather(
                *(
                    engine.async_generate(input_ids=[token], sampling_params=params)
                    for token in [8, 9, 10]
                )
            ),
            timeout=10,
        )
        assert [output["token_ids"] for output in outputs] == [[8] * 3, [9] * 3, [10] * 3]
        pending = asyncio.create_task(
            engine.async_generate(input_ids=[11], sampling_params={"max_new_tokens": 100})
        )
        # A control reply fences dispatch of the preceding generation request on the same socket.
        from loopsgl.managers.io_struct import GetInternalStateReq

        await asyncio.sleep(0)
        await asyncio.wait_for(engine.tokenizer_manager.scheduler_request(GetInternalStateReq), 10)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        followup = await asyncio.wait_for(
            engine.async_generate(input_ids=[12], sampling_params=params), 10
        )
        assert followup["token_ids"] == [12] * 3
        state = await asyncio.wait_for(
            engine.tokenizer_manager.scheduler_request(GetInternalStateReq), 10
        )
        assert state["num_aborted_reqs"] == 1 and state["pending"] == 0

        async def is_disconnected() -> bool:
            return True

        uid = engine.tokenizer_manager.new_request()
        await engine.tokenizer_manager.send_request(
            GenerateReqInput(
                uid=uid, text=None, input_ids=[11], sampling_params=SamplingParams(max_tokens=100)
            )
        )
        response = engine.tokenizer_manager._wait_one_response(
            uid, request=SimpleNamespace(is_disconnected=is_disconnected)
        )
        with pytest.raises(ValueError, match="disconnected"):
            await asyncio.wait_for(response.__anext__(), 1)
        state = await asyncio.wait_for(
            engine.tokenizer_manager.scheduler_request(GetInternalStateReq), 10
        )
        assert state["num_aborted_reqs"] == 2 and state["pending"] == 0
        with pytest.raises(RuntimeError, match="worker failed"):
            await asyncio.wait_for(
                engine.async_generate(input_ids=[63], sampling_params=params), 10
            )
        assert not engine.tokenizer_manager.rid_to_state
        await engine.tokenizer_manager.close()

    async def check_shutdown() -> None:
        from loopsgl.managers.io_struct import GetInternalStateReq

        pending = asyncio.create_task(
            engine.async_generate(input_ids=[11], sampling_params={"max_new_tokens": 100})
        )
        await asyncio.sleep(0)
        control = asyncio.create_task(engine.tokenizer_manager.scheduler_request(ResetMetricsReq))
        await asyncio.sleep(0)
        await asyncio.wait_for(engine.tokenizer_manager.scheduler_request(GetInternalStateReq), 10)
        engine.shutdown()
        for task in (pending, control):
            with pytest.raises(RuntimeError, match="shut down"):
                await asyncio.wait_for(task, 1)
        assert not engine.tokenizer_manager.rid_to_state
        assert not engine.tokenizer_manager.control_futures
        await engine.tokenizer_manager.close()

    try:
        engine._run(check())
        engine.shutdown()
        assert not any(process.is_alive() for process in engine._processes)
        engine = _CPUEngine(str(tiny_checkpoint("ouro")), startup_timeout=30)
        engine._run(check_shutdown())
    finally:
        try:
            engine.shutdown()
            engine.loop.close()
            assert not any(process.is_alive() for process in engine._processes)
            assert unrelated.is_alive()
        finally:
            release.set()
            unrelated.join(timeout=5)
            if unrelated.is_alive():
                unrelated.kill()
                unrelated.join(timeout=5)
    assert torch.cuda.is_initialized() == cuda_initialized
    for address in (
        engine.server_args.zmq_backend_addr,
        engine.server_args.zmq_detokenizer_addr,
        engine.server_args.zmq_frontend_addr,
    ):
        assert not Path(address.removeprefix("ipc://")).exists()
