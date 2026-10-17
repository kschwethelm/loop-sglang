from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import queue
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from loopsgl.core import SamplingParams
from loopsgl.distributed import DistributedInfo
from loopsgl.managers.detokenizer_manager import run_detokenizer_process
from loopsgl.managers.io_struct import (
    FlushCacheReqInput,
    FlushProfileReq,
    GetInternalStateReq,
    ResetMetricsReq,
)
from loopsgl.managers.scheduler import run_scheduler_process
from loopsgl.managers.tokenizer_manager import TokenizerManager
from loopsgl.server_args import ServerArgs


class Engine:
    """Tokenizer request management and the scheduler/detokenizer subprocesses."""

    run_scheduler_process_func = staticmethod(run_scheduler_process)
    run_detokenizer_process_func = staticmethod(run_detokenizer_process)

    def __init__(
        self,
        model_path: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        *,
        tp_size: int = 1,
        server_args: ServerArgs | None = None,
        startup_timeout: float = 600.0,
        **kwargs: Any,
    ) -> None:
        if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 1:
            raise ValueError("tp_size must be positive.")
        if server_args is None:
            if model_path is None:
                raise ValueError("model_path is required.")
            server_args = ServerArgs(
                model_path=model_path, dtype=dtype, tp_info=DistributedInfo(0, tp_size), **kwargs
            )
        elif model_path is not None or kwargs:
            raise ValueError("Pass server_args or model options, not both.")
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive.")
        self.server_args = server_args
        self.loop = asyncio.new_event_loop()
        self.closed = False
        try:
            self.tokenizer_manager, self._server_info, self._processes = self._launch_subprocesses(
                server_args, startup_timeout
            )
        except BaseException:
            self.loop.close()
            raise
        self.tokenizer_manager.check_workers = self._check_workers
        atexit.register(self.shutdown)

    @classmethod
    def _launch_subprocesses(
        cls, server_args: ServerArgs, startup_timeout: float
    ) -> tuple[TokenizerManager, dict[str, Any], list[mp.Process]]:
        """Bind frontend IPC, launch TP workers, and wait for every process to be ready."""
        context = mp.get_context("spawn")
        manager = TokenizerManager(server_args)
        ack_queue = context.Queue()
        processes: list[mp.Process] = []
        deadline = time.monotonic() + startup_timeout

        def wait_ready(count: int) -> list[Any]:
            replies = []
            while len(replies) < count:
                for process in processes:
                    if process.exitcode is not None:
                        raise RuntimeError(
                            f"Worker {process.name} exited with code {process.exitcode} during startup."
                        )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Inference workers did not become ready before startup_timeout."
                    )
                try:
                    reply = ack_queue.get(timeout=min(remaining, 0.2))
                except queue.Empty:
                    continue
                if isinstance(reply, dict) and "error" in reply:
                    raise RuntimeError(reply["error"])
                replies.append(reply)
            return replies

        try:
            for rank in range(server_args.tp_info.size):
                process = context.Process(
                    target=cls.run_scheduler_process_func,
                    args=(
                        replace(
                            server_args, tp_info=DistributedInfo(rank, server_args.tp_info.size)
                        ),
                        ack_queue,
                    ),
                    kwargs={"silent_output": server_args.silent_output},
                    name=f"loopsgl-TP{rank}-scheduler",
                )
                process.start()
                processes.append(process)
            scheduler_infos = wait_ready(server_args.tp_info.size)
            info = min(scheduler_infos, key=lambda entry: entry["max_seq_len"])
            manager.max_req_input_len = info["max_seq_len"]
            manager.max_loop_depth = info["loop_params"]["loop_steps"] or 1
            process = context.Process(
                target=cls.run_detokenizer_process_func,
                args=(server_args, ack_queue),
                name="loopsgl-detokenizer",
            )
            process.start()
            processes.append(process)
            wait_ready(1)
            return manager, info, processes
        except BaseException:
            manager.shutdown()
            cls._shutdown_processes(processes)
            cls._unlink_ipc(server_args)
            raise
        finally:
            ack_queue.close()
            ack_queue.join_thread()

    @staticmethod
    def _shutdown_processes(processes: list[mp.Process]) -> None:
        for process in processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 3
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=3)

    def _check_workers(self) -> None:
        for process in self._processes:
            if process.exitcode is not None:
                raise RuntimeError(f"Worker {process.name} exited with code {process.exitcode}.")

    @property
    def tokenizer(self) -> Any:
        return self.tokenizer_manager.tokenizer

    @property
    def max_seq_len(self) -> int:
        return self._server_info["max_seq_len"]

    @property
    def num_retracted_reqs(self) -> int:
        return self._server_info["num_retracted_reqs"]

    @property
    def batch_size_metrics(self) -> dict[str, float]:
        return self._server_info["batch_size_metrics"]

    def _run(self, coroutine: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coroutine.close()
            raise RuntimeError("Use async_generate from an asyncio event loop.")
        if self.closed:
            coroutine.close()
            raise RuntimeError("The inference engine is shut down.")
        return self.loop.run_until_complete(coroutine)

    def get_server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    def get_internal_state(self) -> dict[str, Any]:
        self._server_info = self._run(self.tokenizer_manager.scheduler_request(GetInternalStateReq))
        return self.get_server_info()

    def flush_cache(self) -> None:
        self._server_info = self._run(self.tokenizer_manager.scheduler_request(FlushCacheReqInput))

    def reset_metrics(self) -> None:
        self._server_info = self._run(self.tokenizer_manager.scheduler_request(ResetMetricsReq))

    def flush_profile(self) -> None:
        """Flush CUDA profiler buffers in every idle scheduler worker before process termination."""
        self._run(self.tokenizer_manager.scheduler_request(FlushProfileReq))

    async def async_generate(
        self,
        prompt: str | list[str] | None = None,
        sampling_params: SamplingParams | dict | list[SamplingParams | dict] | None = None,
        *,
        input_ids: list[int] | list[list[int]] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        self.tokenizer_manager._raise_if_unavailable()
        if (prompt is None) == (input_ids is None):
            raise ValueError("Specify exactly one of prompt and input_ids.")
        if input_ids is not None:
            single = not input_ids or isinstance(input_ids[0], int)
            prompts = [input_ids] if single else input_ids
        else:
            single = isinstance(prompt, str)
            prompts = [prompt] if single else prompt
        params = (
            sampling_params
            if isinstance(sampling_params, list)
            else [sampling_params] * len(prompts)
        )
        if len(params) != len(prompts):
            raise ValueError("sampling_params must match the number of prompts.")
        normalized = []
        for value in params:
            if isinstance(value, SamplingParams):
                normalized.append(value)
            else:
                options = dict(value or {})
                if "max_new_tokens" in options and "max_tokens" in options:
                    raise ValueError("Specify only one of max_new_tokens and max_tokens.")
                if "max_new_tokens" in options:
                    options["max_tokens"] = options.pop("max_new_tokens")
                normalized.append(SamplingParams(**options))
        generator = self.tokenizer_manager.generate_request(prompts, normalized)
        outputs = await generator.__anext__()
        return outputs[0] if single else outputs

    def generate(
        self,
        prompt: str | list[str] | None = None,
        sampling_params: SamplingParams | dict | list[SamplingParams | dict] | None = None,
        *,
        input_ids: list[int] | list[list[int]] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        self.reset_metrics()
        try:
            return self._run(self.async_generate(prompt, sampling_params, input_ids=input_ids))
        finally:
            if self.tokenizer_manager.failure is None:
                self.get_internal_state()

    def shutdown(self) -> None:
        """Reap only this engine's workers and release its frontend IPC resources."""
        if self.closed:
            return
        self.closed = True
        self.tokenizer_manager.shutdown()
        if not self.loop.is_closed() and not self.loop.is_running():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                self.loop.run_until_complete(asyncio.sleep(0))
            self.loop.close()
        self._shutdown_processes(self._processes)
        self._unlink_ipc(self.server_args)
        atexit.unregister(self.shutdown)

    @staticmethod
    def _unlink_ipc(server_args: ServerArgs) -> None:
        for address in (
            server_args.zmq_backend_addr,
            server_args.zmq_detokenizer_addr,
            server_args.zmq_frontend_addr,
        ):
            Path(address.removeprefix("ipc://")).unlink(missing_ok=True)
