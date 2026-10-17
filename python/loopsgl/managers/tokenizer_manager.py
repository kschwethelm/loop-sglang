from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

import torch

from loopsgl.managers.io_struct import (
    AbortReq,
    BaseReq,
    BatchStrOutput,
    BatchTokenizedGenerateReqInput,
    GenerateReqInput,
    SchedulerControlOutput,
    SchedulerControlReq,
    StrOutput,
    TokenizedGenerateReqInput,
)
from loopsgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger, load_tokenizer

if TYPE_CHECKING:
    from fastapi import Request

    from loopsgl.core import SamplingParams
    from loopsgl.server_args import ServerArgs


logger = init_logger(__name__)
_REQUEST_STATE_WAIT_TIMEOUT = 4.0


@dataclass
class ReqState:
    """Track responses and wake the frontend waiting for one request."""

    out_list: list[StrOutput] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    error: Exception | None = None
    output_prefix: str = ""


class TokenizerManager:
    """Tokenize frontend requests, dispatch them, and route worker replies to callers."""

    def __init__(self, server_args: ServerArgs) -> None:
        self.server_args = server_args
        self.tokenizer = load_tokenizer(server_args.model_path)
        self.max_req_input_len = server_args.max_seq_len
        self.max_loop_depth = server_args.model_config.loop_params.loop_steps or 1
        self.min_exit_depth = 1 if server_args.cdb_config.disable_lookahead_exit else 2
        self.send_to_scheduler = ZmqAsyncPushQueue(
            server_args.zmq_backend_addr, create=False, encoder=BaseReq.encoder
        )
        self.recv_from_detokenizer = ZmqAsyncPullQueue(
            server_args.zmq_frontend_addr, create=True, decoder=BaseReq.decoder
        )
        self.rid_to_state: dict[int, ReqState] = {}
        self.control_futures: dict[int, asyncio.Future[SchedulerControlOutput]] = {}
        self.uid_counter = 0
        self.handle_task: asyncio.Task | None = None
        self.watchdog_task: asyncio.Task | None = None
        self.check_workers: Callable[[], None] | None = None
        self.closed = False
        self.failure: Exception | None = None

    def new_request(self) -> int:
        """Allocate response state before a request can reach a worker."""
        self._raise_if_unavailable()
        uid = self.uid_counter
        self.uid_counter += 1
        self.rid_to_state[uid] = ReqState()
        return uid

    def _raise_if_unavailable(self) -> None:
        if self.closed:
            raise RuntimeError("The inference engine is shut down.")
        if self.failure is not None:
            raise RuntimeError("An inference worker failed.") from self.failure
        if self.check_workers is not None:
            self.check_workers()

    def auto_create_handle_loop(self) -> None:
        """Start response dispatch and worker monitoring on the caller's event loop."""
        self._raise_if_unavailable()
        if (
            self.handle_task is not None
            and self.handle_task.get_loop() is not asyncio.get_running_loop()
        ):
            raise RuntimeError("Use a single event loop for each inference engine.")
        if self.handle_task is None:
            self.handle_task = asyncio.create_task(self.handle_loop())
            self.watchdog_task = asyncio.create_task(self._watch_workers())

    def _fail_requests(self, error: Exception) -> None:
        self.failure = error
        for state in self.rid_to_state.values():
            state.error = error
            state.event.set()
        for future in self.control_futures.values():
            if not future.done():
                future.set_exception(error)

    async def _watch_workers(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.5)
                self._raise_if_unavailable()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_requests(exc)

    async def handle_loop(self) -> None:
        """Dispatch generation and control replies without losing concurrent requests."""
        try:
            while True:
                message = await self.recv_from_detokenizer.get()
                messages = message.data if isinstance(message, BatchStrOutput) else [message]
                for reply in messages:
                    if isinstance(reply, SchedulerControlOutput):
                        future = self.control_futures.get(reply.uid)
                        if future is not None and not future.done():
                            future.set_result(reply)
                    elif isinstance(reply, StrOutput):
                        state = self.rid_to_state.get(reply.uid)
                        if state is not None:
                            state.out_list.append(reply)
                            state.event.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_requests(exc)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Render chat messages with request kwargs overriding server defaults."""
        template_kwargs = dict(self.server_args.default_chat_template_kwargs or {})
        if chat_template_kwargs is not None:
            template_kwargs.update(chat_template_kwargs)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

    async def _tokenize_one_request(self, obj: GenerateReqInput) -> TokenizedGenerateReqInput:
        """Prepare supplied token IDs or tokenize text before scheduler dispatch."""
        if (obj.text is None) == (obj.input_ids is None):
            raise ValueError("Specify exactly one of text and input_ids.")
        token_ids = obj.input_ids
        output_prefix = ""
        if token_ids is None:
            prompt = obj.text
            if isinstance(prompt, list):
                prompt = self.apply_chat_template(
                    prompt, chat_template_kwargs=obj.chat_template_kwargs
                )
                assert isinstance(prompt, str)
                _, opening_tag, suffix = prompt.rpartition("<think>")
                if opening_tag and not suffix.strip():
                    output_prefix = opening_tag + suffix
            token_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(token_ids, dtype=torch.int32, device="cpu")
        self._validate_one_request(obj, input_ids)
        state = self.rid_to_state.get(obj.uid)
        if state is not None:
            state.output_prefix = output_prefix
        return TokenizedGenerateReqInput(
            uid=obj.uid, input_ids=input_ids, sampling_params=obj.sampling_params
        )

    def _validate_one_request(self, obj: GenerateReqInput, input_ids: torch.Tensor) -> None:
        """Validate input lengths, sampling parameters, and exit depths before admission."""
        params = obj.sampling_params
        params.normalize()
        params.verify()
        if len(input_ids) == 0:
            raise ValueError("Prompts and generation budgets must be nonempty.")
        if len(input_ids) >= self.max_req_input_len:
            raise ValueError("Input sequence leaves no room for generation.")
        depths = params.replay_exit_depths
        if depths is not None:
            if len(input_ids) + params.max_tokens > self.max_req_input_len:
                raise ValueError("Recorded exit replay exceeds the context length.")
            if len(depths) != params.max_tokens - 1 or any(
                isinstance(depth, bool)
                or not isinstance(depth, int)
                or not self.min_exit_depth <= depth <= self.max_loop_depth
                for depth in depths
            ):
                raise ValueError(
                    "Recorded exit depths must match the output budget and configured depth bounds."
                )

    async def _send_one_request(self, obj: TokenizedGenerateReqInput) -> None:
        """Send a prepared generation request directly to the scheduler."""
        await self.send_to_scheduler.put(obj)

    async def send_request(self, obj: GenerateReqInput) -> None:
        """Prepare and submit one HTTP request with response state already allocated."""
        self.auto_create_handle_loop()
        dispatch_started = False
        try:
            tokenized_obj = await self._tokenize_one_request(obj)
            dispatch_started = True
            await self._send_one_request(tokenized_obj)
        except BaseException:
            self.rid_to_state.pop(obj.uid, None)
            if dispatch_started and not self.closed:
                await asyncio.gather(self.abort_request(obj.uid), return_exceptions=True)
            raise

    async def _wait_one_response(
        self, uid: int, request: Request | None = None
    ) -> AsyncIterator[StrOutput]:
        """Yield replies and abort HTTP work when its waiting client disconnects."""
        state = self.rid_to_state[uid]
        prefix_sent = False
        try:
            while True:
                if request is None:
                    await state.event.wait()
                else:
                    try:
                        await asyncio.wait_for(
                            state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        pass
                state.event.clear()
                if state.error is not None:
                    message = (
                        "The inference engine is shut down."
                        if self.closed
                        else "An inference worker failed."
                    )
                    raise RuntimeError(message) from state.error
                if request is not None and await request.is_disconnected():
                    raise ValueError(f"Client disconnected. Abort request {uid}.")
                pending, state.out_list = state.out_list, []
                for reply in pending:
                    if reply.error is not None:
                        raise ValueError(reply.error)
                    if state.output_prefix and not prefix_sent:
                        yield StrOutput(
                            uid=uid,
                            incremental_output=state.output_prefix,
                            finished=False,
                        )
                        prefix_sent = True
                    if reply.finished and reply.text is not None:
                        reply.text = state.output_prefix + reply.text
                    if reply.finished:
                        self.rid_to_state.pop(uid, None)
                    yield reply
                    if reply.finished:
                        return
        except BaseException:
            if request is not None and uid in self.rid_to_state and not self.closed:
                await asyncio.gather(self.abort_request(uid), return_exceptions=True)
            raise
        finally:
            self.rid_to_state.pop(uid, None)

    async def _send_batch_request(self, requests: list[TokenizedGenerateReqInput]) -> None:
        """Send prepared requests together to preserve offline batch admission."""
        await self.send_to_scheduler.put(BatchTokenizedGenerateReqInput(batch=requests))

    async def _handle_batch_request(
        self, prompts: list[str] | list[list[int]], sampling_params: list[SamplingParams]
    ) -> list[dict[str, Any]]:
        uids = [self.new_request() for _ in prompts]
        requests = [
            GenerateReqInput(
                uid=uid,
                text=prompt if isinstance(prompt, str) else None,
                input_ids=None if isinstance(prompt, str) else prompt,
                sampling_params=params,
            )
            for uid, prompt, params in zip(uids, prompts, sampling_params, strict=True)
        ]
        completed_uids: set[int] = set()

        async def collect(uid: int) -> dict[str, Any]:
            text = []
            async for reply in self._wait_one_response(uid):
                text.append(reply.incremental_output)
                if reply.finished:
                    completed_uids.add(uid)
                    return {
                        "text": reply.text if reply.text is not None else "".join(text),
                        "token_ids": reply.token_ids or [],
                    }
            raise RuntimeError("Request ended without a completed response.")

        tasks = []
        dispatch_started = False
        try:
            tokenized_requests = [await self._tokenize_one_request(obj) for obj in requests]
            dispatch_started = True
            await self._send_batch_request(tokenized_requests)
            tasks = [asyncio.create_task(collect(uid)) for uid in uids]
            return await asyncio.gather(*tasks)
        except BaseException:
            unfinished = (
                [uid for uid in uids if uid not in completed_uids] if dispatch_started else []
            )
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if not self.closed:
                await asyncio.gather(
                    *(self.abort_request(uid) for uid in unfinished), return_exceptions=True
                )
            raise
        finally:
            for uid in uids:
                self.rid_to_state.pop(uid, None)

    async def generate_request(
        self, prompts: list[str] | list[list[int]], sampling_params: list[SamplingParams]
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Submit a Python batch and yield its completed outputs in input order."""
        self.auto_create_handle_loop()
        if not prompts:
            yield []
            return
        yield await self._handle_batch_request(prompts, sampling_params)

    async def abort_request(self, uid: int) -> None:
        self.rid_to_state.pop(uid, None)
        logger.warning("Aborting request for user %s", uid)
        self.auto_create_handle_loop()
        await self.send_to_scheduler.put(AbortReq(uid=uid))

    async def scheduler_request(self, request_type: type[SchedulerControlReq]) -> dict[str, Any]:
        """Wait for a TP-wide scheduler control operation to complete at idle."""
        self.auto_create_handle_loop()
        uid = self.uid_counter
        self.uid_counter += 1
        future = asyncio.get_running_loop().create_future()
        self.control_futures[uid] = future
        try:
            await self.send_to_scheduler.put(request_type(uid=uid))
            reply = await future
            if reply.error is not None:
                raise RuntimeError(reply.error)
            return reply.result
        finally:
            self.control_futures.pop(uid, None)

    def shutdown(self) -> None:
        """Stop frontend tasks and close the IPC sockets owned by this manager."""
        if self.closed:
            return
        self.closed = True
        self._fail_requests(RuntimeError("The inference engine is shut down."))
        for task in (self.handle_task, self.watchdog_task):
            if task is not None:
                task.cancel()
        for channel in (self.send_to_scheduler, self.recv_from_detokenizer):
            channel.stop()

    async def close(self) -> None:
        self.shutdown()
        tasks = [task for task in (self.handle_task, self.watchdog_task) if task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
