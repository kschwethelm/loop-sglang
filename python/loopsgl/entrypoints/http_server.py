from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, AsyncIterator, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import WordCompleter
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from loopsgl.core import SamplingParams
from loopsgl.env import ENV
from loopsgl.managers.io_struct import (
    FlushCacheReqInput,
    GenerateReqInput,
    GetInternalStateReq,
    StrOutput,
)
from loopsgl.managers.tokenizer_manager import TokenizerManager
from loopsgl.server_args import ServerArgs
from loopsgl.stopping import normalize_stops
from loopsgl.utils import init_logger

from .engine import Engine
from .shell import ShellOutputFormatter, loop_step_legend

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> TokenizerManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


class StopRequest(BaseModel):
    stop: str | List[str] | None = None
    no_stop_trim: bool = False

    @model_validator(mode="before")
    @classmethod
    def reject_regex_stops(cls, value: Any) -> Any:
        if isinstance(value, dict) and "stop_regex" in value:
            raise ValueError("stop_regex is unsupported; use literal stop strings.")
        return value

    @field_validator("stop")
    @classmethod
    def validate_stops(cls, value: str | List[str] | None) -> List[str]:
        return normalize_stops(value)


class GenerateRequest(BaseModel):
    """Supported single-request SGLang native generation input."""

    model_config = ConfigDict(extra="forbid")
    text: StrictStr | None = None
    input_ids: list[StrictInt] | None = None
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    stream: StrictBool = False

    @model_validator(mode="after")
    def validate_input(self) -> GenerateRequest:
        if (self.text is None) == (self.input_ids is None):
            raise ValueError("Specify exactly one of text and input_ids.")
        if self.text == "":
            raise ValueError("Text prompts must be nonempty.")
        return self


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(StopRequest):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "loop-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


async def stream_generate(
    state: TokenizerManager, uid: int, request: Request
) -> AsyncIterator[bytes]:
    prompt_tokens = state.rid_to_state[uid].prompt_tokens
    text = ""
    responses = state._wait_one_response(uid, request=request)
    try:
        async for ack in responses:
            text += ack.incremental_output
            if ack.finished and ack.text is not None:
                text = ack.text
            result = {
                "text": text,
                "meta_info": {
                    "id": str(uid),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": ack.completion_tokens,
                    "finish_reason": ack.finish_reason,
                },
            }
            yield ("data: " + json.dumps(result, ensure_ascii=False) + "\n\n").encode()
        yield b"data: [DONE]\n\n"
    except Exception as exc:
        yield ("data: " + json.dumps({"error": {"message": str(exc)}}) + "\n\n").encode()
    finally:
        await responses.aclose()


async def stream_chat_completions(state: TokenizerManager, uid: int) -> AsyncIterator[bytes]:
    first_chunk = True
    async for ack in state._wait_one_response(uid):
        delta = {}
        if first_chunk:
            delta["role"] = "assistant"
            first_chunk = False
        if ack.incremental_output:
            delta["content"] = ack.incremental_output

        chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()

        if ack.finished:
            break

    # send final finish_reason
    end_chunk = {
        "id": f"cmpl-{uid}",
        "object": "text_completion.chunk",
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end_chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"
    logger.debug("Finished streaming response for user %s", uid)


async def stream_with_cancellation(
    state: TokenizerManager, generator: AsyncIterator[bytes], request: Request, uid: int
) -> AsyncIterator[bytes]:
    try:
        async for chunk in generator:
            # detect if the client has disconnected
            if await request.is_disconnected():
                logger.info("Client disconnected for user %s", uid)
                raise asyncio.CancelledError
            yield chunk
    except asyncio.CancelledError:
        asyncio.create_task(state.abort_request(uid))
        raise


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        await _GLOBAL_STATE.close()


app = FastAPI(title="LoopSGL API Server", version="0.0.1", lifespan=lifespan)


@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Return frontend request validation errors without failing a worker."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.post("/generate", response_model=None)
async def generate(req: GenerateRequest, request: Request) -> JSONResponse | StreamingResponse:
    state = get_global_state()
    if req.input_ids is not None and any(
        token < 0 or token >= state.server_args.model_config.vocab_size for token in req.input_ids
    ):
        raise ValueError("input_ids must contain token IDs within the model vocabulary.")
    try:
        params = SamplingParams(**req.sampling_params)
    except TypeError as exc:
        raise ValueError(f"Unsupported sampling parameters: {exc}") from exc
    uid = state.new_request()
    await state.send_request(
        GenerateReqInput(uid=uid, text=req.text, input_ids=req.input_ids, sampling_params=params)
    )
    if req.stream:
        return StreamingResponse(
            stream_with_cancellation(state, stream_generate(state, uid, request), request, uid),
            media_type="text/event-stream",
        )

    prompt_tokens = state.rid_to_state[uid].prompt_tokens
    responses = state._wait_one_response(uid, request=request)
    try:
        async for ack in responses:
            if ack.finished:
                return JSONResponse(
                    {
                        "text": ack.text or "",
                        "meta_info": {
                            "id": str(uid),
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": ack.completion_tokens,
                            "finish_reason": ack.finish_reason,
                        },
                    }
                )
    finally:
        await responses.aclose()
    raise RuntimeError("Generation ended without a completed response.")


@app.get("/health")
async def health() -> JSONResponse:
    state = get_global_state()
    try:
        state._raise_if_unavailable()
    except RuntimeError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return JSONResponse({"status": "ok"})


@app.get("/server_info")
async def server_info() -> JSONResponse:
    state = get_global_state()
    info = await state.scheduler_request(GetInternalStateReq)
    server_args = {
        key: value for key, value in asdict(state.server_args).items() if not key.startswith("_")
    }
    server_args["dtype"] = str(state.server_args.dtype)
    return JSONResponse({**info, "server_args": server_args})


@app.post("/flush_cache")
async def flush_cache() -> JSONResponse:
    info = await get_global_state().scheduler_request(FlushCacheReqInput)
    return JSONResponse(info)


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_request()
    await state.send_request(
        GenerateReqInput(
            uid=uid,
            text=prompt,
            chat_template_kwargs=req.chat_template_kwargs,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                stop=req.stop,
                no_stop_trim=req.no_stop_trim,
            ),
        )
    )

    if req.stream:
        return StreamingResponse(
            stream_with_cancellation(state, stream_chat_completions(state, uid), request, uid),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""
    async for ack in state._wait_one_response(uid, request=request):
        full_content += ack.incremental_output
        if ack.finished:
            break

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(
        data=[ModelCard(id=state.server_args.model_path, root=state.server_args.model_path)]
    )


async def shell_completion(req: OpenAICompletionRequest) -> AsyncIterator[StrOutput]:
    """Stream text and loop counts directly to the interactive terminal."""
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_request()
    await state.send_request(
        GenerateReqInput(
            uid=uid,
            text=prompt,
            chat_template_kwargs=req.chat_template_kwargs,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                stop=req.stop,
                no_stop_trim=req.no_stop_trim,
                return_loop_steps=True,
            ),
        )
    )

    finished = False
    try:
        async for reply in state._wait_one_response(uid):
            finished = reply.finished
            yield reply
    finally:
        if not finished and not state.closed:
            await state.abort_request(uid)


async def shell() -> None:
    commands = ["/exit", "/clear", "/colors"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)
    colors_enabled = "NO_COLOR" not in os.environ
    max_steps = get_global_state().max_loop_depth

    try:
        if colors_enabled:
            print_formatted_text(loop_step_legend(max_steps))
            print("Use /colors to toggle colors.")
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/clear":
                    history = []
                    continue
                if cmd == "/colors":
                    colors_enabled = not colors_enabled
                    if colors_enabled:
                        print_formatted_text(loop_step_legend(max_steps))
                    else:
                        print("Loop-step colors disabled.")
                    continue
                print(f"Unknown command: {cmd}. Available commands: {', '.join(commands)}")
                continue
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            formatter = ShellOutputFormatter(max_steps)
            async for reply in shell_completion(req):
                cur_msg += reply.incremental_output
                if colors_enabled:
                    print_formatted_text(
                        formatter.feed(
                            reply.incremental_output,
                            reply.incremental_loop_steps,
                            finished=reply.finished,
                        ),
                        end="",
                        flush=True,
                    )
                else:
                    print(reply.incremental_output, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        await get_global_state().close()


def _setup_and_run_http_server(
    server_args: ServerArgs, tokenizer_manager: TokenizerManager, run_shell: bool
) -> None:
    """
    Run the HTTP frontend or interactive shell with the shared tokenizer manager.

    Args:
        server_args: Server configuration (host/port, ZMQ IPC addresses, etc).
        tokenizer_manager: Shared runtime request manager connected to ready workers.
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not server_args.use_dummy_weight, "Shell mode does not support dummy weights."

    host = server_args.server_host
    port = server_args.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = tokenizer_manager

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())


def launch_server(server_args: ServerArgs, *, run_shell: bool = False) -> None:
    """Start inference workers and serve HTTP requests or the interactive shell."""
    engine = Engine(server_args=server_args)
    try:
        _setup_and_run_http_server(server_args, engine.tokenizer_manager, run_shell=run_shell)
    finally:
        engine.shutdown()
