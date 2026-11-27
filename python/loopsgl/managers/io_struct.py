from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from loopsgl.core import SamplingParams
from loopsgl.utils.serialization import deserialize_type, serialize_type


@dataclass
class BaseReq:
    """Serialize request and response payloads exchanged between serving processes."""

    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(data: Dict) -> BaseReq:
        return deserialize_type(globals(), data)


@dataclass
class BaseBatchReq(BaseReq):
    """Share serialization across batches of serving requests or responses."""


@dataclass
class GenerateReqInput:
    """Frontend generation input prepared locally before scheduler dispatch."""

    uid: int
    text: str | list[dict[str, str]] | None
    sampling_params: SamplingParams
    input_ids: list[int] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    return_logprob: bool = False
    logprob_start_len: int = -1
    top_logprobs_num: int = 0


@dataclass
class TokenizedGenerateReqInput(BaseReq):
    """Carry tokenized input and sampling settings to the scheduler."""

    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    return_logprob: bool = False
    logprob_start_len: int = -1
    top_logprobs_num: int = 0


@dataclass
class BatchTokenizedGenerateReqInput(BaseBatchReq):
    """Submit a group of tokenized requests together for scheduler admission."""

    batch: List[TokenizedGenerateReqInput]


@dataclass
class AbortReq(BaseReq):
    """Cancel the generation request identified by its frontend UID."""

    uid: int


@dataclass
class ExitReq(BaseReq):
    """Stop the scheduler event loop."""


@dataclass
class SchedulerControlReq(BaseReq):
    """Identify a scheduler control operation applied when generation is idle."""

    uid: int


@dataclass
class GetInternalStateReq(SchedulerControlReq):
    """Request scheduler configuration and generation statistics."""


@dataclass
class FlushCacheReqInput(SchedulerControlReq):
    """Clear reusable prefixes from the scheduler cache."""


@dataclass
class ResetMetricsReq(SchedulerControlReq):
    """Reset scheduler statistics before a measured generation call."""


@dataclass
class FlushProfileReq(SchedulerControlReq):
    """Flush CUDA profiler buffers after all pending generation has drained."""


@dataclass
class SchedulerControlOutput(BaseReq):
    """Return scheduler state or an error for one control operation."""

    uid: int
    result: Dict[str, Any]
    error: str | None = None


@dataclass(kw_only=True)
class LogprobOutput:
    """Carry compact per-request score updates independently of displayed text."""

    input_token_logprobs_val: list[float | None] | None = None
    input_token_logprobs_idx: list[int] | None = None
    output_token_logprobs_val: list[float] | None = None
    output_token_logprobs_idx: list[int] | None = None
    input_top_logprobs_val: list[list[float] | None] | None = None
    input_top_logprobs_idx: list[list[int] | None] | None = None
    output_top_logprobs_val: list[list[float]] | None = None
    output_top_logprobs_idx: list[list[int]] | None = None


@dataclass
class TokenIDOutput(BaseReq):
    """Carry one request's generated token and stopping state to the detokenizer."""

    uid: int
    next_token: int
    finished: bool
    skip_token: bool = False
    stop_text: str | None = None
    stop_buffer_length: int = 0
    skip_special_tokens: bool = False
    loop_steps: int | None = None  # core steps used to produce next_token
    completion_tokens: int | None = None
    finish_reason: dict[str, Any] | None = None


@dataclass(kw_only=True)
class TokenIDOutputWithDepth(TokenIDOutput):
    decode_loop_steps: int | None = None


@dataclass(kw_only=True)
class ScoredTokenIDOutput(TokenIDOutputWithDepth):
    logprobs: LogprobOutput


@dataclass
class BatchTokenIDOutput(BaseBatchReq):
    """Carry scheduler token outputs for several requests to the detokenizer."""

    data: List[TokenIDOutput]


@dataclass
class StrOutput(BaseReq):
    """Return incremental text and optional completed output to the frontend."""

    uid: int
    incremental_output: str
    finished: bool
    token_ids: List[int] | None = None
    text: str | None = None
    error: str | None = None
    incremental_loop_steps: List[int] | None = None  # one count per emitted character
    completion_tokens: int = 0
    finish_reason: dict[str, Any] | None = None


@dataclass(kw_only=True)
class StrOutputWithDepth(StrOutput):
    decode_token_loop_steps: tuple[int, int] | None = None


@dataclass(kw_only=True)
class ScoredStrOutput(StrOutputWithDepth):
    logprobs: LogprobOutput


@dataclass
class BatchStrOutput(BaseBatchReq):
    """Carry detokenized outputs for several requests to the frontend."""

    data: List[StrOutput]
