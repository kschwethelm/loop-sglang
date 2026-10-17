from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List

from loopsgl.managers.io_struct import TokenIDOutput
from loopsgl.stopping import StopChecker

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase

    from loopsgl.core import Req
    from loopsgl.managers.base_scheduler import ForwardInput
    from loopsgl.managers.scheduler_components.cache import CacheManager
    from loopsgl.managers.scheduler_components.decode import DecodeManager
    from loopsgl.model_executor import ForwardOutput


@dataclass(frozen=True, slots=True)
class PendingBatchResult:
    """Retain a forward snapshot, token output, and any outstanding token-row write until scheduling processes them."""

    forward_input: ForwardInput
    output: ForwardOutput
    row_write_ready: torch.cuda.Event | None = None
    loop_steps: tuple[int, ...] | None = None


@dataclass(kw_only=True, frozen=True, slots=True)
class SchedulerBatchResultProcessor:
    """Apply completed tokens, stopping rules, and prefix-cache updates to requests."""

    tokenizer: PreTrainedTokenizerBase
    eos_token_ids: set[int]
    cache_manager: CacheManager
    decode_manager: DecodeManager
    free_req_resources: Callable[[Req], None]
    prefix_insert_limit: Callable[[Req], int | None]
    loop_steps: int

    def process_batch_result(
        self,
        forward_input: ForwardInput,
        output: ForwardOutput,
        loop_steps: tuple[int, ...] | None = None,
    ) -> List[TokenIDOutput]:
        batch = forward_input.batch
        next_tokens_cpu = output.next_tokens_cpu
        reply: List[TokenIDOutput] = []
        with self.cache_manager.lazy_free_region():
            for i, (req, is_middle_chunk) in enumerate(
                zip(forward_input.reqs, forward_input.is_middle_chunks)
            ):
                if is_middle_chunk:
                    req.inflight_middle_chunks -= 1
                    assert req.inflight_middle_chunks >= 0
                    continue
                # Overlap scheduling can leave a decode in flight after a request finishes.
                if req.finished():
                    continue
                next_token = int(next_tokens_cpu[i].item())
                req.append_host(next_token)
                # Overlap can advance device_len before its token reaches the host.
                finished = req.num_generated_tokens >= req.sampling_params.max_tokens
                params = req.sampling_params
                matched_eos = not params.ignore_eos and next_token in self.eos_token_ids
                stop_text = None
                if params.stop:
                    if req.stop_checker is None:
                        req.stop_checker = StopChecker(self.tokenizer, params)
                    if not matched_eos:
                        stop_text = req.stop_checker.check(next_token)
                finished |= matched_eos or stop_text is not None
                reply.append(
                    TokenIDOutput(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        skip_token=matched_eos and not params.no_stop_trim,
                        stop_text=stop_text,
                        skip_special_tokens=params.skip_special_tokens,
                        loop_steps=(
                            (loop_steps[i] if loop_steps is not None else self.loop_steps)
                            if params.return_loop_steps
                            else None
                        ),
                        stop_buffer_length=(
                            req.stop_checker.buffer_length if req.stop_checker is not None else 0
                        ),
                    )
                )

                if finished:
                    req.finished_reason = (
                        "eos" if matched_eos else "stop" if stop_text is not None else "length"
                    )
                    self.decode_manager.remove_req(req)
                    self.free_req_resources(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(
                        req, finished=False, insert_limit=self.prefix_insert_limit(req)
                    )

        return reply
