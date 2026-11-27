from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List

from loopsgl.managers.io_struct import (
    LogprobOutput,
    ScoredTokenIDOutput,
    TokenIDOutput,
    TokenIDOutputWithDepth,
)
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
        next_tokens = output.next_tokens_cpu.tolist()
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
                next_token = next_tokens[i]
                req.append_host(next_token)
                # Overlap can advance device_len before its token reaches the host.
                finished = req.num_generated_tokens >= req.sampling_params.max_new_tokens
                params = req.sampling_params
                matched_eos = not params.ignore_eos and next_token in self.eos_token_ids
                stop_text = None
                if params.stop:
                    if req.stop_checker is None:
                        req.stop_checker = StopChecker(self.tokenizer, params)
                    if not matched_eos:
                        stop_text = req.stop_checker.check(next_token)
                finished |= matched_eos or stop_text is not None
                finish_reason = None
                if matched_eos:
                    finish_reason = {"type": "stop", "matched": next_token}
                elif stop_text is not None:
                    finish_reason = {"type": "stop", "matched": req.stop_checker.matched_stop}
                elif finished:
                    finish_reason = {"type": "length", "length": req.num_generated_tokens}
                reply.append(
                    TokenIDOutput(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        completion_tokens=req.num_generated_tokens,
                        finish_reason=finish_reason,
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

                decode_depth = reply[-1].loop_steps if not batch.is_prefill else None
                if req.return_logprob:
                    reply[-1] = ScoredTokenIDOutput(
                        **vars(reply[-1]),
                        decode_loop_steps=decode_depth,
                        logprobs=self._logprob_fields(
                            forward_input, output, i, next_token, req.num_generated_tokens
                        ),
                    )
                elif decode_depth is not None:
                    reply[-1] = TokenIDOutputWithDepth(
                        **vars(reply[-1]), decode_loop_steps=decode_depth
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

    def _logprob_fields(
        self, data: ForwardInput, output: ForwardOutput, row: int, token: int, count: int
    ) -> LogprobOutput:
        req = data.reqs[row]
        scores, metadata = output.logits_output, data.logits_metadata
        if scores is None or metadata is None or scores.next_token_logprobs is None:
            raise ValueError("Missing token logprobs for a scoring request.")
        if metadata.output_positions_cpu[row] - metadata.prompt_lens_cpu[row] != count - 1:
            raise ValueError("Logprob position does not match the completed token.")
        result = LogprobOutput(
            output_token_logprobs_val=[scores.next_token_logprobs[row].item()],
            output_token_logprobs_idx=[token],
        )
        top_count = req.top_logprobs_num
        if top_count:
            result.output_top_logprobs_val = [
                scores.next_token_top_logprobs_val[row, :top_count].tolist()
            ]
            result.output_top_logprobs_idx = [
                scores.next_token_top_logprobs_idx[row, :top_count].tolist()
            ]
        if data.batch.is_prefill:
            start = metadata.extend_logprob_start_lens_cpu[row]
            length = metadata.extend_seq_lens_cpu[row]
            if start < length:
                begin, end = metadata.input_logprob_offsets_cpu[row : row + 2]
                if scores.input_token_logprobs is None:
                    raise ValueError("Missing input logprobs for a scoring request.")
                result.input_token_logprobs_val = [None] + scores.input_token_logprobs[
                    begin:end
                ].tolist()
                result.input_token_logprobs_idx = list(metadata.input_ids_cpu[row][start:])
                if top_count:
                    result.input_top_logprobs_val = [None] + scores.input_top_logprobs_val[
                        begin:end, :top_count
                    ].tolist()
                    result.input_top_logprobs_idx = [None] + scores.input_top_logprobs_idx[
                        begin:end, :top_count
                    ].tolist()
        return result
