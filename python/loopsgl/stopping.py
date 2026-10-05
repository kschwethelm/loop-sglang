# Portions adapted from SGLang's sampling_params.py and schedule_batch.py.
# Copyright 2023-2024 SGLang Team.
# Licensed under the Apache License, Version 2.0.
# See LICENSES/sglang.txt in the repository root.

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from loopsgl.core import SamplingParams
    from transformers import PreTrainedTokenizerBase

MAX_STOP_COUNT = 32


def normalize_stops(value: str | List[str] | None) -> List[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ValueError("stop must be a string or a list of strings.")
    if len(values) > MAX_STOP_COUNT:
        raise ValueError(f"At most {MAX_STOP_COUNT} stop strings are allowed.")
    return list(values)


class StopChecker:
    """Match generated text and bound the suffix withheld from streaming clients."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, params: SamplingParams) -> None:
        self.tokenizer = tokenizer
        self.stop_strs = normalize_stops(params.stop)
        self.no_stop_trim = params.no_stop_trim
        self.skip_special_tokens = params.skip_special_tokens
        self.special_token_ids = (
            set(tokenizer.all_special_ids) if self.skip_special_tokens else set()
        )
        self.output_ids: List[int] = []
        self.buffer_length = max((len(stop) - 1 for stop in self.stop_strs), default=0)
        # A Unicode character can span four byte tokens, and generation may tokenize a
        # stop differently from encode(stop).
        self.tail_tokens = (
            max(
                (
                    max(
                        len(stop.encode("utf-8")),
                        len(tokenizer.encode(stop, add_special_tokens=False)),
                    )
                    for stop in self.stop_strs
                ),
                default=0,
            )
            + 1
        )

    def check(self, token_id: int) -> str | None:
        """Return trimmed output on a match, or None while generation continues."""
        if token_id not in self.special_token_ids:
            self.output_ids.append(token_id)
        tail = self.tokenizer.decode(
            self.output_ids[-self.tail_tokens :], skip_special_tokens=self.skip_special_tokens
        )
        if not any(stop in tail for stop in self.stop_strs):
            return None

        # Validate on the complete generated text to preserve token boundaries.
        text = self.tokenizer.decode(self.output_ids, skip_special_tokens=self.skip_special_tokens)
        for stop in self.stop_strs:
            start = text.find(stop)
            if start != -1:
                return text[: start + len(stop) if self.no_stop_trim else start]
        return None
