# Adapted from SGLang v0.5.20: python/sglang/benchmark/utils.py.
# Licensed under the Apache License, Version 2.0. See LICENSES/sglang.txt.

from __future__ import annotations

import json
import os
import random
from typing import TYPE_CHECKING, Any, TypeVar

from loopsgl.core import SamplingParams

if TYPE_CHECKING:
    from loopsgl.benchmark.datasets.common import DatasetRow

_RequestT = TypeVar("_RequestT")


def remove_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text


def download_and_cache_hf_file(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Download a file from Hugging Face and cache it locally."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def is_file_valid_json(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"{path} exists but json loading fails ({e=}), thus treat as invalid file")
        return False


def get_warmup_requests(reqs: list[_RequestT], *, num_requests: int, seed: int) -> list[_RequestT]:
    """Sample real requests without changing their contents or measured order."""
    return random.Random(seed).sample(reqs, min(num_requests, len(reqs)))


def replay_mean_decode_depth(reqs: list[DatasetRow], max_depth: int, minimum: int) -> float:
    """Validate recorded exit depths and return their mean over decode tokens.

    Each request's first output comes from prefill and has no recorded depth.
    """
    depth_sum = decode_tokens = 0
    for req in reqs:
        if req.exit_depths is None or any(
            type(value) is not int or not minimum <= value <= max_depth for value in req.exit_depths
        ):
            raise ValueError("Every replay request must supply valid recorded exit depths.")
        depth_sum += sum(req.exit_depths)
        decode_tokens += len(req.exit_depths)
    if not decode_tokens:
        raise ValueError("Replay requires decode tokens after the first outputs.")
    return depth_sum / decode_tokens


def prepare_sampling_params(
    reqs: list[DatasetRow],
    ignore_eos: bool,
    extra_request_body: dict[str, Any],
    replay_exits: bool,
    max_seq_len: int,
) -> list[SamplingParams]:
    """Prepare ordered benchmark sampling settings and validate replay budgets."""
    if not reqs:
        raise ValueError("No requests remain after dataset filtering.")
    sampling_params = []
    for req in reqs:
        params = {
            "temperature": 0,
            "max_new_tokens": req.output_len,
            "ignore_eos": ignore_eos,
            **extra_request_body,
        }
        if "max_tokens" in params:
            raise ValueError("Use max_new_tokens in --extra-request-body, as in SGLang.")
        if replay_exits:
            params["replay_exit_depths"] = req.exit_depths
        prepared = SamplingParams(**params)
        prepared.verify()
        if req.prompt_len + prepared.max_new_tokens > max_seq_len:
            raise ValueError("A request exceeds the engine context length.")
        if replay_exits:
            if req.exit_depths is None or len(req.exit_depths) != prepared.max_new_tokens - 1:
                raise ValueError("Replay sampling budgets must match the recorded output lengths.")
            if not prepared.ignore_eos or prepared.stop:
                raise ValueError("Recorded exit replay requires ignore_eos and no stop sequences.")
        sampling_params.append(prepared)
    return sampling_params
