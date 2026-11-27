# Adapted from SGLang v0.5.20: python/sglang/benchmark/serving.py.
# Reference revision 94602c9c2b7cbdb8efd5c52802dac6a1c180089e.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py
# Licensed under the Apache License, Version 2.0. See LICENSES/sglang.txt.

"""Benchmark the native HTTP serving interface with independently scheduled arrivals.

For example, run ``python -m loopsgl.benchmark.serving --model MODEL
--dataset-name random-ids --request-rate 2 --num-prompts 10 --output-file results.jsonl``.
Latency starts at HTTP submission, and TTFT follows the first nonempty cumulative text.
The first request is sent immediately; subsequent arrival gaps are exponentially distributed.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import random
import time
import traceback
import warnings
from dataclasses import dataclass, field
from itertools import cycle, islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, List, Optional, Tuple

import aiohttp
import numpy as np
import orjson
from tqdm import tqdm

from loopsgl.benchmark.datasets import DATASET_MAPPING, DatasetRow, get_dataset
from loopsgl.benchmark.utils import (
    get_warmup_requests,
    prepare_sampling_params,
    replay_mean_decode_depth,
)
from loopsgl.core import SamplingParams
from loopsgl.utils import load_tokenizer, set_ulimit

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

TIMING_CONVENTIONS = {
    "arrivals": "seeded-cumulative-exponential-first-arrival-immediate",
    "latency": "http-post-to-last-response-event",
    "ttft": "first-nonempty-cumulative-text",
    "itl": "chunk-gap-divided-by-completion-count-increase",
}


@dataclass
class RequestFuncInput:
    prompt: str | list[int]
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    sampling_params: SamplingParams


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    start_time: float = 0.0


@dataclass
class BenchmarkMetrics:
    # Request counts and token totals
    completed: int
    total_input: int
    total_input_text: int
    total_output: int
    total_output_retokenized: int

    # Throughput (req/s and tok/s)
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    total_throughput: float
    total_throughput_retokenized: float

    # TTFT - Time to First Token (ms)
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p90_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float

    # TPOT - Time per Output Token, excluding the first token (ms)
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p90_tpot_ms: float
    p95_tpot_ms: float
    p99_tpot_ms: float

    # ITL - Inter-Token Latency (ms)
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p90_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    max_itl_ms: float

    # E2E - End-to-End request latency (ms)
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p90_e2e_latency_ms: float
    p95_e2e_latency_ms: float
    p99_e2e_latency_ms: float

    # Concurrency and peak metrics
    concurrency: float
    mean_normalized_latency_s_per_token: float
    max_output_tokens_per_s: float = 0.0
    max_concurrent_requests: int = 0


def calculate_metrics(
    input_requests: Optional[List[DatasetRow]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    total_input_text = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(outputs[i].generated_text, add_special_tokens=False)
            )
            retokenized_output_lens.append(retokenized_output_len)
            if input_requests is not None:
                total_input += input_requests[i].prompt_len
                total_input_text += input_requests[i].text_prompt_len
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )

    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0

    successful_outputs = [output for output in outputs if output.success]
    if successful_outputs:
        min_start_time = min(output.start_time for output in successful_outputs)
        max_end_time = max(output.start_time + output.latency for output in successful_outputs)

        duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
        tokens_per_second = np.zeros(duration_seconds)
        concurrent_requests_per_second = np.zeros(duration_seconds)

        for output in outputs:
            if not output.success:
                continue

            token_times = [output.start_time + output.ttft]
            current_time = token_times[0]
            for itl_value in output.itl:
                current_time += itl_value
                token_times.append(current_time)

            for token_time in token_times:
                second_bucket = int(token_time - min_start_time)
                if 0 <= second_bucket < duration_seconds:
                    tokens_per_second[second_bucket] += 1

            request_start_second = int(output.start_time - min_start_time)
            request_end_second = int((output.start_time + output.latency) - min_start_time)

            for second in range(
                request_start_second, min(request_end_second + 1, duration_seconds)
            ):
                concurrent_requests_per_second[second] += 1

        if len(tokens_per_second) > 0:
            max_output_tokens_per_s = float(np.max(tokens_per_second))
            max_concurrent_requests = int(np.max(concurrent_requests_per_second))

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_input_text=total_input_text,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        total_throughput=(total_input + sum(output_lens)) / dur_s,
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
        p95_ttft_ms=np.percentile(ttfts or 0, 95) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p90_tpot_ms=np.percentile(tpots or 0, 90) * 1000,
        p95_tpot_ms=np.percentile(tpots or 0, 95) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p90_itl_ms=np.percentile(itls or 0, 90) * 1000,
        p95_itl_ms=np.percentile(itls or 0, 95) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        max_itl_ms=float(np.max(itls or 0)) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p90_e2e_latency_ms=np.percentile(e2e_latencies, 90) * 1000,
        p95_e2e_latency_ms=np.percentile(e2e_latencies, 95) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
        mean_normalized_latency_s_per_token=float(
            np.mean([output.latency / output.output_len for output in outputs if output.success])
        ),
        max_output_tokens_per_s=max_output_tokens_per_s,
        max_concurrent_requests=max_concurrent_requests,
    )

    return metrics, output_lens


def _create_bench_client_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60), read_bufsize=10 * 1024**2
    )


async def async_request_sglang_generate(
    request_func_input: RequestFuncInput, pbar: tqdm | None = None
) -> RequestFuncOutput:
    """Collect native cumulative-text events using SGLang's timing conventions."""
    params = request_func_input.sampling_params
    payload = {
        (
            "text" if isinstance(request_func_input.prompt, str) else "input_ids"
        ): request_func_input.prompt,
        "sampling_params": dataclasses.asdict(params),
        "stream": True,
    }
    output = RequestFuncOutput(prompt_len=request_func_input.prompt_len)
    async with _create_bench_client_session() as session:
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        last_output_len = 0
        completion_tokens = 0
        finish_reason = None
        done = False
        try:
            async with session.post(request_func_input.api_url, json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    sse_data = chunk_bytes[6:] if chunk_bytes.startswith(b"data: ") else chunk_bytes
                    output.latency = time.perf_counter() - st
                    if sse_data == b"[DONE]":
                        done = True
                        continue
                    data = orjson.loads(sse_data)
                    if "error" in data:
                        raise RuntimeError(str(data["error"]))
                    meta = data["meta_info"]
                    count = meta["completion_tokens"]
                    if type(count) is not int or count < completion_tokens:
                        raise ValueError("Invalid cumulative generated-token count.")
                    completion_tokens = count
                    if meta.get("finish_reason") is not None:
                        finish_reason = meta["finish_reason"]
                    if data.get("text"):
                        timestamp = time.perf_counter()
                        output.generated_text = data["text"]
                        if output.ttft == 0.0:
                            output.ttft = time.perf_counter() - st
                        else:
                            num_new_tokens = count - last_output_len
                            if num_new_tokens == 0:
                                continue
                            chunk_gap = timestamp - most_recent_timestamp
                            output.itl.extend([chunk_gap / num_new_tokens] * num_new_tokens)
                        most_recent_timestamp = timestamp
                        last_output_len = count
                if (
                    not done
                    or not isinstance(finish_reason, dict)
                    or finish_reason.get("type") not in {"length", "stop"}
                ):
                    raise RuntimeError(
                        "Stream ended without successful terminal metadata and [DONE]."
                    )
                if completion_tokens < 1:
                    raise RuntimeError("A completed request must generate at least one token.")
                if (
                    params.ignore_eos
                    and not params.stop
                    and completion_tokens != params.max_new_tokens
                ):
                    raise RuntimeError(
                        "A fixed-length request did not return its requested token count."
                    )
                output.output_len = completion_tokens
                output.success = True
        except Exception:
            output.error = traceback.format_exc()
    if pbar is not None:
        pbar.update(1)
    return output


async def get_request(
    input_requests: list[RequestFuncInput],
    request_rate: float,
    arrival_seed: int,
    start_time: float,
) -> AsyncIterator[RequestFuncInput]:
    """Submit the first request immediately and retain absolute arrival deadlines."""
    offsets = np.zeros(len(input_requests))
    if not math.isinf(request_rate):
        offsets[1:] = (
            np.random.default_rng(arrival_seed)
            .exponential(1 / request_rate, size=max(0, len(input_requests) - 1))
            .cumsum()
        )
    for request, offset in zip(input_requests, offsets, strict=True):
        if offset > 0:
            await asyncio.sleep(max(0.0, start_time + float(offset) - time.perf_counter()))
        yield request


async def wait_for_server(base_url: str, timeout: float) -> None:
    deadline = time.perf_counter() + timeout
    async with aiohttp.ClientSession() as session:
        while time.perf_counter() < deadline:
            try:
                async with session.get(
                    base_url + "/health",
                    timeout=aiohttp.ClientTimeout(
                        total=min(5, max(0.01, deadline - time.perf_counter()))
                    ),
                ) as response:
                    if response.status == 200:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(min(1, max(0, deadline - time.perf_counter())))
    raise TimeoutError(f"Server was not ready after {timeout} seconds: {base_url}")


async def get_server_info(base_url: str) -> dict[str, Any]:
    """Read scheduler counters and effective server configuration while generation is idle."""
    async with _create_bench_client_session() as session:
        async with session.get(base_url + "/server_info") as response:
            response.raise_for_status()
            return await response.json()


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Warm up, submit and drain a complete trace, then save aggregate metrics."""
    set_ulimit()
    base_url = (args.base_url or f"http://{args.host}:{args.port}").rstrip("/")
    if args.ready_check_timeout_sec > 0:
        await wait_for_server(base_url, args.ready_check_timeout_sec)
    server_info = await get_server_info(base_url)
    args.model = args.model or server_info["model_path"]
    tokenizer = load_tokenizer(args.model)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.sharegpt_context_len = min(
        args.sharegpt_context_len or server_info["max_seq_len"], server_info["max_seq_len"]
    )
    rows = get_dataset(args, tokenizer, model_id=args.model)
    if not rows:
        raise ValueError("No requests remain after dataset filtering.")
    rows = list(islice(cycle(rows), args.num_prompts))
    server_args = server_info["server_args"]
    if args.warmup_requests is None:
        args.warmup_requests = 2 * server_args["max_running_req"]
    extra_request_body = json.loads(args.extra_request_body) if args.extra_request_body else {}
    if not isinstance(extra_request_body, dict):
        raise ValueError("extra_request_body must be a JSON object.")
    mean_decode_depth = None
    replaying_cdb = args.replay_exits and server_info["scheduler"] == "cdb"
    if args.replay_exits:
        if args.dataset_name != "custom" or args.custom_input_format != "token-ids" or args.shuffle:
            raise ValueError("Recorded replay requires unshuffled custom token-ID input.")
        # Lookahead routing completes at least two steps before a replayed CDB exit.
        lookahead = replaying_cdb and not server_args["cdb_config"]["disable_lookahead_exit"]
        mean_decode_depth = replay_mean_decode_depth(
            rows, server_info["loop_params"]["loop_steps"], minimum=2 if lookahead else 1
        )
    params = prepare_sampling_params(
        rows,
        not args.disable_ignore_eos,
        {"temperature": args.temperature, "top_p": args.top_p, **extra_request_body},
        replaying_cdb,
        server_info["max_seq_len"],
    )
    requests = [
        RequestFuncInput(
            row.prompt,
            base_url + "/generate",
            row.prompt_len,
            param.max_new_tokens,
            args.model,
            param,
        )
        for row, param in zip(rows, params, strict=True)
    ]
    config = vars(args).copy()
    config["request_rate"] = args.request_rate if math.isfinite(args.request_rate) else "inf"
    config["timing_conventions"] = TIMING_CONVENTIONS
    warmup_requests = get_warmup_requests(
        requests, num_requests=args.warmup_requests, seed=args.seed
    )
    warmup_outputs = await asyncio.gather(
        *(async_request_sglang_generate(request) for request in warmup_requests)
    )
    for output in warmup_outputs:
        if not output.success:
            raise RuntimeError("Warmup failed:\n" + output.error)
    if args.flush_cache:
        async with _create_bench_client_session() as session:
            async with session.post(base_url + "/flush_cache") as response:
                response.raise_for_status()
                await response.read()
    retractions_before = (await get_server_info(base_url))["num_retracted_reqs"]
    pbar = None if args.disable_tqdm else tqdm(total=len(requests))
    tasks = []
    benchmark_start_time = time.perf_counter()
    try:
        async for request in get_request(
            requests, args.request_rate, args.arrival_seed, benchmark_start_time
        ):
            tasks.append(asyncio.create_task(async_request_sglang_generate(request, pbar)))
        outputs = await asyncio.gather(*tasks)
        benchmark_duration = time.perf_counter() - benchmark_start_time
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if pbar is not None:
            pbar.close()
    failed = [output for output in outputs if not output.success]
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(requests)} requests failed.\n"
            + "\n".join(output.error for output in failed[:3])
        )
    server_info = await get_server_info(base_url)
    metrics, _ = calculate_metrics(rows, outputs, benchmark_duration, tokenizer)
    result = {
        "backend": "sglang",
        "duration": benchmark_duration,
        "num_retracted_reqs": server_info["num_retracted_reqs"] - retractions_before,
        **dataclasses.asdict(metrics),
        "server_info": server_info,
        "config": config,
    }
    if mean_decode_depth is not None:
        result["mean_decode_depth"] = mean_decode_depth
    if args.output_file:
        path = Path(args.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"config", "server_info"}},
            indent=2,
            allow_nan=False,
        )
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--model")
    parser.add_argument("--dataset-name", choices=list(DATASET_MAPPING), default="sharegpt")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--custom-input-format", choices=["text", "token-ids"], default="text")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replay-exits", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--trace-duration-s", type=float)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-seed", type=int, default=0)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=None,
        help="Number of sampled real requests for upfront warmup; defaults to twice server concurrency.",
    )
    parser.add_argument("--ready-check-timeout-sec", type=float, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--disable-ignore-eos", action="store_true")
    parser.add_argument(
        "--extra-request-body",
        type=str,
        default=None,
        help="Additional sampling parameters as a JSON object. Accepts max_new_tokens.",
    )
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--output-file")
    parser.add_argument("--sharegpt-output-len", type=int)
    parser.add_argument("--sharegpt-context-len", type=int)
    parser.add_argument("--random-input-len", type=int, default=1024)
    parser.add_argument("--random-output-len", type=int, default=1024)
    parser.add_argument("--random-range-ratio", type=float, default=0.0)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--prompt-suffix", default="")
    args = parser.parse_args(argv)
    if math.isnan(args.request_rate) or args.request_rate <= 0:
        parser.error("request-rate must be positive and finite, or positive infinity")
    if args.trace_duration_s is not None:
        if (
            not math.isfinite(args.trace_duration_s)
            or args.trace_duration_s <= 0
            or not math.isfinite(args.request_rate)
        ):
            parser.error("trace-duration-s requires a positive finite duration and request rate")
        if args.num_prompts is not None:
            parser.error("specify only one of num-prompts and trace-duration-s")
        args.num_prompts = math.ceil(args.request_rate * args.trace_duration_s)
    if args.num_prompts is None:
        args.num_prompts = 1000
    if (
        args.num_prompts < 1
        or (args.warmup_requests is not None and args.warmup_requests < 0)
        or not math.isfinite(args.ready_check_timeout_sec)
        or args.ready_check_timeout_sec < 0
    ):
        parser.error(
            "counts must be positive, warmup must be nonnegative, and readiness timeout must be finite and nonnegative"
        )
    return args


def cli_main() -> None:
    asyncio.run(benchmark(parse_args()))


if __name__ == "__main__":
    cli_main()
