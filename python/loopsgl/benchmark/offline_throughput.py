# Adapted from SGLang v0.5.20: python/sglang/benchmark/offline_throughput.py.
# Licensed under the Apache License, Version 2.0. See LICENSES/sglang.txt.

"""Benchmark throughput in offline mode.

Dataset preparation and warmup precede the timed generation call.
Warmup samples real requests up to twice the configured concurrency and clears cached prefixes.
The measured call includes IPC submission, prefill, decode, text decoding, and result collection.
Text inputs also include prompt tokenization; token ID inputs bypass tokenization.
Random datasets submit sampled token IDs directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import time
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from loopsgl.benchmark.datasets import DATASET_MAPPING, DatasetRow, get_dataset
from loopsgl.core import SamplingParams
from loopsgl.env import ENV
from loopsgl.managers.scheduler_components.config import SchedulerConfig
from loopsgl.server_args import ServerArgs
from loopsgl.utils import profile_range

if TYPE_CHECKING:
    from loopsgl import Engine


@dataclasses.dataclass
class BenchArgs:
    backend: str = "engine"
    result_filename: str = ""
    dataset_name: str = "sharegpt"
    dataset_path: str = ""
    custom_input_format: str = "text"
    replay_exits: bool = False
    num_prompts: int = 1000
    shuffle: bool = True
    sharegpt_output_len: int | None = None
    sharegpt_context_len: int | None = None
    random_input_len: int = 1024
    random_output_len: int = 1024
    random_range_ratio: float = 0.0
    seed: int = 42
    disable_ignore_eos: bool = False
    extra_request_body: str | None = None
    apply_chat_template: bool = False
    skip_warmup: bool = False
    prompt_suffix: str = ""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--backend", choices=["engine"], default=BenchArgs.backend)
        parser.add_argument("--result-filename", type=str, default=BenchArgs.result_filename)
        parser.add_argument(
            "--dataset-name",
            choices=list(DATASET_MAPPING),
            default=BenchArgs.dataset_name,
            help="Random datasets submit token IDs directly: random samples from ShareGPT, "
            "and random-ids samples from the tokenizer vocabulary.",
        )
        parser.add_argument("--dataset-path", type=str, default="", help="Path to the dataset.")
        parser.add_argument(
            "--custom-input-format",
            choices=["text", "token-ids"],
            default=BenchArgs.custom_input_format,
            help="Custom JSONL input representation. Token IDs bypass prompt tokenization "
            "and use the stored output_len or output_ids length as the generation budget.",
        )
        parser.add_argument(
            "--replay-exits",
            action=argparse.BooleanOptionalAction,
            default=BenchArgs.replay_exits,
            help="Replay exit_depths in the custom token-ID JSONL requests for CDB.",
        )
        parser.add_argument(
            "--shuffle",
            action=argparse.BooleanOptionalAction,
            default=BenchArgs.shuffle,
            help="Shuffle dataset records before sampling requests. Enabled by default.",
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=BenchArgs.num_prompts,
            help="Number of prompts to process. Default is 1000.",
        )
        parser.add_argument(
            "--sharegpt-output-len",
            type=int,
            default=BenchArgs.sharegpt_output_len,
            help="Output length for each request. Overrides the output length from ShareGPT.",
        )
        parser.add_argument(
            "--sharegpt-context-len",
            type=int,
            default=BenchArgs.sharegpt_context_len,
            help="Drop requests exceeding this context length. Defaults to the engine context length.",
        )
        parser.add_argument(
            "--random-input-len",
            type=int,
            default=BenchArgs.random_input_len,
            help="Number of input tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-output-len",
            type=int,
            default=BenchArgs.random_output_len,
            help="Number of output tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-range-ratio",
            type=float,
            default=BenchArgs.random_range_ratio,
            help="Range of sampled ratio of input/output length, used only for random dataset.",
        )
        parser.add_argument("--seed", type=int, default=BenchArgs.seed, help="The random seed.")
        parser.add_argument(
            "--disable-ignore-eos", action="store_true", help="Disable ignore EOS token."
        )
        parser.add_argument(
            "--extra-request-body",
            type=str,
            default=None,
            metavar='{"key1": "value1", "key2": "value2"}',
            help="Additional sampling parameters as a JSON object. Accepts max_new_tokens.",
        )
        parser.add_argument(
            "--apply-chat-template", action="store_true", help="Apply chat template."
        )
        parser.add_argument("--skip-warmup", action="store_true", help="Skip the warmup batch.")
        parser.add_argument(
            "--prompt-suffix", type=str, default="", help="Suffix appended to prompts."
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> BenchArgs:
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def get_warmup_requests(
    reqs: List[DatasetRow], *, num_requests: int, seed: int
) -> List[DatasetRow]:
    """Sample real requests without changing their contents or measured order."""
    return random.Random(seed).sample(reqs, min(num_requests, len(reqs)))


def throughput_test_once(
    backend_name: str,
    backend: Engine,
    reqs: List[DatasetRow],
    ignore_eos: bool,
    extra_request_body: Dict[str, Any],
    replay_exits: bool = False,
    nvtx_label: str | None = None,
) -> Dict[str, Any]:
    if not reqs:
        raise ValueError("No requests remain after dataset filtering.")
    prompt = [r.prompt for r in reqs]
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
        params["max_tokens"] = params.pop("max_new_tokens")
        if params["max_tokens"] <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if req.prompt_len + params["max_tokens"] > backend.max_seq_len:
            raise ValueError("A request exceeds the engine context length.")
        if replay_exits:
            params["replay_exit_depths"] = req.exit_depths
        sampling_params.append(SamplingParams(**params))

    if replay_exits:
        if any(req.exit_depths is None for req in reqs):
            raise ValueError("Every replay request must supply recorded exit depths.")
        if any(
            len(req.exit_depths) != params.max_tokens - 1
            for req, params in zip(reqs, sampling_params)
        ):
            raise ValueError("Replay sampling budgets must match the recorded output lengths.")
        if any(not params.ignore_eos or params.stop for params in sampling_params):
            raise ValueError("Recorded exit replay requires ignore_eos and no stop sequences.")

    expected_output_lens = [
        params.max_tokens if params.ignore_eos and not params.stop else None
        for params in sampling_params
    ]
    with profile_range(nvtx_label):
        st = time.perf_counter()
        if isinstance(prompt[0], str):
            gen_out = backend.generate(prompt=prompt, sampling_params=sampling_params)
        else:
            gen_out = backend.generate(input_ids=prompt, sampling_params=sampling_params)
        latency = time.perf_counter() - st

    if nvtx_label is not None and ENV.ENABLE_NVTX:
        backend.flush_profile()
        # Nsight saves trace buffers asynchronously; keep workers alive after the measured range.
        time.sleep(1.0)

    if len(gen_out) != len(reqs):
        raise RuntimeError("The engine did not return every benchmark request.")
    for output, expected_len in zip(gen_out, expected_output_lens):
        if expected_len is not None and len(output["token_ids"]) != expected_len:
            raise RuntimeError("A fixed-length request did not return its requested token count.")

    total_input_tokens = sum(r.prompt_len for r in reqs)
    total_output_tokens = sum(len(o["token_ids"]) for o in gen_out)
    return {
        "backend": backend_name,
        "successful_requests": len(gen_out),
        "total_latency": latency,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "num_retracted_reqs": backend.num_retracted_reqs,
        **backend.batch_size_metrics,
        "request_throughput": len(gen_out) / latency,
        "input_throughput": total_input_tokens / latency,
        "output_throughput": total_output_tokens / latency,
        "total_throughput": (total_input_tokens + total_output_tokens) / latency,
    }


def throughput_test(server_args: ServerArgs, bench_args: BenchArgs) -> Dict[str, Any]:
    if bench_args.backend != "engine":
        raise ValueError("Only the engine backend is supported.")
    if bench_args.custom_input_format not in {"text", "token-ids"}:
        raise ValueError(f"Unknown custom input format: {bench_args.custom_input_format}")
    if bench_args.custom_input_format == "token-ids":
        if bench_args.dataset_name != "custom":
            raise ValueError("custom_input_format applies only to the custom JSONL dataset.")
    if (
        bench_args.dataset_name in {"random", "random-ids"}
        or bench_args.custom_input_format == "token-ids"
    ):
        if bench_args.prompt_suffix or bench_args.apply_chat_template:
            raise ValueError("Token ID input cannot use prompt_suffix or apply_chat_template.")
    if bench_args.num_prompts <= 0:
        raise ValueError("num_prompts must be positive.")
    if bench_args.replay_exits:
        if bench_args.dataset_name != "custom" or bench_args.custom_input_format != "token-ids":
            raise ValueError("Exit replay requires custom token-ID input.")
        if bench_args.disable_ignore_eos:
            raise ValueError("Exit replay requires fixed output lengths with ignore_eos.")
    if bench_args.sharegpt_context_len is not None and bench_args.sharegpt_context_len <= 0:
        raise ValueError("sharegpt_context_len must be positive.")
    extra_request_body = (
        json.loads(bench_args.extra_request_body) if bench_args.extra_request_body else {}
    )
    if not isinstance(extra_request_body, dict):
        raise ValueError("extra_request_body must be a JSON object.")

    from loopsgl import Engine

    kwargs = {
        attr.name: getattr(server_args, attr.name)
        for attr in dataclasses.fields(SchedulerConfig)
        if attr.name not in {"model_path", "tp_info", "dtype"} and not attr.name.startswith("_")
    }
    backend = Engine(
        server_args.model_path,
        dtype=server_args.dtype,
        tp_size=server_args.tp_info.size,
        **kwargs,
    )
    try:
        tokenizer = backend.tokenizer
        random.seed(bench_args.seed)
        np.random.seed(bench_args.seed)
        context_len = min(
            bench_args.sharegpt_context_len or backend.max_seq_len,
            backend.max_seq_len,
        )
        dataset_args = dataclasses.replace(bench_args, sharegpt_context_len=context_len)
        input_requests = get_dataset(dataset_args, tokenizer)
        if not input_requests:
            raise ValueError("No requests remain after dataset filtering.")

        replaying_cdb = bench_args.replay_exits and server_args.scheduler == "cdb"
        if replaying_cdb and any(req.exit_depths is None for req in input_requests):
            raise ValueError("Every replay request must supply recorded exit depths.")

        if not bench_args.skip_warmup:
            logging.info("\nWarmup...")
            warmup_requests = get_warmup_requests(
                input_requests,
                num_requests=2 * server_args.max_running_req,
                seed=bench_args.seed,
            )
            throughput_test_once(
                backend_name=bench_args.backend,
                backend=backend,
                reqs=warmup_requests,
                ignore_eos=not bench_args.disable_ignore_eos,
                extra_request_body=extra_request_body,
                replay_exits=replaying_cdb,
            )
            backend.flush_cache()
            time.sleep(0.5)

        logging.info("\nBenchmark...")
        result = throughput_test_once(
            backend_name=bench_args.backend,
            backend=backend,
            reqs=input_requests,
            ignore_eos=not bench_args.disable_ignore_eos,
            extra_request_body=extra_request_body,
            replay_exits=replaying_cdb,
            nvtx_label="benchmark.generate",
        )
        result["config"] = {
            "enable_nvtx": bool(ENV.ENABLE_NVTX),
            "disable_overlap_scheduling": bool(ENV.DISABLE_OVERLAP_SCHEDULING),
            "model_path": server_args.model_path,
            "dtype": str(server_args.dtype),
            "tp_size": server_args.tp_info.size,
            **kwargs,
            **dataclasses.asdict(bench_args),
            "loop_steps": backend.get_server_info()["loop_params"]["loop_steps"],
            "sharegpt_context_len": context_len,
        }
        result["config"]["loop_params"] = backend.get_server_info()["loop_params"]
        result["config"]["cdb_config"] = dataclasses.asdict(server_args.cdb_config)
    finally:
        backend.shutdown()

    if bench_args.result_filename:
        with open(bench_args.result_filename, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(result) + "\n")

    print("\n{s:{c}^{n}}".format(s=" Offline Throughput Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", result["backend"]))
    print("{:<40} {:<10}".format("Successful requests:", result["successful_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["total_latency"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print("{:<40} {:<10}".format("Total generated tokens:", result["total_output_tokens"]))
    print("{:<40} {:<10}".format("Request retractions:", result["num_retracted_reqs"]))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", result["request_throughput"]))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", result["input_throughput"]))
    print(
        "{:<40} {:<10.2f}".format("Output token throughput (tok/s):", result["output_throughput"])
    )
    print("{:<40} {:<10.2f}".format("Total token throughput (tok/s):", result["total_throughput"]))
    print("=" * 50)
    return result


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the offline throughput.")
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    if args.shell_mode:
        parser.error("The offline benchmark does not support shell mode.")
    bench_args = BenchArgs.from_cli_args(args)
    server_args = ServerArgs.from_cli_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    throughput_test(server_args, bench_args)


if __name__ == "__main__":
    cli_main()
