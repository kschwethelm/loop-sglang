# Adapted from SGLang v0.5.20: python/sglang/benchmark/one_batch.py.
# Licensed under the Apache License, Version 2.0. See LICENSES/sglang.txt.

"""Benchmark a single static batch through the model runner APIs.

Prefill populates real KV caches before the fixed batch decodes, without a serving scheduler.
Decode-call wall time includes batch preparation, model execution, sampling, and token updates.
CUDA-event model time excludes batch preparation, attention metadata preparation, and sampling.
All JSONL latency values are in seconds.

Usage:
    python -m loopsgl.benchmark.one_batch --model-path KristianS7/Ouro-1.4B \\
        --batch-size 1 8 32 --input-len 128 512 --output-len 32 --loop-steps 4
    python -m loopsgl.benchmark.one_batch --model-path KristianS7/Ouro-1.4B \\
        --batch-size 8 --input-len 512 --profile --profile-stage decode
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch

from loopsgl.core import Req, SamplingParams
from loopsgl.kernel.batch import prepare_batch_inputs
from loopsgl.managers.schedule_batch import ScheduleBatch
from loopsgl.managers.scheduler_components.cache import CacheManager
from loopsgl.managers.scheduler_components.prefill import PrefillAdder
from loopsgl.managers.scheduler_components.table import TableManager
from loopsgl.model_executor import ModelRunner
from loopsgl.model_executor.sample import make_device_tensor
from loopsgl.server_args import ServerArgs


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int, ...] = (1,)
    input_len: Tuple[int, ...] = (1024,)
    output_len: Tuple[int, ...] = (16,)
    prompt_filename: str = ""
    result_filename: str = "result.jsonl"
    log_decode_step: int = 0
    profile: bool = False
    profile_record_shapes: bool = False
    profile_activities: Tuple[str, ...] = ("CPU", "GPU")
    profile_stage: str = "all"
    profile_prefix: str = "profile"
    profile_start_step: int | None = None
    profile_steps: int | None = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--run-name", default=BenchArgs.run_name)
        parser.add_argument("--batch-size", type=int, nargs="+", default=BenchArgs.batch_size)
        parser.add_argument("--input-len", type=int, nargs="+", default=BenchArgs.input_len)
        parser.add_argument("--output-len", type=int, nargs="+", default=BenchArgs.output_len)
        parser.add_argument(
            "--prompt-filename",
            default=BenchArgs.prompt_filename,
            help="Text file with one prompt per line. Overrides synthetic input lengths.",
        )
        parser.add_argument("--result-filename", default=BenchArgs.result_filename)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log every Nth decode step. The first five steps are always logged.",
        )
        parser.add_argument("--profile", action="store_true", help="Enable profiling.")
        parser.add_argument("--profile-record-shapes", action="store_true")
        parser.add_argument(
            "--profile-activities",
            nargs="+",
            default=BenchArgs.profile_activities,
            choices=["CPU", "GPU", "CUDA_PROFILER"],
            help="CPU/GPU use the PyTorch profiler; CUDA_PROFILER controls Nsight capture.",
        )
        parser.add_argument(
            "--profile-stage",
            choices=["all", "prefill", "decode"],
            default=BenchArgs.profile_stage,
        )
        parser.add_argument(
            "--profile-prefix",
            "--profile-filename-prefix",
            default=BenchArgs.profile_prefix,
            help="Trace filename prefix. Directory defaults to LOOPSGL_TORCH_PROFILER_DIR or '.'.",
        )
        parser.add_argument(
            "--profile-start-step",
            type=int,
            default=None,
            help="First decode step to profile (0-indexed). Defaults to output_len // 2.",
        )
        parser.add_argument(
            "--profile-steps",
            type=int,
            default=None,
            help="Number of decode steps to profile. Defaults to one.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> BenchArgs:
        values = {field.name: getattr(args, field.name) for field in dataclasses.fields(cls)}
        for name in ("batch_size", "input_len", "output_len", "profile_activities"):
            values[name] = tuple(values[name])
        return cls(**values)


class TorchModelRunner:
    """Static-batch adapter over the model runner, page allocator, and batch utilities."""

    def __init__(self, config: ServerArgs) -> None:
        self.model_runner = ModelRunner(config)
        self.device = self.model_runner.device
        self.table_manager = TableManager(config.max_running_req, self.model_runner.page_table)
        self.cache_manager = CacheManager(
            self.model_runner.num_pages,
            config.page_size,
            self.model_runner.page_table,
            disable_radix_cache=True,
        )
        self.prefill_budget = config.max_extend_tokens
        self.reqs: List[Req] = []
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def clear(self) -> None:
        self.synchronize()
        for req in self.reqs:
            if req.kv.holds_kv:
                self.cache_manager.release_req(req)
                assert req.kv.req_pool_idx is not None
                self.table_manager.free(req.kv.req_pool_idx)
                req.kv.reset_device_state()
        self.reqs = []
        self.cache_manager.clear()

    def max_batch_size(self, input_len: int, output_len: int) -> int:
        pages_per_req = (
            input_len + output_len + self.cache_manager.page_size - 1
        ) // self.cache_manager.page_size
        return min(self.table_manager.available_size, self.model_runner.num_pages // pages_per_req)

    @torch.inference_mode()
    def forward(self, batch: ScheduleBatch) -> torch.Tensor:
        assert torch.cuda.current_stream(self.device) == self.model_runner.stream
        self.cache_manager.allocate_paged(batch.reqs, num_running_reqs=len(self.reqs))
        batch.prepare(self.model_runner.sampler)
        forward_batch = batch.get_forward_batch()
        self.model_runner.graph_runner.pad_batch(forward_batch)
        input_mapping = prepare_batch_inputs(forward_batch, self.model_runner.page_table)
        forward_batch.input_ids = self.table_manager.token_pool[input_mapping]
        is_middle_chunks = [
            batch.is_prefill and req.device_len < req.input_len for req in batch.reqs
        ]
        write_positions = (
            make_device_tensor(
                [
                    req.device_len if not middle else -1
                    for req, middle in zip(batch.reqs, is_middle_chunks)
                ],
                torch.int32,
                self.device,
            )
            if batch.is_prefill
            else forward_batch.seq_lens[: batch.size]
        )
        if not batch.is_prefill:
            batch.prepare_for_decode()

        graph_runner = self.model_runner.graph_runner
        if self.model_runner.prefill_graph_runner.bucket_for(forward_batch) is not None:
            logits = self.model_runner.prefill_graph_runner.replay(
                forward_batch,
                self.model_runner.attn_backend,
                timing_events=(self.start_event, self.end_event),
            )
        else:
            with self.model_runner.ctx.forward_batch(forward_batch):
                if graph_runner.can_use_cuda_graph(forward_batch):
                    # Match GraphRunner.replay, bracketing only graph execution with CUDA events.
                    graph_runner.buffer.copy_from(forward_batch)
                    self.model_runner.attn_backend.prepare_for_replay(forward_batch)
                    forward_batch.record_metadata_ready()
                    self.start_event.record(self.model_runner.stream)
                    graph_runner.graph_map[forward_batch.padded_size].replay()
                    self.end_event.record(self.model_runner.stream)
                    logits = graph_runner.buffer.logits[: batch.size]
                else:
                    self.model_runner.attn_backend.prepare_metadata(forward_batch)
                    forward_batch.record_metadata_ready()
                    self.start_event.record(self.model_runner.stream)
                    logits = self.model_runner.model.forward()
                    self.end_event.record(self.model_runner.stream)

        next_token_ids = self.model_runner.sampler.sample(
            logits[: batch.size], batch.sampling_info
        ).to(torch.int32)
        self.table_manager.token_pool[batch.req_pool_indices, write_positions] = next_token_ids
        for req, middle in zip(batch.reqs, is_middle_chunks):
            req.complete_one(is_middle_chunk=middle)
            if middle:
                req.inflight_middle_chunks -= 1
        return next_token_ids

    def extend(self, reqs: List[Req]) -> Tuple[torch.Tensor, ScheduleBatch]:
        self.reqs = reqs
        pending = list(reqs)
        next_token_ids = torch.empty(len(reqs), dtype=torch.int32, device=self.device)
        while pending:
            adder = PrefillAdder(
                token_budget=self.prefill_budget,
                reserved_size=0,
                cache_manager=self.cache_manager,
                table_manager=self.table_manager,
                watermark_size=0,
                has_running_reqs=self.table_manager.allocated_size > 0,
            )
            added: List[Req] = []
            for req in pending:
                if adder.try_add_one(req) is None:
                    break
                added.append(req)
            if not added:
                raise RuntimeError("Cannot assemble the static batch with the available KV cache.")
            batch = ScheduleBatch(added, "prefill")
            middle = [req.device_len < req.input_len for req in added]
            tokens = self.forward(batch)
            completed_indices = [index for index, is_middle in enumerate(middle) if not is_middle]
            if completed_indices:
                token_indices = make_device_tensor(completed_indices, torch.int64, self.device)
                request_indices = make_device_tensor(
                    [added[index].uid for index in completed_indices], torch.int64, self.device
                )
                next_token_ids[request_indices] = tokens[token_indices]
            pending = [req for req, is_middle in zip(added, middle) if is_middle] + pending[
                len(added) :
            ]
        return next_token_ids, ScheduleBatch(reqs, "decode")

    def decode(self, input_token_ids: torch.Tensor, batch: ScheduleBatch) -> torch.Tensor:
        # The token pool already contains the previous forward's sampled tokens.
        return self.forward(batch)

    def gpu_model_latency(self) -> float:
        """Return the last synchronized model forward's elapsed time in seconds."""
        return self.start_event.elapsed_time(self.end_event) / 1000.0

    def shutdown(self) -> None:
        self.clear()
        self.model_runner.shutdown()


def load_model(server_args: ServerArgs) -> TorchModelRunner:
    return TorchModelRunner(server_args)


def prepare_synthetic_inputs_for_latency_test(
    batch_size: int,
    input_len: int,
    output_len: int,
    vocab_size: int,
    custom_inputs: List[List[int]] | None = None,
) -> List[Req]:
    input_ids = (
        custom_inputs
        if custom_inputs
        else np.random.randint(
            0, min(10000, vocab_size), (batch_size, input_len), dtype=np.int32
        ).tolist()
    )
    return [
        Req(
            uid=index,
            origin_input_ids=torch.tensor(tokens, dtype=torch.int32),
            sampling_params=SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True),
        )
        for index, tokens in enumerate(input_ids)
    ]


def start_profile(
    profile_activities: Tuple[str, ...],
    profile_record_shapes: bool = False,
) -> Any:
    if "CUDA_PROFILER" in profile_activities:
        torch.cuda.cudart().cudaProfilerStart()
        print("CUDA Profiler started (nsys will begin capturing)")
        return None
    activities = []
    if "CPU" in profile_activities:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if "GPU" in profile_activities:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(
        activities=activities,
        with_stack=True,
        record_shapes=profile_record_shapes,
    )
    profiler.start()
    return profiler


def stop_profile(
    profiler: Any,
    profile_activities: Tuple[str, ...],
    trace_filename: str,
) -> None:
    if "CUDA_PROFILER" in profile_activities:
        torch.cuda.cudart().cudaProfilerStop()
        print("CUDA Profiler stopped")
        return
    profiler.stop()
    Path(trace_filename).parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(trace_filename)
    print(f"PyTorch profiler trace saved to {trace_filename}")


def _create_torch_profiler_filename(
    profile_prefix: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    stage: str,
) -> str:
    output_dir = os.environ.get("LOOPSGL_TORCH_PROFILER_DIR", ".")
    return str(
        Path(output_dir)
        / f"{profile_prefix}_batch{batch_size}_input{input_len}_output{output_len}_{stage}.trace.json.gz"
    )


@torch.inference_mode()
def latency_test_run_once(
    model_runner: TorchModelRunner,
    reqs: List[Req],
    batch_size: int,
    input_len: int,
    output_len: int,
    bench_args: BenchArgs,
) -> dict[str, Any] | None:
    model_runner.clear()
    lengths = [req.input_len for req in reqs]
    max_input_len = max(lengths)
    if max_input_len + output_len > model_runner.model_runner.max_seq_len:
        print(
            f"Skipping ({batch_size}, {input_len}, {output_len}): exceeds model runner context length"
        )
        return None
    needed_pages = sum(
        (length + output_len + model_runner.cache_manager.page_size - 1)
        // model_runner.cache_manager.page_size
        for length in lengths
    )
    if (
        batch_size > model_runner.table_manager.available_size
        or needed_pages > model_runner.model_runner.num_pages
    ):
        print(f"Skipping ({batch_size}, {input_len}, {output_len}): exceeds request/KV capacity")
        return None

    measurement_results: dict[str, Any] = {
        "run_name": bench_args.run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "input_lens": lengths,
        "output_len": output_len,
    }
    profiler = None
    profile_active = False
    trace_filename = ""
    try:
        if bench_args.profile and bench_args.profile_stage in ("all", "prefill"):
            trace_filename = _create_torch_profiler_filename(
                bench_args.profile_prefix, batch_size, input_len, output_len, "prefill"
            )
            profiler = start_profile(
                bench_args.profile_activities, bench_args.profile_record_shapes
            )
            profile_active = True
        model_runner.synchronize()
        prefill_replays_before = model_runner.model_runner.prefill_graph_runner.num_replays
        step_name = f"step[PREFILL bs={batch_size} toks={sum(lengths)}]"
        tic = time.perf_counter()
        with (
            torch.cuda.nvtx.range(step_name),
            torch.profiler.record_function(step_name) if profile_active else nullcontext(),
        ):
            next_token_ids, batch = model_runner.extend(reqs)
        model_runner.synchronize()
        prefill_latency = time.perf_counter() - tic
        if profile_active:
            stop_profile(profiler, bench_args.profile_activities, trace_filename)
            profile_active = False
        print(
            f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {sum(lengths) / prefill_latency:9.2f} token/s"
        )
        measurement_results.update(
            prefill_latency=prefill_latency,
            prefill_throughput=sum(lengths) / prefill_latency,
            prefill_graph_replays=(
                model_runner.model_runner.prefill_graph_runner.num_replays - prefill_replays_before
            ),
        )

        decode_latencies = []
        gpu_decode_latencies = []
        decode_context_lens = []
        profile_start = (
            bench_args.profile_start_step
            if bench_args.profile_start_step is not None
            else output_len // 2
        )
        profile_end = profile_start + (bench_args.profile_steps or 1)
        enable_profile_decode = bench_args.profile and bench_args.profile_stage in ("all", "decode")
        for index in range(output_len - 1):
            model_runner.synchronize()
            if enable_profile_decode and index == profile_start:
                trace_filename = _create_torch_profiler_filename(
                    bench_args.profile_prefix, batch_size, input_len, output_len, "decode"
                )
                profiler = start_profile(
                    bench_args.profile_activities, bench_args.profile_record_shapes
                )
                profile_active = True
            contexts = [req.device_len - 1 for req in reqs]
            decode_context_lens.append([min(contexts), max(contexts)])
            step_name = f"step[DECODE bs={batch_size} ctx={min(contexts)}:{max(contexts)}]"
            tic = time.perf_counter()
            with (
                torch.cuda.nvtx.range(step_name),
                torch.profiler.record_function(step_name) if profile_active else nullcontext(),
            ):
                next_token_ids = model_runner.decode(next_token_ids, batch)
            model_runner.synchronize()
            latency = time.perf_counter() - tic
            gpu_latency = model_runner.gpu_model_latency()
            if profile_active and index >= profile_end - 1:
                stop_profile(profiler, bench_args.profile_activities, trace_filename)
                profile_active = False
            decode_latencies.append(latency)
            gpu_decode_latencies.append(gpu_latency)
            if index < 5 or (
                bench_args.log_decode_step > 0 and index % bench_args.log_decode_step == 0
            ):
                print(
                    f"Decode {index}. Batch size: {batch_size}, latency: {latency:6.5f} s, GPU model: {gpu_latency:6.5f} s"
                )

        if decode_latencies:
            median_decode_latency = float(np.median(decode_latencies))
            median_gpu_decode_latency = float(np.median(gpu_decode_latencies))
            measurement_results.update(
                median_decode_latency=median_decode_latency,
                median_decode_throughput=batch_size / median_decode_latency,
                median_gpu_decode_latency=median_gpu_decode_latency,
            )
            print(
                f"Decode. median latency: {median_decode_latency:6.5f} s, GPU model: {median_gpu_decode_latency:6.5f} s"
            )
        total_latency = prefill_latency + sum(decode_latencies)
        measurement_results.update(
            total_latency=total_latency,
            total_throughput=(sum(lengths) + batch_size * output_len) / total_latency,
            decode_latencies=decode_latencies,
            gpu_decode_latencies=gpu_decode_latencies,
            decode_context_lens=decode_context_lens,
            cuda_graph=model_runner.model_runner.graph_runner.max_graph_bs >= batch_size,
            padded_batch_size=next(
                (
                    size
                    for size in model_runner.model_runner.graph_runner.graph_bs_list
                    if size >= batch_size
                ),
                batch_size,
            ),
        )
        return measurement_results
    finally:
        if profile_active:
            stop_profile(profiler, bench_args.profile_activities, trace_filename)
        model_runner.clear()


def latency_test(server_args: ServerArgs, bench_args: BenchArgs) -> List[dict[str, Any]]:
    if server_args.scheduler != "cb":
        raise ValueError("The static-batch benchmark requires scheduler='cb'.")
    if server_args.loop_cache_policy == "last_exited":
        raise ValueError("The static-batch benchmark does not support last_exited KV.")
    # Capture requested sizes exactly so a static batch does not silently acquire padding.
    graph_limit = server_args.cuda_graph_max_bs
    graph_sizes = sorted(
        {size for size in bench_args.batch_size if graph_limit is None or size <= graph_limit}
    )
    server_args = dataclasses.replace(
        server_args,
        max_running_req=max(bench_args.batch_size),
        cuda_graph_bs=graph_sizes,
    )
    model_runner = load_model(server_args)
    result_list: List[dict[str, Any]] = []
    custom_inputs: List[List[int]] = []
    try:
        if bench_args.prompt_filename:
            from loopsgl.utils import load_tokenizer

            tokenizer = load_tokenizer(server_args.model_path)
            prompts = Path(bench_args.prompt_filename).read_text(encoding="utf-8").splitlines()
            custom_inputs = [
                tokenizer.encode(prompt.strip()) for prompt in prompts if prompt.strip()
            ]
            if not custom_inputs or any(not tokens for tokens in custom_inputs):
                raise ValueError("The prompt file must contain nonempty tokenized prompts.")
        np.random.seed(42)
        with torch.cuda.stream(model_runner.model_runner.stream):
            for batch_size, input_len, output_len in itertools.product(
                bench_args.batch_size, bench_args.input_len, bench_args.output_len
            ):
                aligned_inputs = (
                    (
                        custom_inputs[:batch_size]
                        + [custom_inputs[-1]] * max(0, batch_size - len(custom_inputs))
                    )
                    if custom_inputs
                    else None
                )
                reqs = prepare_synthetic_inputs_for_latency_test(
                    batch_size,
                    input_len,
                    output_len,
                    server_args.model_config.vocab_size,
                    aligned_inputs,
                )
                print(f"Warmup ... batch={batch_size}, input={input_len}, output={output_len}")
                warmup_args = dataclasses.replace(bench_args, profile=False)
                warmup_output_len = min(32, output_len)
                warmup_reqs = prepare_synthetic_inputs_for_latency_test(
                    batch_size,
                    input_len,
                    warmup_output_len,
                    server_args.model_config.vocab_size,
                    [req.origin_input_ids.tolist() for req in reqs],
                )
                try:
                    warmup_result = latency_test_run_once(
                        model_runner,
                        warmup_reqs,
                        batch_size,
                        input_len,
                        warmup_output_len,
                        warmup_args,
                    )
                    if warmup_result is None:
                        continue
                    print("Benchmark ...")
                    result = latency_test_run_once(
                        model_runner, reqs, batch_size, input_len, output_len, bench_args
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"Skipping ({batch_size}, {input_len}, {output_len}): CUDA out of memory")
                    torch.cuda.empty_cache()
                    continue
                if result is not None:
                    result["config"] = {
                        "model_path": server_args.model_path,
                        "dtype": str(server_args.dtype),
                        "attention_backend": server_args.attention_backend,
                        "page_size": server_args.page_size,
                        "num_pages": model_runner.model_runner.num_pages,
                        "max_seq_len": model_runner.model_runner.max_seq_len,
                        "max_prefill_length": server_args.max_extend_tokens,
                        "cuda_graph_backend_prefill": server_args.cuda_graph_backend_prefill,
                        "cuda_graph_bs_prefill": model_runner.model_runner.prefill_graph_runner.buckets,
                        "loop_steps": getattr(
                            model_runner.model_runner.model.model, "loop_steps", 1
                        ),
                        "loop_cache_policy": server_args.loop_cache_policy,
                        "json_model_override_args": server_args.json_model_override_args,
                        "device_name": torch.cuda.get_device_name(model_runner.device),
                        "torch_version": torch.__version__,
                        "profile": bench_args.profile,
                        "timing_scope": "synchronized_decode_call_and_gpu_model",
                    }
                    result_list.append(result)
                    if bench_args.result_filename:
                        with open(bench_args.result_filename, "a", encoding="utf-8") as fout:
                            fout.write(json.dumps(result) + "\n")
    finally:
        model_runner.shutdown()
    print(f"Completed {len(result_list)} static-batch measurements.")
    return result_list


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the latency of a single static batch.")
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tensor_parallel_size != 1 or args.shell_mode:
        parser.error(
            "The static-batch benchmark supports tensor-parallel-size=1 and no shell mode."
        )
    if args.exit_threshold is not None:
        parser.error("The static-batch benchmark requires fixed loop depth.")
    if any(
        value <= 0
        for name in ("batch_size", "input_len", "output_len")
        for value in getattr(args, name)
    ):
        parser.error("Batch sizes and input/output lengths must be positive.")
    if args.max_extend_tokens <= 0:
        parser.error("max-prefill-length must be positive.")
    if args.log_decode_step < 0 or (
        args.profile_start_step is not None and args.profile_start_step < 0
    ):
        parser.error("Decode logging and profile start steps must be nonnegative.")
    if args.profile_steps is not None and args.profile_steps <= 0:
        parser.error("profile-steps must be positive.")
    if "CUDA_PROFILER" in args.profile_activities and len(args.profile_activities) != 1:
        parser.error("CUDA_PROFILER cannot be combined with CPU/GPU profiling.")
    if args.profile and args.profile_stage in ("all", "decode"):
        for length in args.output_len:
            start = args.profile_start_step if args.profile_start_step is not None else length // 2
            if start + (args.profile_steps or 1) > length - 1:
                parser.error("The decode profiling window must fit within output-len - 1 steps.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    latency_test(ServerArgs.from_cli_args(args), BenchArgs.from_cli_args(args))


if __name__ == "__main__":
    cli_main()
