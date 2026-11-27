r"""Export memory-constrained throughput summaries into the paper's plotting format.

Each workload compares full-depth CB with shared and depth-indexed KV against
adaptive CDB with refill using shared and last-exited KV.
Throughput is normalized against full-depth CB with depth-indexed KV.
Policy-specific token capacities are supplied by ``ablations/benchmark_memory.sh``.
Budget labels are read from the ``_<budget>gib`` summary filename suffix.
Decode batch medians use logged batches with requests queued, after warmup.
CDB uses recurrent-core batch sizes.

Example:
    uv run python shells/paper/exporters/export_memory_results.py \
        outputs/ablations/memory/memory_ouro_*gib.jsonl \
        --output docs/paper/figs/throughput/memory_ouro_loopsgl.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _throughput import (
    Run,
    add_cli_args,
    fingerprint,
    json_key,
    load_runs,
    model_info,
    positive_int,
    write_export,
)

CONFIGURATIONS = (
    ("cb", "shared"),
    ("cb", "depth_indexed"),
    ("cdb", "shared"),
    ("cdb", "last_exited"),
)
THRESHOLD = 0.0001
VOLATILE_FIELDS = {"nccl_port", "result_filename"}


def memory_budget(run: Run) -> int:
    """Read the KV budget from a paper summary filename, including repeat suffixes."""
    match = re.search(r"_(\d+)gib(?:_repeat\d+)?$", run.source.stem)
    if not match:
        raise ValueError(f"{run.source}: expected a _<budget>gib filename suffix")
    return int(match[1])


def decode_batch_sizes(run: Run) -> list[int]:
    """Read measured decode samples from the matching paper experiment log."""
    backend = "cb" if run.backend == "cb" else "cdb-refill"
    pattern = f"{run.source.stem}_{backend}_{run.config['loop_cache_policy']}_*.out"
    logs = sorted((run.source.parent / "logs").glob(pattern))
    if len(logs) != 1:
        raise ValueError(f"{run.source}: expected one decode log matching {pattern}")
    windows: list[list[int]] = []
    samples: list[int] = []
    idle = False
    started = False
    phase = "Decode batch" if run.backend == "cb" else "Recurrent batch"
    field = "running-req" if run.backend == "cb" else "batch-token"
    with logs[0].open() as handle:
        for line in handle:
            if "Scheduler is idle, waiting for new reqs" in line:
                idle = True
            if not re.search(r"(?:Prefill|Decode|Recurrent|Coda) batch,", line):
                continue
            if idle and started:
                windows.append(samples)
                samples = []
            idle = False
            started = True
            if phase not in line:
                continue
            queued = re.search(r"#queue-req: (\d+)", line)
            batch = re.search(rf"#{field}: (\d+)", line)
            if queued and int(queued[1]) > 0 and batch:
                samples.append(int(batch[1]))
    if started:
        windows.append(samples)
    expected = 1 if run.config["skip_warmup"] else 2
    if len(windows) != expected or not windows[-1]:
        raise ValueError(f"{logs[0]}: expected {expected} generation windows with decode samples")
    return windows[-1]


def validate_workload(runs: list[Run]) -> None:
    """Require the four configurations to replay one population with common settings."""
    if {(run.backend, run.config["loop_cache_policy"]) for run in runs} != set(CONFIGURATIONS):
        raise ValueError(
            "Each workload requires CB shared/depth-indexed and CDB shared/last-exited"
        )
    rows = Counter((run.source, run.backend, run.config["loop_cache_policy"]) for run in runs)
    if max(rows.values()) != 1:
        raise ValueError("A summary file holds more than one row for a configuration")
    repeats = Counter((run.backend, run.config["loop_cache_policy"]) for run in runs)
    if len(set(repeats.values())) != 1:
        raise ValueError("A workload has unequal repeat counts across configurations")
    if len({(run.num_requests, run.input_tokens, run.output_tokens) for run in runs}) != 1:
        raise ValueError("A workload mixes request counts or token totals")
    ignored = VOLATILE_FIELDS | {
        "scheduler",
        "loop_cache_policy",
        "max_total_tokens",
        "page_size",
        "loop_params",
        "cdb_config",
    }
    settings = [
        {key: value for key, value in run.config.items() if key not in ignored} for run in runs
    ]
    if len({json_key(value) for value in settings}) != 1:
        raise ValueError("A workload mixes common engine or benchmark settings")
    loop_settings = [
        {key: value for key, value in run.config["loop_params"].items() if key != "exit_threshold"}
        for run in runs
    ]
    if len({json_key(value) for value in loop_settings}) != 1:
        raise ValueError("A workload mixes loop settings beyond the exit threshold")
    for run in runs:
        config = run.config
        policy = config["loop_cache_policy"]
        positive_int(config["max_total_tokens"], "max_total_tokens")
        if config["page_size"] != (1 if policy == "last_exited" else 16):
            raise ValueError("Expected page size 1 for last-exited KV and 16 otherwise")
        if run.depth != 4:
            raise ValueError("Memory experiments require four loop steps")
        if config["retraction_mode"] != "none" or run.row["num_retracted_reqs"] != 0:
            raise ValueError(
                "Memory experiments require disabled retractions and zero retracted requests"
            )
        if config["min_free_slots_delay"] not in (None, 1):
            raise ValueError("Memory experiments require no slot-admission batching delay")
        if (
            run.backend == "cdb"
            and type(config["cdb_config"]["disable_lookahead_exit"]) is not bool
        ):
            raise ValueError("CDB must record disable_lookahead_exit as a boolean")
        if run.backend == "cdb" and (
            run.threshold != THRESHOLD
            or config["cdb_config"]["refill"] is not True
            or config["cdb_config"]["min_coda_batch_size"] != 1
        ):
            raise ValueError(
                f"CDB requires refill, threshold {THRESHOLD}, and minimum coda batch size 1"
            )
    cdb_runs = [run for run in runs if run.backend == "cdb"]
    if len({json_key(run.config["cdb_config"]) for run in cdb_runs}) != 1:
        raise ValueError("A workload mixes CDB settings")
    if not all(
        math.isclose(run.mean_depth, cdb_runs[0].mean_depth, rel_tol=1e-10) for run in cdb_runs
    ):
        raise ValueError("CDB runs disagree on replay mean_decode_depth")
    for policy in {run.config["loop_cache_policy"] for run in runs}:
        if (
            len(
                {
                    run.config["max_total_tokens"]
                    for run in runs
                    if run.config["loop_cache_policy"] == policy
                }
            )
            != 1
        ):
            raise ValueError(f"A workload mixes token capacities for {policy} KV")


def aggregate_workload(runs: list[Run]) -> dict[str, Any]:
    """Average repeats separately for each scheduler and KV policy."""
    validate_workload(runs)
    points = []
    for backend, policy in CONFIGURATIONS:
        group = [
            run
            for run in runs
            if (run.backend, run.config["loop_cache_policy"]) == (backend, policy)
        ]
        run = group[0]
        rates = [item.row["output_throughput"] for item in group]
        points.append(
            {
                "backend": backend,
                "refill": True,
                "threshold": run.threshold,
                "depth": run.depth,
                "width": run.width,
                "min_coda_batch_size": 1,
                "loop_cache_policy": policy,
                "max_total_tokens": run.config["max_total_tokens"],
                "page_size": run.config["page_size"],
                "gen_tps": statistics.mean(rates),
                "gen_tps_std": statistics.stdev(rates) if len(rates) > 1 else None,
                "median_decode_batch_size": statistics.median(
                    size for item in group for size in decode_batch_sizes(item)
                ),
                "wall_s": statistics.mean(item.row["total_latency"] for item in group),
                "preemptions": 0,
                "repeats": len(group),
                "mean_exit_depth": None if backend == "cb" else run.mean_depth,
            }
        )
    baseline = next(
        point["gen_tps"] for point in points if point["loop_cache_policy"] == "depth_indexed"
    )
    for point in points:
        point["speedup"] = point["gen_tps"] / baseline
    return {
        "name": runs[0].name,
        "memory_budget_gib": memory_budget(runs[0]),
        "num_requests": runs[0].num_requests,
        "source_summaries": sorted({str(run.source) for run in runs}),
        "measurement_configs": [
            json.loads(value)
            for value in sorted(
                {
                    json_key(
                        {
                            key: value
                            for key, value in run.config.items()
                            if key not in VOLATILE_FIELDS
                        }
                    )
                    for run in runs
                }
            )
        ],
        "baseline_gen_tps": baseline,
        "points": points,
    }


def build_results(paths: list[Path], *, model_config: Path | None = None) -> dict[str, Any]:
    """Build one comparison per workload and KV budget from benchmark summaries."""
    if not paths:
        raise ValueError("Supply at least one benchmark summary")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("A summary file was supplied more than once")
    runs = load_runs(paths)
    _, model = model_info(runs, model_config)
    if model["effective_config"]["model_type"] != "ouro":
        raise ValueError("Memory experiments support only Ouro")
    grouped: dict[tuple[str, int], list[Run]] = {}
    for run in runs:
        if run.name not in {"sharegpt", "arxiv"}:
            raise ValueError("Memory experiments support only ShareGPT and ArXiv")
        grouped.setdefault((run.name, memory_budget(run)), []).append(run)
    return {
        "meta": {
            "model": runs[0].config["model_path"],
            "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": [fingerprint(path) for path in paths],
            "model_configuration": model,
            "timing_scope": "whole generation, including IPC, detokenization, and result collection",
            "depth_basis": "decode-token mean exit depth",
            "decode_batch_scope": (
                "Median of logged CB decode or CDB recurrent-core batches with requests queued. "
                "Warmup and queue drain are excluded."
            ),
        },
        "workloads": [
            aggregate_workload(group)
            for _, group in sorted(grouped.items(), key=lambda item: item[0])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_cli_args(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        write_export(build_results(args.summaries, model_config=args.model_config), args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
