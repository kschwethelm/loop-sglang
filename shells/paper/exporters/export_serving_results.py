r"""Export a serving-rate sweep into a JSON file for the paper plots.

Reads raw summary JSONL written by ``python -m loopsgl.benchmark.serving``.
Mean normalized latency is plotted against the offered request rate, with one panel per workload.
CB, CDB-norefill, and CDB-refill series retain their exit thresholds and minimum coda batch sizes.
Repeated measurements are averaged for each configuration and request rate.
Standard latency summaries are converted from milliseconds to seconds.
Each panel also lists the mean exit depth of every replayed CDB recording, averaged over its decode
tokens. ``--requests`` restricts that average to the first requests of a workload, matching the
offline benchmarks. Relative recording paths resolve against ``--workload-root``.

The sweep is produced by ``shells/paper/serving_rate-sweep.sh``.

Example:
    uv run python shells/paper/exporters/export_serving_results.py \
        outputs/serving-rate/serving-rate_ouro_alpaca_shared.jsonl \
        outputs/serving-rate/serving-rate_ouro_sharegpt_shared.jsonl \
        outputs/serving-rate/serving-rate_ouro_arxiv_shared.jsonl \
        --output docs/paper/figs/serving/serving_ouro_loopsgl.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _throughput import fingerprint, json_key, positive_int
from _workloads import load_recording, parse_request_limits

TIMING_CONVENTIONS = {
    "arrivals": "seeded-cumulative-exponential-first-arrival-immediate",
    "latency": "http-post-to-last-response-event",
    "ttft": "first-nonempty-cumulative-text",
    "itl": "chunk-gap-divided-by-completion-count-increase",
}
MODEL_KEYS = {"Ouro-1.4B": "ouro", "huginn-0125": "huginn"}


def positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return float(value)


def build_results(
    paths: list[Path], workload_root: Path, request_limits: dict[str, int] | None = None
) -> dict[str, Any]:
    """One panel per workload, each assembled from its matching benchmark summaries."""
    request_limits = request_limits or {}
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("A summary file was supplied more than once.")
    models = set()
    panels: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    seen = set()
    recordings: dict[tuple[str, float], tuple[Path, int]] = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"{path}: no serving rows.")
        for row in rows:
            config = row["config"]
            info = row["server_info"]
            server = info["server_args"]
            if row["backend"] != "sglang" or info["scheduler"] not in {"cb", "cdb"}:
                raise ValueError("Expected native CB/CDB serving results.")
            model = config["model"]
            model_key = MODEL_KEYS.get(model.rstrip("/").rsplit("/", 1)[-1])
            server_model_key = MODEL_KEYS.get(info["model_path"].rstrip("/").rsplit("/", 1)[-1])
            dataset_path = Path(config["dataset_path"])
            match = re.fullmatch(
                r"(?:.*/)?(?P<workload>sharegpt|alpaca|arxiv)/(?P<model>ouro|huginn)/"
                r"[^/]+_q(?P<threshold>[^/]+)\.jsonl",
                dataset_path.as_posix(),
            )
            if (
                model_key is None
                or server_model_key != model_key
                or match is None
                or match["model"] != model_key
            ):
                raise ValueError("Unrecognized model or paper workload identity.")
            models.add((model_key, model, info["model_path"]))
            name = match["workload"]
            if (
                config["dataset_name"] != "custom"
                or config["custom_input_format"] != "token-ids"
                or config["shuffle"]
                or not config["replay_exits"]
                or config["disable_ignore_eos"]
                or config["sharegpt_output_len"] is not None
                or config.get("extra_request_body") not in (None, "", "{}")
                or not config["flush_cache"]
            ):
                raise ValueError(
                    "Paper exports require unshuffled, fixed-budget token-ID replay with flushed prefixes."
                )
            if config["timing_conventions"] != TIMING_CONVENTIONS:
                raise ValueError("Incompatible serving timing conventions.")
            if (
                server.get("enable_cuda_event_timing")
                or info.get("cuda_event_timing")
                or server.get("enable_nvtx")
            ):
                raise ValueError("Profiled rows cannot supply paper serving measurements.")
            count = positive_int(config["num_prompts"], "num_prompts")
            if positive_int(row["completed"], "completed") != count:
                raise ValueError("A serving point did not complete its requested population.")
            rate = positive_float(config["request_rate"], "request_rate")
            duration = positive_float(row["duration"], "duration")
            positive_float(row["mean_normalized_latency_s_per_token"], "normalized latency")
            for total_key, throughput_key in (
                ("total_input", "input_throughput"),
                ("total_output", "output_throughput"),
            ):
                total = positive_int(row[total_key], total_key)
                if total < count or not math.isclose(
                    positive_float(row[throughput_key], throughput_key), total / duration
                ):
                    raise ValueError("Inconsistent token totals or throughput.")
            if not math.isclose(
                positive_float(row["request_throughput"], "request_throughput"), count / duration
            ):
                raise ValueError("Inconsistent request throughput.")
            trace_duration = config["trace_duration_s"]
            if (
                trace_duration is not None
                and math.ceil(rate * positive_float(trace_duration, "trace_duration_s")) != count
            ):
                raise ValueError("Request count does not match the trace-duration rule.")
            repeat = config["repeat_index"]
            if type(repeat) is not int or repeat < 0 or type(config["arrival_seed"]) is not int:
                raise ValueError("Invalid repeat index or arrival seed.")
            backend = info["scheduler"]
            threshold = info["loop_params"]["exit_threshold"] if backend == "cdb" else None
            width = positive_int(server["max_running_req"], "max_running_req")
            refill = server["cdb_config"]["refill"] if backend == "cdb" else True
            if type(refill) is not bool:
                raise ValueError("Invalid refill mode.")
            min_coda = (
                min(
                    positive_int(
                        server["cdb_config"]["min_coda_batch_size"], "min_coda_batch_size"
                    ),
                    width,
                )
                if backend == "cdb" and refill
                else 1
            )
            if backend == "cdb":
                if (
                    isinstance(threshold, bool)
                    or not isinstance(threshold, (int, float))
                    or not math.isfinite(threshold)
                    or threshold != float(match["threshold"])
                ):
                    raise ValueError("Invalid or mismatched CDB replay threshold.")
                recording = (
                    dataset_path if dataset_path.is_absolute() else workload_root / dataset_path,
                    positive_int(info["loop_params"]["loop_steps"], "loop_steps"),
                )
                if recordings.setdefault((name, threshold), recording) != recording:
                    raise ValueError(f"The {name} panel replays two recordings at q={threshold}.")
            elif info["loop_params"]["exit_threshold"] is not None:
                raise ValueError("CB must run at full depth.")
            common_server = {
                key: value
                for key, value in server.items()
                if key
                not in {
                    "scheduler",
                    "cdb_config",
                    "loop_params",
                    "server_host",
                    "server_port",
                    "nccl_port",
                    "silent_output",
                    "decode_log_interval",
                }
            }
            common_server["disable_lookahead_exit"] = server["cdb_config"]["disable_lookahead_exit"]
            common = {
                "server_args": common_server,
                "max_seq_len": info["max_seq_len"],
                "num_pages": info["num_pages"],
                "loop_steps": info["loop_params"]["loop_steps"],
                "dataset_seed": config["seed"],
                "arrival_seed_base": config["arrival_seed"] - repeat,
                "trace_duration_s": trace_duration,
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "extra_request_body": config.get("extra_request_body"),
                "warmup_requests": config["warmup_requests"],
            }
            if json_key(panels.setdefault(name, common)) != json_key(common):
                raise ValueError(f"The {name} panel mixes incompatible execution settings.")
            key = name, backend, refill, min_coda, threshold, rate
            if (*key, repeat) in seen:
                raise ValueError("Duplicate point/repeat record.")
            seen.add((*key, repeat))
            grouped.setdefault(key, []).append(row)
    if not grouped or len(models) != 1:
        raise ValueError("An export must contain measurements for one model.")
    model_key, model, model_path = models.pop()
    workloads: dict[str, list[dict[str, Any]]] = {}
    for (name, backend, refill, min_coda, threshold, rate), rows in grouped.items():
        point = {
            "backend": backend,
            "refill": refill,
            "min_coda_batch_size": min_coda,
            "threshold": threshold,
            "rate_rps": rate,
            "max_num_seqs": panels[name]["server_args"]["max_running_req"],
            "norm_lat_mean_s_per_tok": statistics.mean(
                row["mean_normalized_latency_s_per_token"] for row in rows
            ),
            "gen_tps": statistics.mean(row["output_throughput"] for row in rows),
            "num_requests": rows[0]["completed"],
            "repeats": len(rows),
        }
        for source_key, target_key in (
            ("mean_ttft_ms", "ttft_mean_s"),
            ("p99_ttft_ms", "ttft_p99_s"),
            ("mean_tpot_ms", "tpot_mean_s"),
            ("mean_e2e_latency_ms", "e2e_mean_s"),
            ("p99_e2e_latency_ms", "e2e_p99_s"),
        ):
            values = [row[source_key] for row in rows]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in values
            ):
                raise ValueError(f"Invalid {source_key} summary.")
            point[target_key] = statistics.mean(values) / 1000
        workloads.setdefault(name, []).append(point)
    return {
        "meta": {
            "model_key": model_key,
            "model": model,
            "model_path": model_path,
            "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": [fingerprint(path) for path in paths],
            "timing_conventions": TIMING_CONVENTIONS,
        },
        "workloads": [
            {
                "name": name,
                "configuration": panels[name],
                "recorded_mean_exit_depths": [
                    {
                        "threshold": threshold,
                        "mean_exit_depth": load_recording(
                            *recording, limit=request_limits.get(workload)
                        )[0]["mean_exit_depth"],
                        "num_requests": request_limits.get(workload),
                    }
                    for (workload, threshold), recording in sorted(recordings.items())
                    if workload == name
                ],
                "points": sorted(
                    points,
                    key=lambda point: (
                        point["backend"],
                        point["refill"],
                        point["min_coda_batch_size"],
                        point["threshold"] or 0.0,
                        point["rate_rps"],
                    ),
                ),
            }
            for name, points in sorted(workloads.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "summaries",
        type=Path,
        nargs="+",
        help="Benchmark summary JSONL files; rows are grouped by the workload they replayed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON, committed alongside the paper's plotting script.",
    )
    parser.add_argument(
        "--workload-root",
        type=Path,
        default=Path.cwd(),
        help="Directory that relative replay recording paths resolve against (default: cwd).",
    )
    parser.add_argument(
        "--requests",
        nargs="+",
        default=[],
        metavar="WORKLOAD=N",
        help="Average recorded depths over the first N requests of a workload, e.g. sharegpt=6000.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        limits = parse_request_limits(args.requests, ("sharegpt", "alpaca", "arxiv"))
        results = build_results(args.summaries, args.workload_root, limits)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
