"""Shared code for exporting benchmark summaries into the paper's plotting format.

Mean exit depths cover decode tokens and exclude each request's first output token, which
prefill produces. Benchmarks record this mean directly. Bounds use these decode depths and
component FLOP estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _file_hash(path: Path, mtime_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": _file_hash(path.resolve(), stat.st_mtime_ns, stat.st_size),
    }


def json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class StageFlops:
    """Parameter-count FLOP weights for boundary stages and one recurrent step."""

    f0: int
    fr: int

    def ideal_speedup(self, max_depth: int, mean_depth: float) -> float:
        if not 1 <= mean_depth <= max_depth or self.fr <= 0 or self.f0 < 0:
            raise ValueError("Invalid stage FLOP weights or mean exit depth")
        return (self.f0 + max_depth * self.fr) / (self.f0 + mean_depth * self.fr)

    def to_json_dict(self) -> dict[str, int]:
        return {"f0_params": self.f0, "fr_params": self.fr}


def stage_flops(config: dict[str, Any]) -> StageFlops:
    """Count the dense Ouro/Huginn stage weights without loading model tensors.

    Each decoder layer has QKV/output projections, three MLP matrices, and four RMSNorm weights.
    Embedding lookups and exit gates are excluded, matching the paper's FLOP approximation.
    A tied output head is counted because it still performs the vocabulary projection.
    """
    hidden = positive_int(config["hidden_size"], "hidden_size")
    heads = positive_int(config["num_attention_heads"], "num_attention_heads")
    kv_heads = positive_int(config.get("num_key_value_heads", heads), "num_key_value_heads")
    head_dim = positive_int(config.get("head_dim") or hidden // heads, "head_dim")
    intermediate = positive_int(config["intermediate_size"], "intermediate_size")
    vocab = positive_int(config["vocab_size"], "vocab_size")
    qkv_width = (heads + 2 * kv_heads) * head_dim
    layer = hidden * qkv_width + heads * head_dim * hidden + 3 * hidden * intermediate + 4 * hidden
    boundary = vocab * hidden + hidden
    if config["model_type"] == "ouro":
        return StageFlops(
            boundary, positive_int(config["num_hidden_layers"], "num_hidden_layers") * layer
        )
    if config["model_type"] != "huginn_raven":
        raise ValueError("Throughput bounds support only Ouro and Huginn")
    if config["attention_bias"]:
        layer += qkv_width
    prelude, core, coda = (config[f"num_{phase}_layers"] for phase in ("prelude", "core", "coda"))
    if type(prelude) is not int or type(coda) is not int or min(prelude, coda) < 0:
        raise ValueError("Huginn boundary layer counts must be nonnegative integers")
    positive_int(core, "num_core_layers")
    if prelude + core + coda != config["num_hidden_layers"]:
        raise ValueError("Huginn layer counts disagree with num_hidden_layers")
    adapter = 2 * hidden * hidden + (hidden if config["bias"] else 0)
    return StageFlops(boundary + (prelude + coda) * layer, core * layer + adapter)


@dataclass
class Run:
    source: Path
    row: dict[str, Any]

    @property
    def config(self) -> dict[str, Any]:
        return self.row["config"]

    @property
    def name(self) -> str:
        """Read the workload label from the recorded paper dataset filename."""
        # Replays are stored as <workload>/<model>/<recording>.jsonl.
        parts = Path(self.config["dataset_path"]).parts
        name = parts[-3].lower() if len(parts) >= 3 else None
        if name not in ("sharegpt", "alpaca", "arxiv"):
            raise ValueError(f"{self.source}: dataset path must identify one paper workload")
        return name

    @property
    def num_requests(self) -> int:
        return self.row["successful_requests"]

    @property
    def input_tokens(self) -> int:
        return self.row["total_input_tokens"]

    @property
    def output_tokens(self) -> int:
        return self.row["total_output_tokens"]

    @property
    def backend(self) -> str:
        return self.config["scheduler"]

    @property
    def refill(self) -> bool:
        return bool(self.config["cdb_config"]["refill"])

    @property
    def width(self) -> int:
        return self.config["max_running_req"]

    @property
    def depth(self) -> int:
        return self.config["loop_steps"]

    @property
    def mean_depth(self) -> float:
        """Mean exit depth over decode tokens, excluding each request's first output."""
        return float(self.row["mean_decode_depth"])

    @property
    def threshold(self) -> float | None:
        return self.config["loop_params"]["exit_threshold"] if self.backend == "cdb" else None

    @property
    def series(self) -> str:
        return "cb" if self.backend == "cb" else "refill" if self.refill else "norefill"

    def settings(self, *, across_widths: bool = False) -> dict[str, Any]:
        """Settings that must match across CB/CDB and any requested repeats."""
        ignored = {
            "nccl_port",
            "result_filename",
            "dataset_path",
            "scheduler",
            "cdb_config",
            "loop_params",
            "enable_nvtx",
        }
        if across_widths:
            ignored.update({"max_running_req", "min_free_slots_delay", "cuda_graph_max_bs"})
        return {key: value for key, value in self.config.items() if key not in ignored}


def load_runs(paths: list[Path]) -> list[Run]:
    runs = []
    for source in paths:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"{source}: no benchmark rows")
        for row in rows:
            config = row["config"]
            if row["backend"] != "engine" or config["scheduler"] not in {"cb", "cdb"}:
                raise ValueError(f"{source}: expected CB/CDB engine results")
            if config.get("enable_nvtx"):
                raise ValueError(f"{source}: NVTX-enabled profile results cannot supply throughput")
            if (
                config["dataset_name"] != "custom"
                or config["custom_input_format"] != "token-ids"
                or config["shuffle"]
                or not config["replay_exits"]
                or config["disable_ignore_eos"]
                or config.get("extra_request_body") not in (None, "", "{}")
                or config.get("sharegpt_output_len") is not None
            ):
                raise ValueError(
                    f"{source}: exports require unshuffled, fixed-length token-ID exit replay"
                )
            count = positive_int(row["successful_requests"], "successful_requests")
            if count != config["num_prompts"]:
                raise ValueError(
                    f"{source}: completed request count differs from the requested count"
                )
            depth = positive_int(config["loop_steps"], "loop_steps")
            if config["loop_params"]["loop_steps"] != depth:
                raise ValueError(f"{source}: inconsistent loop depths")
            positive_int(config["max_running_req"], "max_running_req")
            positive_int(row["total_input_tokens"], "total_input_tokens")
            outputs = positive_int(row["total_output_tokens"], "total_output_tokens")
            if outputs <= count:
                raise ValueError(f"{source}: replay runs require outputs beyond the first token")
            wall = float(row["total_latency"])
            if (
                not math.isfinite(wall)
                or wall <= 0
                or not math.isclose(row["output_throughput"], outputs / wall, rel_tol=1e-8)
            ):
                raise ValueError(f"{source}: invalid duration or output throughput")
            run = Run(source.resolve(), row)
            if row.get("mean_decode_depth") is None:
                raise ValueError(f"{source}: benchmark JSON must record mean_decode_depth")
            if not math.isfinite(run.mean_depth) or not 1 <= run.mean_depth <= depth:
                raise ValueError(f"{source}: mean_decode_depth must be between 1 and {depth}")
            if run.backend == "cb":
                if config["loop_params"]["exit_threshold"] is not None:
                    raise ValueError(f"{source}: CB baseline must run full depth")
            elif run.threshold is None or not math.isfinite(float(run.threshold)):
                raise ValueError(f"{source}: CDB rows must record a finite exit threshold")
            runs.append(run)
    return runs


def model_info(runs: list[Run], model_config: Path | None) -> tuple[StageFlops, dict[str, Any]]:
    models = {run.config["model_path"] for run in runs}
    overrides = {json_key(json.loads(run.config["json_model_override_args"])) for run in runs}
    if len(models) != 1 or len(overrides) != 1:
        raise ValueError("Export one model and one model configuration at a time")
    model = models.pop()
    if model_config is None:
        local = Path(model) / "config.json"
        if local.is_file():
            model_config = local
        else:
            from huggingface_hub import hf_hub_download

            model_config = Path(hf_hub_download(model, "config.json"))
    config = json.loads(model_config.read_text())
    config.update(json.loads(overrides.pop()))
    return stage_flops(config), {
        **fingerprint(model_config),
        "effective_config": config,
        "scope": "model configuration at export time",
    }


def validate_panel(runs: list[Run]) -> None:
    if len({(run.num_requests, run.input_tokens, run.output_tokens) for run in runs}) != 1:
        raise ValueError("A panel mixes request counts or token totals")
    if len({json_key(run.settings(across_widths=True)) for run in runs}) != 1:
        raise ValueError(
            "A panel mixes engine or benchmark settings beyond width and admission threshold"
        )
    cdb_timing = {
        run.config["cdb_config"]["disable_lookahead_exit"] for run in runs if run.backend == "cdb"
    }
    if len(cdb_timing) > 1:
        raise ValueError("A panel mixes immediate and delayed exit consumption")
    for threshold in {run.threshold for run in runs if run.backend == "cdb"}:
        means = [
            run.mean_depth for run in runs if run.backend == "cdb" and run.threshold == threshold
        ]
        if not all(math.isclose(value, means[0], rel_tol=1e-10) for value in means):
            raise ValueError(f"CDB rows at q={threshold} disagree on mean_decode_depth")
    for width in {run.width for run in runs}:
        if len({json_key(run.settings()) for run in runs if run.width == width}) != 1:
            raise ValueError(f"Width {width} has mismatched CB/CDB admission or engine settings")
    if any(run.row.get("num_retracted_reqs", 0) for run in runs):
        logger.warning("Panel contains retractions; inspect the exported pressure settings")


def aggregate(runs: list[Run], flops: StageFlops) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, float | None], list[Run]] = {}
    for run in runs:
        groups.setdefault((run.series, run.width, run.threshold), []).append(run)
    points = []
    for group in groups.values():
        run = group[0]
        coda = {item.config["cdb_config"]["min_coda_batch_size"] for item in group}
        if run.series == "refill" and len(coda) != 1:
            raise ValueError("A sweep point mixes minimum coda batch sizes")
        rates = [item.row["output_throughput"] for item in group]
        point = {
            "backend": run.backend,
            "refill": run.refill,
            "threshold": run.threshold,
            "depth": run.depth,
            "width": run.width,
            "min_coda_batch_size": run.config["cdb_config"]["min_coda_batch_size"],
            "gen_tps": statistics.mean(rates),
            "gen_tps_std": statistics.stdev(rates) if len(rates) > 1 else None,
            "wall_s": statistics.mean(item.row["total_latency"] for item in group),
            "preemptions": statistics.mean(item.row["num_retracted_reqs"] for item in group),
            "flop_bound": (
                None if run.backend == "cb" else flops.ideal_speedup(run.depth, run.mean_depth)
            ),
            "mean_exit_depth": None if run.backend == "cb" else run.mean_depth,
            "repeats": len(group),
        }
        points.append(point)
    return points


def panel_metadata(runs: list[Run], flops: StageFlops) -> dict[str, Any]:
    first = runs[0]
    return {
        "name": first.name,
        "num_requests": first.num_requests,
        "source_summaries": sorted({str(run.source) for run in runs}),
        "stage_flops": flops.to_json_dict(),
        "bound_basis": "decode-token mean exit depth",
        "measurement_configs": [
            json.loads(value) for value in sorted({json_key(run.settings()) for run in runs})
        ],
    }


def prefill_adjusted_speedup(bound: float, fraction: float) -> float:
    """End-to-end bound from the decode bound and the CB GPU prefill fraction.

    Prefill produces each request's first output, so the decode bound covers the remaining outputs.
    """
    if not 0 <= fraction <= 1 or not math.isfinite(bound) or bound <= 0:
        raise ValueError("Invalid prefill fraction or FLOP bound")
    return 1 / (fraction + (1 - fraction) / bound)


def cb_timing(runs: list[Run]) -> dict[str, Any] | None:
    """Average the CB CUDA event timings across repeats when available."""
    recorded = [run.row.get("cuda_event_timing") for run in runs]
    if all(value is None for value in recorded):
        return None
    if any(value is None for value in recorded):
        raise ValueError("CB repeats mix runs with and without CUDA event timing")
    prefill = statistics.mean(value["prefill_s"] for value in recorded)
    decode = statistics.mean(value["decode_s"] for value in recorded)
    if not all(math.isfinite(value) and value > 0 for value in (prefill, decode)):
        raise ValueError("CB event timings must be finite and positive")
    return {
        "method": "cuda_events",
        "prefill_s": prefill,
        "decode_s": decode,
        "prefill_fraction": prefill / (prefill + decode),
    }


def b_star_at(path: Path, runs: list[Run], mean_context: float) -> dict[str, Any]:
    """Read B* at the nearest log-scale context to the mean sequence length."""
    export = json.loads(path.read_text())
    run = runs[0]
    if export["model_id"] != run.config["model_path"] or export["max_depth"] != run.depth:
        raise ValueError(f"{path}: B* belongs to a different model or loop-depth budget")
    metadata = export["export_metadata"]
    config = metadata["config"]
    for key in ("attention_backend", "loop_cache_policy", "page_size", "dtype"):
        if config[key] != run.config[key]:
            raise ValueError(f"{path}: B* and throughput disagree on {key}")
    overrides = config["json_model_override_args"]
    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    if overrides != json.loads(run.config["json_model_override_args"]):
        raise ValueError(f"{path}: B* and throughput use different model overrides")
    fit = min(
        export["contexts"], key=lambda item: abs(math.log(item["context_length"] / mean_context))
    )
    value = float(fit["b_star"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}: invalid B* fit")
    if value > fit["batch_range"][1]:
        logger.warning(
            "B* %.1f at context %s exceeds the largest measured batch", value, fit["context_length"]
        )
    return {
        "b_star": value,
        "b_star_approx": round(value),
        "b_star_context_length": fit["context_length"],
        "b_star_source": fingerprint(path),
    }


def prepare(
    paths: list[Path], model_config: Path | None
) -> tuple[dict[str, list[Run]], StageFlops, dict[str, Any]]:
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("A summary file was supplied more than once")
    runs = load_runs(paths)
    flops, model = model_info(runs, model_config)
    grouped: dict[str, list[Run]] = {}
    for run in runs:
        grouped.setdefault(run.name, []).append(run)
    for group in grouped.values():
        validate_panel(group)
    meta = {
        "model": runs[0].config["model_path"],
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [fingerprint(path) for path in paths],
        "model_configuration": model,
        "timing_scope": "whole generation, including IPC, detokenization, and result collection",
    }
    return grouped, flops, meta


def add_cli_args(parser: argparse.ArgumentParser) -> None:
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
        "--model-config",
        type=Path,
        help="Checkpoint config.json; otherwise resolve it from model_path.",
    )


def write_export(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    for panel in results["workloads"]:
        print(f"  {panel['name']}: {panel['num_requests']} requests")
