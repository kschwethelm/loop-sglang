r"""Export the last-exited KV comparison on Ouro into the paper's plotting format.

Compares adaptive CDB with refill under three KV caches: shared, last-exited with freed slots
(``last_exited``), and last-exited with copied states (``last_exited_copy``).
The width comparison reads ``throughput_width-sweep.sh`` summaries.
It uses the refill rows of the shared, last-exited, and copying sweeps, plus the CB rows of the
shared and depth-indexed sweeps.
The memory comparison reads ``ablations/benchmark_memory.sh`` summaries whose runs together cover
CB with shared and depth-indexed KV and CDB with all three caches.
Shared CDB is normalized to CB with shared KV, and both last-exited caches to CB with
depth-indexed KV, at the same width or KV budget.
Decode batch medians follow ``export_memory_results.py``.

The copying cache is implemented on the ``feat/last-exited-copy`` branch.
Its runs use ``CACHE_POLICY=last_exited_copy BACKENDS=cdb-refill`` for the width sweeps and
``RUNS=cdb-refill:last_exited_copy`` for the memory sweeps.
Copy the resulting width summaries into ``outputs/width-sweep/`` and the memory summaries,
together with their ``logs/`` directory, into ``outputs/ablations/memory/copy/``.

Example:
    uv run python shells/paper/exporters/export_last_exited_results.py \
        --width outputs/width-sweep/width-sweep_ouro_*.jsonl \
        --memory outputs/ablations/memory/memory_ouro_*gib.jsonl \
            outputs/ablations/memory/copy/memory_ouro_*gib.jsonl \
        --output docs/paper/figs/throughput/last_exited_ouro_loopsgl.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _throughput import Run, fingerprint, json_key, load_runs
from export_memory_results import decode_batch_sizes, memory_budget

THRESHOLD = 0.0001
LOOP_STEPS = 4
PAGE_SIZES = {"shared": 16, "depth_indexed": 16, "last_exited": 1, "last_exited_copy": 16}
# Role in the export -> (scheduler, KV policy) of the run that provides it.
ROLES = {
    "cb": ("cb", "depth_indexed"),
    "shared_cb": ("cb", "shared"),
    "shared": ("cdb", "shared"),
    "freeing": ("cdb", "last_exited"),
    "copying": ("cdb", "last_exited_copy"),
}
POLICY_FIELDS = {"loop_cache_policy", "page_size", "max_total_tokens"}
IGNORED_FIELDS = POLICY_FIELDS | {
    "nccl_port",
    "result_filename",
    "scheduler",
    "loop_params",
    "cdb_config",
}


def role(run: Run) -> str | None:
    """Map a run to its export role, skipping configurations the comparison does not use."""
    if run.backend == "cdb" and not run.refill:
        return None
    key = (run.backend, run.config["loop_cache_policy"])
    return next((name for name, value in ROLES.items() if value == key), None)


def validate_group(runs: list[Run], label: str) -> None:
    """Require one replayed population and matching settings apart from the KV cache."""
    if len({(run.num_requests, run.input_tokens, run.output_tokens) for run in runs}) != 1:
        raise ValueError(f"{label}: runs mix request counts or token totals")
    settings = {
        json_key({key: value for key, value in run.config.items() if key not in IGNORED_FIELDS})
        for run in runs
    }
    if len(settings) != 1:
        raise ValueError(f"{label}: runs mix engine or benchmark settings beyond the KV cache")
    for run in runs:
        policy = run.config["loop_cache_policy"]
        if run.depth != LOOP_STEPS:
            raise ValueError(f"{label}: expected {LOOP_STEPS} loop steps")
        if run.config["page_size"] != PAGE_SIZES[policy]:
            raise ValueError(f"{label}: expected page size {PAGE_SIZES[policy]} for {policy} KV")
        if run.backend == "cdb" and (
            run.threshold != THRESHOLD or run.config["cdb_config"]["min_coda_batch_size"] != 1
        ):
            raise ValueError(f"{label}: CDB requires threshold {THRESHOLD} and minimum coda 1")
    cdb = [run for run in runs if run.backend == "cdb"]
    if not all(math.isclose(run.mean_depth, cdb[0].mean_depth, rel_tol=1e-10) for run in cdb):
        raise ValueError(f"{label}: CDB runs disagree on replay mean_decode_depth")


def by_role(runs: list[Run], label: str) -> dict[str, Run]:
    """Index runs by role and reject duplicate measurements."""
    counts = Counter(role(run) for run in runs)
    duplicates = sorted(name for name, count in counts.items() if name and count > 1)
    if duplicates:
        raise ValueError(f"{label}: more than one row for {', '.join(duplicates)}")
    return {name: run for run in runs if (name := role(run))}


def width_sweep(runs: list[Run]) -> list[dict[str, Any]]:
    """One entry per workload with matched last-exited widths and the full shared range."""
    grouped: dict[str, dict[int, list[Run]]] = {}
    for run in runs:
        if role(run):
            grouped.setdefault(run.name, {}).setdefault(run.width, []).append(run)
    workloads = []
    for name, widths in sorted(grouped.items()):
        points = []
        for width, group in sorted(widths.items()):
            label = f"{name} W={width}"
            validate_group(group, label)
            roles = by_role(group, label)
            if not {"shared_cb", "shared"} <= roles.keys():
                raise ValueError(f"{label}: shared CB and shared CDB are required at every width")
            last_exited = {"cb", "freeing", "copying"} & roles.keys()
            if last_exited and last_exited != {"cb", "freeing", "copying"}:
                raise ValueError(f"{label}: last-exited caches and depth-indexed CB must match")
            points.append(
                {"width": width}
                | {
                    f"{key}_gen_tps": roles[key].row["output_throughput"] if key in roles else None
                    for key in ROLES
                }
            )
        measured = [point["width"] for point in points if point["cb_gen_tps"] is not None]
        if not measured or measured != [point["width"] for point in points[: len(measured)]]:
            raise ValueError(f"{name}: last-exited widths must start at the smallest shared width")
        workloads.append({"name": name, "points": points})
    return workloads


def memory_sweep(runs: list[Run]) -> list[dict[str, Any]]:
    """One entry per workload with all five configurations at every KV budget."""
    grouped: dict[str, dict[int, list[Run]]] = {}
    for run in runs:
        if role(run):
            grouped.setdefault(run.name, {}).setdefault(memory_budget(run), []).append(run)
    workloads = []
    for name, budgets in sorted(grouped.items()):
        points = []
        for budget, group in sorted(budgets.items()):
            label = f"{name} {budget} GiB"
            validate_group(group, label)
            roles = by_role(group, label)
            if roles.keys() != ROLES.keys():
                raise ValueError(f"{label}: expected {', '.join(ROLES)}")
            if group[0].config["retraction_mode"] != "none" or any(
                run.row["num_retracted_reqs"] != 0 for run in group
            ):
                raise ValueError(f"{label}: memory runs require zero retractions")
            point: dict[str, Any] = {"memory_budget_gib": budget}
            point |= {f"{key}_gen_tps": run.row["output_throughput"] for key, run in roles.items()}
            point |= {
                f"{key}_median_decode_batch_size": statistics.median(decode_batch_sizes(roles[key]))
                for key in ("shared", "copying", "freeing")
            }
            points.append(point)
        workloads.append({"name": name, "points": points})
    return workloads


def build_results(width_paths: list[Path], memory_paths: list[Path]) -> dict[str, Any]:
    paths = width_paths + memory_paths
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("A summary file was supplied more than once")
    width_runs = load_runs(width_paths)
    memory_runs = load_runs(memory_paths)
    if {run.config["model_path"] for run in width_runs + memory_runs} != {
        width_runs[0].config["model_path"]
    }:
        raise ValueError("All summaries must come from one model")
    return {
        "meta": {
            "model": width_runs[0].config["model_path"],
            "loop_steps": LOOP_STEPS,
            "exit_threshold": THRESHOLD,
            "refill": True,
            "measurements_per_point": 1,
            "page_sizes": {
                "shared": PAGE_SIZES["shared"],
                "copying": PAGE_SIZES["last_exited_copy"],
                "freeing": PAGE_SIZES["last_exited"],
            },
            "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": [fingerprint(path) for path in paths],
            "depth_basis": "decode-token mean exit depth",
            "decode_batch_scope": (
                "Median CDB recurrent-core batch size with requests queued. "
                "Warmup and queue drain are excluded."
            ),
            "width_reference": (
                "Full-depth CB at the same width, with shared KV for shared CDB and "
                "depth-indexed KV for last-exited CDB."
            ),
            "memory_reference": (
                "Full-depth CB at the same KV budget, with shared KV for shared CDB and "
                "depth-indexed KV for last-exited CDB."
            ),
        },
        "width_sweep": width_sweep(width_runs),
        "memory_sweep": memory_sweep(memory_runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--width", type=Path, nargs="+", required=True, help="Width summaries.")
    parser.add_argument("--memory", type=Path, nargs="+", required=True, help="Memory summaries.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON, committed alongside the paper's plotting script.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        results = build_results(args.width, args.memory)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
