r"""Export an exit-threshold sweep into a JSON file for the paper plots.

Reads raw summary JSONL written by ``python -m loopsgl.benchmark.offline_throughput``.
CDB throughput is normalized against the full-depth CB row, with one panel per workload.
The theoretical maximum decode speedup (FLOP bound) is computed from each threshold's mean exit
depth over decode tokens.
CB CUDA event timings provide the prefill fraction used to compute the end-to-end bound.

The sweep is produced by ``shells/paper/throughput_exit-sweep.sh``.

Example:
    uv run python shells/paper/exporters/export_exit_sweep_results.py \
        outputs/exit-sweep/exit-sweep_ouro_alpaca_shared.jsonl \
        outputs/exit-sweep/exit-sweep_ouro_sharegpt_shared.jsonl \
        outputs/exit-sweep/exit-sweep_ouro_arxiv_shared.jsonl \
        --output docs/paper/figs/throughput/exit_sweep_ouro_loopsgl.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from _throughput import (
    add_cli_args,
    aggregate,
    cb_timing,
    panel_metadata,
    prefill_adjusted_speedup,
    prepare,
    write_export,
)


def build_results(paths: list[Path], *, model_config: Path | None = None) -> dict[str, Any]:
    """One panel per workload, each assembled from its matching benchmark summaries."""
    grouped, flops, meta = prepare(paths, model_config)
    workloads = []
    for _, runs in sorted(grouped.items()):
        if len({run.width for run in runs}) != 1:
            raise ValueError("Exit sweeps must hold maximum decode batch size fixed")
        if (
            len(
                {
                    run.config["cdb_config"]["min_coda_batch_size"]
                    for run in runs
                    if run.series == "refill"
                }
            )
            > 1
        ):
            raise ValueError("Exit sweeps must hold minimum coda batch size fixed")
        points = aggregate(runs, flops)
        baselines = [point for point in points if point["backend"] == "cb"]
        if len(baselines) != 1:
            raise ValueError("An exit sweep needs exactly one full-depth CB baseline")
        thresholds = [
            {
                point["threshold"]
                for point in points
                if point["backend"] == "cdb" and point["refill"] == refill
            }
            for refill in (False, True)
        ]
        if not thresholds[0] or thresholds[0] != thresholds[1]:
            raise ValueError("Exit sweeps require matching refill/no-refill threshold points")
        baseline = baselines[0]
        for point in points:
            point["speedup"] = point["gen_tps"] / baseline["gen_tps"]
        points.sort(
            key=lambda point: (point["backend"] != "cb", point["refill"], point["threshold"] or 0)
        )
        bound_points = [
            point for point in points if point["backend"] == "cdb" and not point["refill"]
        ]
        curve = {
            "thresholds": [point["threshold"] for point in bound_points],
            "flop_bound": [point["flop_bound"] for point in bound_points],
        }
        entry = {
            **panel_metadata(runs, flops),
            "baseline_gen_tps": baseline["gen_tps"],
            "points": points,
            "bound_curve": curve,
        }
        cb_runs = [run for run in runs if run.backend == "cb"]
        timing = cb_timing(cb_runs)
        if timing is not None:
            entry["bound_timing"] = timing
            curve["e2e_bound"] = [
                prefill_adjusted_speedup(value, timing["prefill_fraction"])
                for value in curve["flop_bound"]
            ]
        workloads.append(entry)
    return {"meta": meta, "workloads": workloads}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_cli_args(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        results = build_results(args.summaries, model_config=args.model_config)
        write_export(results, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
