r"""Export a maximum decode batch size sweep for the paper plots.

Reads raw summary JSONL written by ``python -m loopsgl.benchmark.offline_throughput``.
CDB throughput is normalized against the full-depth CB row, with one panel per workload.
The exit threshold is fixed while the maximum decode batch size varies.
The theoretical maximum decode speedup (FLOP bound) is computed from the mean exit depth over
decode tokens.
CB CUDA event timings provide the prefill fraction used to compute the end-to-end bound.
When a decode-step latency export is supplied, its fitted ``B*`` is recorded.

The sweep is produced by ``shells/paper/throughput_width-sweep.sh``.
The decode-step latency results are produced by ``shells/paper/benchmark_decode_step_latency.sh``.

Example:
    uv run python shells/paper/exporters/export_width_sweep_results.py \
        outputs/width-sweep/width-sweep_ouro_alpaca_shared.jsonl \
        outputs/width-sweep/width-sweep_ouro_sharegpt_shared.jsonl \
        outputs/width-sweep/width-sweep_ouro_arxiv_shared.jsonl \
        --b-star-from docs/paper/figs/decode_latency/decode_step_latency_ouro_loopsgl.json \
        --output docs/paper/figs/throughput/width_sweep_ouro_loopsgl.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from _throughput import (
    add_cli_args,
    aggregate,
    b_star_at,
    cb_timing,
    panel_metadata,
    prefill_adjusted_speedup,
    prepare,
    write_export,
)


def build_results(
    paths: list[Path],
    *,
    model_config: Path | None = None,
    b_star_source: Path | None = None,
) -> dict[str, Any]:
    """One panel per workload, each assembled from its matching benchmark summaries."""
    grouped, flops, meta = prepare(paths, model_config)
    workloads = []
    for _, runs in sorted(grouped.items()):
        thresholds = {run.threshold for run in runs if run.backend == "cdb"}
        if len(thresholds) != 1:
            raise ValueError("Width sweeps must hold the exit threshold fixed")
        points = aggregate(runs, flops)
        widths = sorted({run.width for run in runs})
        series: dict[str, dict[int, dict[str, Any]]] = {
            name: {} for name in ("cb", "norefill", "refill")
        }
        for point in points:
            name = "cb" if point["backend"] == "cb" else "refill" if point["refill"] else "norefill"
            series[name][point["width"]] = point
        for name, measured in series.items():
            if set(measured) != set(widths):
                raise ValueError(
                    f"Width sweep is missing {name} points at {sorted(set(widths) - measured.keys())}"
                )
        bound = next(point["flop_bound"] for point in points if point["backend"] == "cdb")
        first = runs[0]
        mean_context = (first.input_tokens + first.output_tokens) / first.num_requests
        entry = {
            **panel_metadata(runs, flops),
            "threshold": thresholds.pop(),
            "widths": widths,
            "flop_bound": bound,
            "mean_sequence_length": round(mean_context),
            "min_coda_batch_sizes": [
                series["refill"][width]["min_coda_batch_size"] for width in widths
            ],
        }
        for name, measured in series.items():
            entry[name] = [measured[width]["gen_tps"] for width in widths]
            entry[f"{name}_std"] = [measured[width]["gen_tps_std"] for width in widths]
            entry[f"{name}_repeats"] = [measured[width]["repeats"] for width in widths]
            entry[f"{name}_preemptions"] = [measured[width]["preemptions"] for width in widths]
            if name != "cb":
                entry[f"{name}_over_cb"] = [
                    measured[width]["gen_tps"] / series["cb"][width]["gen_tps"] for width in widths
                ]
        timings = [
            cb_timing([run for run in runs if run.backend == "cb" and run.width == width])
            for width in widths
        ]
        if any(timing is not None for timing in timings):
            if not all(timing is not None for timing in timings):
                raise ValueError("Width sweeps require CB timings at every width")
            entry["bound_timings"] = timings
            entry["e2e_bound"] = [
                prefill_adjusted_speedup(bound, timing["prefill_fraction"]) for timing in timings
            ]
        if b_star_source is not None:
            entry.update(b_star_at(b_star_source, runs, mean_context))
        workloads.append(entry)
    return {"meta": meta, "workloads": workloads}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_cli_args(parser)
    parser.add_argument(
        "--b-star-from",
        type=Path,
        help="Decode-step latency export containing a fitted balance batch B*.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        results = build_results(
            args.summaries,
            model_config=args.model_config,
            b_star_source=args.b_star_from,
        )
        write_export(results, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
