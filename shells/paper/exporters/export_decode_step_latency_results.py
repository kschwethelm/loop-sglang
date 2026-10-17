"""Export one_batch sweeps in the JSON format used by the paper latency plots.

For each (context length, batch size), fit decode latency against loop depth to
separate recurrent-step latency from the fixed prelude and coda overhead.
Then fit recurrent-step latency against batch size at each context length.
The second fit and figure fields follow the research repository's exporter.
Only batches with all requested regression depths contribute to the figure.
Only GPU model medians are exported.
These synchronized timings retain their measurement scope in export metadata.

Example:
    uv run python shells/paper/exporters/export_decode_step_latency_results.py \
        outputs/decode_step_latency/ouro_shared.jsonl \
        --depths 1 2 3 4 --output docs/paper/figs/decode_latency/decode_step_latency_ouro_loopsgl.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} holds no sweep rows")
    return rows


def fit_linear(x: list[int], y: list[float]) -> tuple[float, float, float]:
    """Return slope, intercept, and R² for an additive linear fit."""
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(xs) < 2 or len(xs) != len(ys) or len(set(x)) != len(x):
        raise ValueError("A linear fit needs at least two distinct, aligned measurements")
    if not np.isfinite(ys).all():
        raise ValueError("Latencies must be finite")
    slope, intercept = np.polyfit(xs, ys, 1)
    residuals = ys - (slope * xs + intercept)
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - float((residuals**2).sum()) / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), r2


def build_results(rows: list[dict[str, Any]], depths: list[int]) -> dict[str, Any]:
    if not rows:
        raise ValueError("The sweep holds no rows")
    if len(depths) < 2 or len(set(depths)) != len(depths) or min(depths) <= 0:
        raise ValueError("Provide at least two distinct positive regression depths")
    depths = sorted(depths)
    reference: dict[str, Any] | None = None
    grouped: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    page_counts: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        config = dict(row["config"])
        depth = config.pop("loop_steps")
        page_counts[depth].add(config.pop("num_pages"))
        if depth not in depths:
            raise ValueError(f"Unexpected loop depth {depth}; requested {depths}")
        if config.get("profile"):
            raise ValueError("Profiled measurements cannot be used for latency fits")
        if "json_model_override_args" in config:
            config["json_model_override_args"] = json.loads(config["json_model_override_args"])
        metadata = {
            "config": config,
            "output_len": row["output_len"],
            "cuda_graph": row["cuda_graph"],
        }
        if reference is None:
            reference = metadata
        elif metadata != reference:
            raise ValueError("Rows disagree on sweep settings; export each campaign separately")
        batch, context = row["batch_size"], row["input_len"]
        if batch <= 0 or context <= 0 or row["input_lens"] != [context] * batch:
            raise ValueError("Paper plots require positive, uniform batch context lengths")
        if row["padded_batch_size"] != batch:
            raise ValueError("Paper plots require unpadded batch measurements")
        samples = row["gpu_decode_latencies"]
        if (
            not samples
            or len(samples) != row["output_len"] - 1
            or any(not math.isfinite(value) or value <= 0 for value in samples)
        ):
            raise ValueError(f"Invalid decode timing samples at L={context}, B={batch}, r={depth}")
        expected_contexts = [[context + index] * 2 for index in range(len(samples))]
        if row["decode_context_lens"] != expected_contexts:
            raise ValueError(f"Unexpected decode context progression at L={context}, B={batch}")
        median = float(np.median(samples))
        if not math.isclose(median, row["median_gpu_decode_latency"], rel_tol=1e-7):
            raise ValueError(f"Recorded median disagrees with samples at L={context}, B={batch}")
        point = grouped[context, batch]
        if depth in point:
            raise ValueError(f"Duplicate measurement at L={context}, B={batch}, r={depth}")
        point[depth] = median * 1000

    points: dict[int, list[dict[str, Any]]] = defaultdict(list)
    excluded = []
    depth_fits = []
    for (context, batch), measurements in sorted(grouped.items()):
        missing = sorted(set(depths) - measurements.keys())
        if missing:
            excluded.append(
                {"context_length": context, "batch_size": batch, "missing_depths": missing}
            )
            logger.warning("Excluding L=%s B=%s: missing depths %s", context, batch, missing)
            continue
        values = [measurements[depth] for depth in depths]
        core, overhead, r2 = fit_linear(depths, values)
        if core <= 0:
            raise ValueError(f"Nonpositive recurrent-step latency at L={context}, B={batch}")
        points[context].append({"batch_size": batch, "core_step_ms": core})
        depth_fits.append(
            {
                "context_length": context,
                "batch_size": batch,
                "depth_step_ms": values,
                "core_step_ms": core,
                "overhead_ms": overhead,
                "regression_r2": r2,
            }
        )

    contexts = []
    for context in sorted({context for context, _ in grouped}):
        context_points = points[context]
        if len(context_points) < 2:
            raise ValueError(f"Context {context} needs at least two batches with all depths")
        batches = [point["batch_size"] for point in context_points]
        latencies = [point["core_step_ms"] for point in context_points]
        per_seq, floor, r2 = fit_linear(batches, latencies)
        if per_seq <= 0 or floor <= 0:
            raise ValueError(f"Context {context} needs positive batch-fit coefficients")
        contexts.append(
            {
                "context_length": context,
                "points": context_points,
                "floor_ms": floor,
                "per_seq_ms": per_seq,
                "b_star": floor / per_seq,
                "num_points": len(context_points),
                "batch_range": [batches[0], batches[-1]],
                "fit_r2": r2,
            }
        )
    assert reference is not None
    return {
        "model_id": reference["config"]["model_path"],
        "device_name": reference["config"]["device_name"],
        "max_depth": max(depths),
        "floor_ms": sum(context["floor_ms"] for context in contexts) / len(contexts),
        "contexts": contexts,
        "export_metadata": {
            **reference,
            "metric": "gpu",
            "statistic": "median",
            "regression_depths": depths,
            "num_pages_by_depth": {
                depth: sorted(counts) for depth, counts in sorted(page_counts.items())
            },
            "depth_fits": depth_fits,
            "excluded_points": excluded,
        },
    }


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sweep", type=Path, help="Raw JSONL produced by benchmark.one_batch.")
    parser.add_argument("--output", type=Path, required=True, help="Paper-format figure JSON.")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        required=True,
        help="Loop depths required at every fitted batch.",
    )
    parser.add_argument(
        "--model-id", help="Model identifier for the figure; defaults to the recorded model path."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        results = build_results(load_rows(args.sweep), args.depths)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    results["export_metadata"]["source_file"] = str(args.sweep.resolve())
    if args.model_id:
        results["model_id"] = args.model_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")
    for context in results["contexts"]:
        low, high = context["batch_range"]
        print(
            f"L={context['context_length']}: {context['num_points']} points (B {low}-{high}), "
            f"floor {context['floor_ms']:.3f} ms, B* {context['b_star']:.1f}, R² {context['fit_r2']:.5f}"
        )
        if context["b_star"] > high:
            logger.warning(
                "L=%s: B* exceeds the largest measured batch %s", context["context_length"], high
            )


if __name__ == "__main__":
    cli_main()
