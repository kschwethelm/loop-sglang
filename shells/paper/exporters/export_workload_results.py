r"""Export workload lengths, exit distributions, and gate-threshold comparisons.

Reads forced-decoding JSONL recordings from ``shells/paper/workloads/record_workload.sh``.
Counts and mean depths cover decode tokens, excluding the first output token.
The exports feed the paper's workload figures and mean-depth and gate-threshold tables.
``--requests`` restricts every recording and length distribution to the first requests that the
benchmarks replay. Length CDFs use tokenized Ouro prompts and full reference responses.

Example:
    uv run python shells/paper/exporters/export_workload_results.py \
        --workload-dir outputs/workloads \
        --output docs/paper/figs/workloads/workload_exit_distributions.json \
        --length-output docs/paper/figs/workloads/workload_length_distributions.json
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from _workloads import load_recording, parse_request_limits

NUM_QUANTILES = 256
MAX_MODEL_LEN = 16384
WORKLOADS = (("ShareGPT", "sharegpt"), ("Alpaca", "alpaca"), ("ArXiv", "arxiv"))
MODEL_SETTINGS = {
    "ouro": {
        "max_depth": 4,
        "cache_policy": "last_exited",
        "gate_type": "early_exit",
        "exit_mode": "lookahead",
        "thresholds": (0.0001, 0.06, 0.2, 0.4, 1.0),
        "control_threshold": 1.0,
    },
    "huginn": {
        "max_depth": 16,
        "cache_policy": "shared",
        "gate_type": "convergence",
        "exit_mode": "lookahead",
        "thresholds": (0.14, 0.06, 0.03, 0.0),
        "control_threshold": 0.0,
    },
}
GATE_COMPARISONS = {
    "ouro": (
        ("Immediate", "early_exit", "immediate", (0.2, 0.4, 0.5, 0.7)),
        ("Lookahead", "early_exit", "lookahead", (0.0001, 0.06, 0.2, 0.4)),
        ("Trained lookahead", "lookahead", "lookahead", (0.2, 0.4, 0.5, 0.7)),
    ),
    "huginn": (
        ("Immediate", "convergence", "immediate", (0.1, 0.044, 0.023)),
        ("Lookahead", "convergence", "lookahead", (0.14, 0.06, 0.03)),
    ),
}


def build_length_distributions(
    workload_dir: Path, request_limits: dict[str, int] | None = None
) -> dict[str, Any]:
    """Export empirical length quantiles from the first requests of the tokenized Ouro workloads."""
    request_limits = request_limits or {}
    quantiles = np.linspace(0.0, 1.0, NUM_QUANTILES)
    workloads = []
    for label, dataset in WORKLOADS:
        path = workload_dir / "tokenized_base" / dataset / "ouro.jsonl"
        if not path.exists():
            workloads.append(
                {
                    "label": label,
                    "num_requests": None,
                    "input": {"values": None},
                    "output": {"values": None},
                }
            )
            continue
        input_lengths = []
        output_lengths = []
        ids = set()
        limit = request_limits.get(dataset)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(islice(handle, limit), 1):
                row = json.loads(line)
                request_id = row["id"]
                if not isinstance(request_id, str) or not request_id or request_id in ids:
                    raise ValueError(f"{path}:{line_number}: invalid or duplicate request ID")
                if not all(
                    isinstance(row[key], list) and row[key] for key in ("input_ids", "output_ids")
                ):
                    raise ValueError(f"{path}:{line_number}: missing prompt or output tokens")
                prompt, output = len(row["input_ids"]), len(row["output_ids"])
                if prompt + output > MAX_MODEL_LEN:
                    raise ValueError(f"{path}:{line_number}: request exceeds the context limit")
                ids.add(request_id)
                input_lengths.append(prompt)
                output_lengths.append(output)
        if limit is not None and len(ids) != limit:
            raise ValueError(f"{path}: holds {len(ids)} requests, fewer than {limit}")
        if not ids:
            raise ValueError(f"{path}: no tokenized requests")
        workloads.append(
            {
                "label": label,
                "num_requests": len(ids),
                "input": {"values": np.quantile(input_lengths, quantiles).tolist()},
                "output": {"values": np.quantile(output_lengths, quantiles).tolist()},
            }
        )
    return {
        "quantiles": quantiles.tolist(),
        "max_model_len": MAX_MODEL_LEN,
        "request_limits": request_limits,
        "workloads": workloads,
    }


def build_results(
    workload_dir: Path, request_limits: dict[str, int] | None = None
) -> dict[str, Any]:
    """Build distributions and matched-threshold rows from a shared set of recordings."""
    request_limits = request_limits or {}
    if not workload_dir.is_dir():
        raise ValueError(f"Workload directory does not exist: {workload_dir}")
    recordings: dict[Path, dict[str, Any]] = {}
    references: dict[tuple[str, str], Path] = {}
    request_ids: dict[str, list[str]] = {}

    def point_for(
        model: str, dataset: str, gate: str, mode: str, threshold: float
    ) -> dict[str, Any]:
        settings = MODEL_SETTINGS[model]
        path = (
            workload_dir
            / dataset
            / model
            / (f"{settings['cache_policy']}_{gate}_{mode}_q{threshold}.jsonl")
        )
        if not path.exists():
            return {
                "threshold": threshold,
                "exit_counts": None,
                "exit_fractions": None,
                "mean_exit_depth": None,
            }
        if path not in recordings:
            stats, ids = load_recording(
                path,
                settings["max_depth"],
                references.get((model, dataset)),
                request_limits.get(dataset),
            )
            if dataset in request_ids and ids != request_ids[dataset]:
                raise ValueError(f"{path}: request IDs or order differ across models")
            recordings[path] = stats
            references.setdefault((model, dataset), path)
            request_ids.setdefault(dataset, ids)
        return {"threshold": threshold, **recordings[path]}

    models = {}
    for model, settings in MODEL_SETTINGS.items():
        panels = [
            {
                "label": label,
                "points": [
                    point_for(
                        model, dataset, settings["gate_type"], settings["exit_mode"], threshold
                    )
                    for threshold in settings["thresholds"]
                ],
            }
            for label, dataset in WORKLOADS
        ]
        reached = [
            depth
            for panel in panels
            for point in panel["points"]
            for depth, count in enumerate(point["exit_counts"] or [], 1)
            if count
        ]
        models[model] = {
            **settings,
            "thresholds": list(settings["thresholds"]),
            "min_exit_depth": min(reached, default=1),
            "workloads": panels,
        }
    comparisons = {}
    for model, variants in GATE_COMPARISONS.items():
        gates = []
        for label, gate, mode, thresholds in variants:
            points = [
                point_for(model, "sharegpt", gate, mode, threshold) for threshold in thresholds
            ]
            gates.append(
                {
                    "label": label,
                    "points": [
                        {key: point[key] for key in ("threshold", "mean_exit_depth")}
                        for point in points
                    ],
                }
            )
        comparisons[model] = {"workload": "ShareGPT", "gates": gates}
    return {
        "token_scope": "decode",
        "request_limits": request_limits,
        "models": models,
        "gate_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workload-dir", type=Path, default=Path("outputs/workloads"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/paper/figs/workloads/workload_exit_distributions.json"),
    )
    parser.add_argument(
        "--length-output",
        type=Path,
        default=Path("docs/paper/figs/workloads/workload_length_distributions.json"),
    )
    parser.add_argument(
        "--requests",
        nargs="+",
        default=[],
        metavar="WORKLOAD=N",
        help="Keep only the first N requests of a workload, e.g. sharegpt=6000.",
    )
    args = parser.parse_args()
    try:
        limits = parse_request_limits(args.requests, [dataset for _, dataset in WORKLOADS])
        results = build_results(args.workload_dir, limits)
        lengths = build_length_distributions(args.workload_dir, limits)
        for path, payload in ((args.output, results), (args.length_output, lengths)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Wrote {path}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
