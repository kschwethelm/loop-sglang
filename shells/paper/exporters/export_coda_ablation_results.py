r"""Export the ShareGPT coda batching and layer-split figure data.

Pass the JSONL summaries from the two ablation runners and use --baselines for
the matching full-depth CB rows from the width sweep. Keep the ablation logs
beside their summaries under logs/, as written by the runners.

Example:
    uv run python shells/paper/exporters/export_coda_ablation_results.py \
        outputs/ablations/min-coda-batch/min-coda-batch_*_sharegpt.jsonl \
        outputs/ablations/layer-split/layer-split_huginn_sharegpt.jsonl \
        --baselines outputs/width-sweep/width-sweep_*_sharegpt_shared.jsonl \
        --output coda_ablations.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from glob import escape
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from _throughput import Run, fingerprint, json_key, load_runs, positive_int

Curve = tuple[str, str, int]
PAPER_GRID: dict[Curve, tuple[int, ...]] = {
    ("ouro", "default", 64): (1, 8, 16, 32),
    **{
        ("huginn", "default", width): (1, width // 8, width // 4, width // 2)
        for width in (16, 32, 64, 512)
    },
    ("huginn", "0-4-0", 64): (1, 8, 16, 32, 64),
    ("huginn", "1-4-1", 64): (1, 8, 16, 32, 64),
}
SAMPLED_CURVE = ("huginn", "default", 64)
LAYER_COUNTS = ("num_prelude_layers", "num_core_layers", "num_coda_layers")
STAGE_SAMPLE = re.compile(r"\b(Recurrent|Coda) batch, .*?#batch-token: (\d+),")


def curve_for(run: Run) -> Curve:
    model_path = run.config["model_path"].lower()
    overrides = json.loads(run.config["json_model_override_args"])
    if "huginn" in model_path:
        model = "huginn"
        counts = tuple(overrides.get(key, default) for key, default in zip(LAYER_COUNTS, (2, 4, 2)))
        if counts not in {(2, 4, 2), (1, 4, 1), (0, 4, 0)} or overrides.get(
            "num_hidden_layers", sum(counts)
        ) != sum(counts):
            raise ValueError(f"{run.source}: unsupported Huginn layer counts {counts}")
        split = "default" if counts == (2, 4, 2) else "-".join(map(str, counts))
    elif "ouro" in model_path:
        model, split = "ouro", "default"
    else:
        raise ValueError(f"{run.source}: unsupported model {model_path}")
    if run.name != "sharegpt":
        raise ValueError(f"{run.source}: coda ablations require ShareGPT")
    return model, split, run.width


def comparison_key(run: Run, *, across_widths: bool = False, across_splits: bool = False) -> str:
    settings = run.settings(across_widths=across_widths)
    overrides = json.loads(settings["json_model_override_args"])
    if across_splits:
        for key in (*LAYER_COUNTS, "num_hidden_layers"):
            overrides.pop(key, None)
    settings["json_model_override_args"] = overrides
    return json_key(
        (
            settings,
            run.config["dataset_path"],
            run.num_requests,
            run.input_tokens,
            run.output_tokens,
        )
    )


def refill_key(run: Run) -> str:
    cdb = run.config["cdb_config"].copy()
    del cdb["min_coda_batch_size"]
    return json_key((run.config["loop_params"], cdb, run.mean_depth))


def check_repeats(runs: list[Run], label: str) -> None:
    campaigns = set()
    repeats = set()
    for run in runs:
        stem = run.source.stem
        suffix = re.search(r"_repeat(\d+)$", stem)
        campaigns.add((run.source.parent, stem[: suffix.start()] if suffix else stem, bool(suffix)))
        repeat = int(suffix[1]) if suffix else 0
        if repeat in repeats or len(campaigns) != 1:
            raise ValueError(f"{label}: duplicate repeat or multiple campaigns at {run.source}")
        repeats.add(repeat)


def stage_samples(run: Run, k: int) -> dict[str, list[int]]:
    log_dir = run.source.parent / "logs"
    summary_name = run.source.name.removesuffix(".jsonl")
    prefix = f"{summary_name}_default_w{run.width}_cdb-refill_k{k}_"
    logs = sorted(log_dir.glob(f"{escape(prefix)}*.out"))
    if len(logs) != 1:
        raise ValueError(
            f"{run.source}: expected one log at {log_dir / prefix}*.out, found {len(logs)}"
        )

    samples: dict[str, list[int]] = {"recurrent": [], "coda": []}
    measured = False
    with logs[0].open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "Benchmark...":
                if measured:
                    raise ValueError(f"{logs[0]}: multiple benchmark-start markers")
                measured = True
            if measured and (match := STAGE_SAMPLE.search(line)):
                size = int(match[2])
                if not 1 <= size <= run.width:
                    raise ValueError(f"{logs[0]}: invalid {match[1]} batch size {size}")
                samples[match[1].lower()].append(size)
    if not measured or not all(samples.values()):
        raise ValueError(f"{logs[0]}: missing benchmark-start marker or measured stage samples")
    return samples


def build_panel(curve: Curve, runs: list[Run], controls: list[Run]) -> dict[str, Any]:
    model, split, width = curve
    ks = PAPER_GRID[curve]
    by_k: dict[int, list[Run]] = defaultdict(list)
    for run in runs:
        k = positive_int(run.config["cdb_config"]["min_coda_batch_size"], "min_coda_batch_size")
        by_k[k].append(run)
    if set(by_k) != set(ks):
        raise ValueError(f"{curve}: expected K values {list(ks)}, got {sorted(by_k)}")
    settings = comparison_key(runs[0])
    if any(comparison_key(run) != settings for run in runs[1:]):
        raise ValueError(f"{curve}: refill points mix replay inputs or benchmark settings")
    if not controls:
        raise ValueError(f"{curve}: missing full-depth CB baseline")
    if any(comparison_key(run) != settings for run in controls):
        raise ValueError(f"{curve}: CB baseline does not match replay inputs or benchmark settings")
    check_repeats(controls, f"{curve} CB")

    rates, deviations = [], []
    batch_sizes: dict[str, list[float]] = {"recurrent_batch_size": [], "coda_batch_size": []}
    for k in ks:
        repeats = by_k[k]
        check_repeats(repeats, f"{curve} K={k}")
        measurements = [run.row["output_throughput"] for run in repeats]
        rates.append(mean(measurements))
        deviations.append(stdev(measurements) if len(measurements) > 1 else None)
        if curve == SAMPLED_CURVE:
            samples = [stage_samples(run, k) for run in repeats]
            for stage in ("recurrent", "coda"):
                batch_sizes[f"{stage}_batch_size"].append(
                    mean(size for sample in samples for size in sample[stage])
                )

    cb_rate = mean(run.row["output_throughput"] for run in controls)
    return {
        "model": model,
        "layer_split": split,
        "width": width,
        "name": "sharegpt",
        "threshold": runs[0].threshold,
        "num_requests": runs[0].num_requests,
        "measurement_configs": [runs[0].settings()],
        "source_summaries": sorted({str(run.source) for run in runs}),
        "cb_source_summaries": sorted({str(run.source) for run in controls}),
        "k_values": list(ks),
        "gen_tps": rates,
        "gen_tps_std": deviations,
        "cb_gen_tps": cb_rate,
        "speedup_over_cb": [rate / cb_rate for rate in rates],
        "repeats": [len(by_k[k]) for k in ks],
        **(batch_sizes if curve == SAMPLED_CURVE else {}),
    }


def build_results(paths: list[Path], baseline_paths: list[Path]) -> dict[str, Any]:
    sources = [fingerprint(path) for path in [*paths, *baseline_paths]]
    if len({source["sha256"] for source in sources}) != len(sources):
        raise ValueError("Duplicate summary input")

    refill: dict[Curve, list[Run]] = defaultdict(list)
    controls: dict[Curve, list[Run]] = defaultdict(list)
    for run in load_runs(paths):
        if run.series == "refill":
            refill[curve_for(run)].append(run)
        elif run.backend == "cb":
            controls[curve_for(run)].append(run)
    for run in load_runs(baseline_paths):
        if run.backend == "cb":
            controls[curve_for(run)].append(run)
    if set(refill) != set(PAPER_GRID):
        missing = sorted(PAPER_GRID.keys() - refill.keys())
        extra = sorted(refill.keys() - PAPER_GRID.keys())
        raise ValueError(f"Invalid paper sweep grid: missing {missing}, unexpected {extra}")

    for model, threshold, depth, recording in (
        ("ouro", 0.0001, 4, "last_exited_early_exit_lookahead"),
        ("huginn", 0.14, 16, "shared_convergence_lookahead"),
    ):
        runs = [run for curve, points in refill.items() if curve[0] == model for run in points]
        replay = f"sharegpt/{model}/{recording}_q{threshold}.jsonl"
        if any(
            run.threshold != threshold
            or run.depth != depth
            or Path(run.config["dataset_path"]).parts[-3:] != Path(replay).parts
            for run in runs
        ):
            raise ValueError(f"{model}: paper sweep requires q={threshold} and replay {replay}")
        if len({refill_key(run) for run in runs}) != 1:
            raise ValueError(f"{model}: refill points mix exit schedules or CDB settings")
        if len({comparison_key(run, across_widths=True, across_splits=True) for run in runs}) != 1:
            raise ValueError(f"{model}: curves differ in settings other than width and layer split")
        width_64 = [run for run in runs if run.width == 64]
        if len({comparison_key(run, across_splits=True) for run in width_64}) != 1:
            raise ValueError(f"{model}: W=64 curves differ in settings other than layer split")

    return {
        "meta": {
            "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": sources,
            "throughput_basis": "Generated tokens divided by whole-run wall time.",
            "batch_size_basis": "Measured recurrent and coda log samples pooled over repeats.",
        },
        "panels": [build_panel(curve, refill[curve], controls[curve]) for curve in PAPER_GRID],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("summaries", nargs="+", type=Path, help="Ablation summary JSONL files.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        type=Path,
        default=[],
        help="Width-sweep JSONL with matched CB rows.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination paper JSON.")
    args = parser.parse_args()
    try:
        results = build_results(args.summaries, args.baselines)
        output = json.dumps(results, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
