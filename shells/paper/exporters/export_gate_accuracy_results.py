r"""Export the GSM8K gate comparison for Ouro and Huginn from Loop-SGLang runs.

Reads the gated lm-eval runs of ``shells/paper/ablations/accuracy_{ouro,huginn}.sh``, which
record exit depths next to each result. The output follows the ``models.<model>.variants``
schema read by the paper's ``plot_gate_comparison.py``, which takes the fixed-depth baseline
from the accuracy-throughput exports.

Each gated point carries its threshold, flexible-extract accuracy, decode-token mean exit
depth, and exit-depth counts. Every run is validated against the variant it is exported as,
and all gated runs of one model must share their engine settings apart from the threshold,
exit routing, gate, and depth output.

Example:
    uv run python shells/paper/exporters/export_gate_accuracy_results.py \
        --run-dir outputs/ablations/accuracy \
        --output docs/paper/figs/accuracy/accuracy_gates_loopsgl.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _throughput import fingerprint

TASK = "gsm8k_cot"
METRIC = "exact_match,flexible-extract"
MAX_GEN_TOKS = 256
# Model arguments that legitimately differ between the gated runs of one model.
PER_RUN_ARGS = {"exit_threshold", "disable_lookahead_exit", "exit_depth_output"}


@dataclass(frozen=True)
class Model:
    """Prompt format, checkpoint, and recurrent depth of one model."""

    checkpoint: str
    num_fewshot: int
    max_depth: int


@dataclass(frozen=True)
class Variant:
    """One gated series and the engine settings its runs must use."""

    model: str
    name: str
    pattern: str
    cache_policy: str
    gate_type: str | None
    lookahead: bool


MODELS = {
    "ouro": Model("Ouro-1.4B", 3, 4),
    "huginn": Model("huginn-0125", 8, 16),
}
VARIANTS = (
    Variant(
        "ouro",
        "gate_last_exited",
        "cdb_last_exited_steps_4_early_exit_immediate_threshold_*",
        "last_exited",
        "early_exit",
        lookahead=False,
    ),
    Variant(
        "ouro",
        "gate_lookahead_last_exited",
        "cdb_last_exited_steps_4_early_exit_delayed_threshold_*",
        "last_exited",
        "early_exit",
        lookahead=True,
    ),
    Variant(
        "ouro",
        "lookahead_last_exited",
        "cdb_last_exited_steps_4_lookahead_threshold_*",
        "last_exited",
        "lookahead",
        lookahead=True,
    ),
    Variant(
        "ouro",
        "preloop_last_exited",
        "cdb_last_exited_steps_4_preloop_threshold_*",
        "last_exited",
        "preloop",
        lookahead=True,
    ),
    Variant(
        "huginn",
        "immediate_single",
        "cdb_shared_steps_16_immediate_threshold_*",
        "shared",
        None,
        lookahead=False,
    ),
    Variant(
        "huginn",
        "delayed_single",
        "cdb_shared_steps_16_delayed_threshold_*",
        "shared",
        None,
        lookahead=True,
    ),
)


def only(paths: list[Path], what: str, run: Path) -> Path:
    if len(paths) != 1:
        raise ValueError(f"{run}: expected one {what}, found {len(paths)}")
    return paths[0]


def model_args(config: dict[str, Any]) -> dict[str, Any]:
    args = config["model_args"]
    if isinstance(args, str):
        args = dict(item.split("=", 1) for item in args.split(",") if item)
    return args


def check_settings(
    run: Path, results: dict[str, Any], args: dict[str, Any], variant: Variant
) -> None:
    """Reject runs whose prompt, generation, or engine settings differ from the variant."""

    model = MODELS[variant.model]
    config = results["config"]
    generation = {
        **results["configs"][TASK].get("generation_kwargs", {}),
        **(config.get("gen_kwargs") or {}),
    }
    max_gen_toks = int(generation.get("max_gen_toks", args.get("max_gen_toks", MAX_GEN_TOKS)))
    overrides = json.loads(args.get("json_model_override_args") or "{}")
    disabled = str(args.get("disable_lookahead_exit", False)).lower() == "true"
    checks = {
        "few-shot count": results["n-shot"][TASK] == model.num_fewshot,
        "greedy decoding": generation.get("do_sample") is False,
        "generation limit": max_gen_toks == MAX_GEN_TOKS,
        "full test split": config.get("limit") is None,
        "checkpoint": model.checkpoint in str(args["pretrained"]),
        "scheduler": args["scheduler"] == "cdb",
        "KV policy": args["loop_cache_policy"] == variant.cache_policy,
        "gate type": overrides.get("exit_gate_type") == variant.gate_type,
        "recurrent depth": int(args["loop_steps"]) == model.max_depth,
        "exit routing": disabled != variant.lookahead,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{run}: unexpected {', '.join(failed)}")


def load_point(run: Path, variant: Variant) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    results_path = only(sorted(run.glob("*/results_*.json")), "lm-eval results file", run)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    args = model_args(results["config"])
    check_settings(run, results, args, variant)
    depths_path = run / "depths.json"
    depths = json.loads(depths_path.read_text(encoding="utf-8"))["tasks"][TASK]
    max_depth = MODELS[variant.model].max_depth
    counts = [0] * max_depth
    for depth, count in depths["exit_depth_counts"].items():
        if not 1 <= int(depth) <= max_depth:
            raise ValueError(f"{depths_path}: exit depth {depth} outside 1..{max_depth}")
        counts[int(depth) - 1] = count
    if sum(counts) != depths["decode_tokens"] or not sum(counts):
        raise ValueError(f"{depths_path}: exit counts do not cover the decode tokens")
    point = {
        "threshold": float(args["exit_threshold"]),
        "acc": results["results"][TASK][METRIC],
        "mean_depth": sum(depth * count for depth, count in enumerate(counts, 1)) / sum(counts),
        "exit_counts": counts,
    }
    return point, args, [results_path, depths_path]


def build_export(run_dir: Path) -> dict[str, Any]:
    sources = []
    models = {}
    for name, model in MODELS.items():
        variants = []
        engine_settings = None
        for variant in (variant for variant in VARIANTS if variant.model == name):
            runs = sorted(run for run in (run_dir / name).glob(variant.pattern) if run.is_dir())
            if not runs:
                raise ValueError(f"No runs for {name} {variant.name} ({variant.pattern})")
            points = []
            for run in runs:
                point, args, paths = load_point(run, variant)
                shared = {key: value for key, value in args.items() if key not in PER_RUN_ARGS}
                shared.pop("json_model_override_args", None)
                if engine_settings is None:
                    engine_settings = shared
                elif shared != engine_settings:
                    raise ValueError(f"{run}: engine settings differ from the other {name} runs")
                points.append(point)
                sources.extend(fingerprint(path) for path in paths)
            thresholds = [point["threshold"] for point in points]
            if len(set(thresholds)) != len(thresholds):
                raise ValueError(f"{name} {variant.name}: duplicate thresholds")
            variants.append(
                {
                    "name": variant.name,
                    "kind": "gated",
                    "kv_policy": variant.cache_policy,
                    "exit_gate_type": variant.gate_type,
                    "lookahead": variant.lookahead,
                    "points": sorted(points, key=lambda point: point["threshold"]),
                }
            )
        models[name] = {
            "checkpoint": f"KristianS7/{model.checkpoint}",
            "num_fewshot": model.num_fewshot,
            "max_depth": model.max_depth,
            "engine_settings": engine_settings,
            "variants": variants,
        }
    return {
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": TASK,
        "metric": METRIC.split(",", 1)[1],
        "sources": sources,
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Directory holding ouro/ and huginn/ runs."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        export = build_export(args.run_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(export, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for name, entry in export["models"].items():
        for variant in entry["variants"]:
            print(f"{name} {variant['name']}: {len(variant['points'])} points")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
