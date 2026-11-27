"""Record live CDB exits or write full-depth controls from tokenized references."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loopsgl import Engine
from loopsgl.core import SamplingParams
from loopsgl.server_args import ServerArgs
from prepare_workload import output_file


def tokenized_rows(path: Path, vocab_size: int, max_seq_len: int) -> Iterator[dict[str, Any]]:
    """Validate the rows without tokenizing, truncating, or changing their order."""
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected an object.")
            request_id = row.get("id")
            if not isinstance(request_id, str) or not request_id or request_id in seen:
                raise ValueError(
                    f"Line {line_number}: missing or duplicate request ID {request_id!r}."
                )
            seen.add(request_id)
            for key in ("input_ids", "output_ids"):
                values = row.get(key)
                if (
                    not isinstance(values, list)
                    or not values
                    or any(
                        type(value) is not int or not 0 <= value < vocab_size for value in values
                    )
                ):
                    raise ValueError(f"Request {request_id}: {key} must contain valid token IDs.")
            if len(row["input_ids"]) + len(row["output_ids"]) > max_seq_len:
                raise ValueError(
                    f"Request {request_id}: prompt and continuation exceed {max_seq_len} tokens."
                )
            yield {"id": request_id, "input_ids": row["input_ids"], "output_ids": row["output_ids"]}


def decode_depths(row: dict[str, Any], output: dict[str, Any], max_depth: int) -> list[int]:
    """Pair returned decode depths with reference positions after the prefill output."""
    request_id = row["id"]
    if output.get("token_ids") != row["output_ids"]:
        raise ValueError(
            f"Request {request_id}: returned tokens differ from the forced continuation."
        )
    pairs = output.get("meta_info", {}).get("decode_token_loop_steps")
    if not isinstance(pairs, list) or len(pairs) != len(row["output_ids"]) - 1:
        raise ValueError(f"Request {request_id}: expected one depth per decode output token.")
    depths = []
    for position, (target, pair) in enumerate(zip(row["output_ids"][1:], pairs, strict=True), 1):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"Request {request_id}, output position {position}: invalid depth pair."
            )
        token_id, depth = pair
        if type(token_id) is not int or token_id != target:
            raise ValueError(
                f"Request {request_id}, output position {position}: depth token mismatch."
            )
        if type(depth) is not int or not 1 <= depth <= max_depth:
            raise ValueError(
                f"Request {request_id}, output position {position}: invalid depth {depth!r}."
            )
        depths.append(depth)
    return depths


def record_workload(args: argparse.Namespace, config: ServerArgs) -> None:
    if config.scheduler != "cdb" or config.loop_params.exit_threshold is None:
        raise ValueError("Recording requires CDB with a live exit threshold.")
    if config.loop_cache_policy not in {"shared", "last_exited"}:
        raise ValueError("Recording requires shared or last_exited KV.")
    if (
        not config.disable_radix_cache
        or config.retraction_mode != "none"
        or config.chunked_prefill_size > 0
    ):
        raise ValueError(
            "Recording requires --disable-radix-cache --retraction-mode none --chunked-prefill-size -1."
        )
    if config.use_dummy_weight:
        raise ValueError("Workload recording requires model weights.")
    rows = list(
        tokenized_rows(args.dataset_path, config.model_config.vocab_size, config.max_seq_len)
    )
    if not rows:
        raise ValueError("Tokenized workload is empty.")
    max_depth = config.model_config.loop_steps
    full_depth = (
        config.model_config.model_type == "ouro" and config.loop_params.exit_threshold == 1.0
    ) or (
        config.model_config.model_type == "huginn_raven"
        and config.loop_params.exit_threshold == 0.0
    )
    if full_depth:
        print(
            f"Writing full-depth exits at depth {max_depth} without running the model", flush=True
        )
    with output_file(args.output_path) as path:
        engine = None if full_depth else Engine(server_args=config)
        try:
            outputs = None
            if engine is not None:
                if engine.max_seq_len < config.max_seq_len:
                    for row in rows:
                        if len(row["input_ids"]) + len(row["output_ids"]) > engine.max_seq_len:
                            raise ValueError(
                                f"Request {row['id']}: prompt and continuation exceed {engine.max_seq_len} tokens."
                            )
                params = [
                    SamplingParams(
                        trace_decode_token_ids=row["output_ids"],
                        max_new_tokens=len(row["output_ids"]),
                        ignore_eos=True,
                        return_loop_steps=True,
                    )
                    for row in rows
                ]
                print(f"Recording {len(rows)} requests", flush=True)
                outputs = engine.generate(
                    input_ids=[row["input_ids"] for row in rows],
                    sampling_params=params,
                    return_logprob=False,
                )
                if not isinstance(outputs, list) or len(outputs) != len(rows):
                    raise ValueError("Engine response does not match the submitted workload.")
            with path.open("w", encoding="utf-8") as handle:
                for index, row in enumerate(rows):
                    depths = (
                        [max_depth] * (len(row["output_ids"]) - 1)
                        if outputs is None
                        else decode_depths(row, outputs[index], max_depth)
                    )
                    handle.write(
                        json.dumps({**row, "exit_depths": depths}, separators=(",", ":")) + "\n"
                    )
        finally:
            if engine is not None:
                engine.shutdown()
    print(f"Recorded {len(rows)} requests to {args.output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = ServerArgs.from_cli_args(args)
        record_workload(args, config)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
