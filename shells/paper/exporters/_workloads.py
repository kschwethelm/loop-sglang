"""Shared helpers for exporting recorded workloads and the benchmarks that replay them."""

from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import nullcontext
from itertools import islice, zip_longest
from pathlib import Path
from typing import Any


def parse_request_limits(items: Iterable[str], workloads: Iterable[str]) -> dict[str, int]:
    """Parse ``WORKLOAD=N`` request limits for the given workload names."""
    names = set(workloads)
    limits = {}
    for item in items:
        name, _, count = item.partition("=")
        if name not in names or not count.isdigit():
            raise ValueError(f"Invalid request limit {item!r}")
        limits[name] = int(count)
    return limits


def load_recording(
    path: Path, max_depth: int, reference: Path | None = None, limit: int | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Summarize depths and verify identical tokens against a reference recording.

    ``limit`` keeps only the first requests of both recordings.
    """
    counts = [0] * max_depth
    ids: list[str] = []
    seen: set[str] = set()
    output_tokens = 0
    with (
        path.open(encoding="utf-8") as handle,
        reference.open(encoding="utf-8") if reference else nullcontext() as baseline,
    ):
        rows = islice(
            zip_longest(handle, baseline) if baseline else ((line, None) for line in handle),
            limit,
        )
        for line_number, (line, expected_line) in enumerate(rows, 1):
            if line is None or (baseline and expected_line is None):
                raise ValueError(f"{path}: request count differs from {reference}")
            row = json.loads(line)
            request_id = row["id"]
            if not isinstance(request_id, str) or not request_id or request_id in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate request ID")
            if not all(
                isinstance(row[key], list) and row[key] for key in ("input_ids", "output_ids")
            ):
                raise ValueError(f"{path}:{line_number}: missing prompt or output tokens")
            if expected_line is not None:
                expected = json.loads(expected_line)
                if any(row[key] != expected[key] for key in ("id", "input_ids", "output_ids")):
                    raise ValueError(
                        f"{path}:{line_number}: tokens or order differ from {reference}"
                    )
            depths = row["exit_depths"]
            if not isinstance(depths, list) or len(depths) != len(row["output_ids"]) - 1:
                raise ValueError(f"{path}:{line_number}: expected one depth per decode token")
            for depth in depths:
                if type(depth) is not int or not 1 <= depth <= max_depth:
                    raise ValueError(f"{path}:{line_number}: invalid exit depth {depth!r}")
                counts[depth - 1] += 1
            seen.add(request_id)
            ids.append(request_id)
            output_tokens += len(row["output_ids"])
    decode_tokens = sum(counts)
    if limit is not None and len(ids) != limit:
        raise ValueError(f"{path}: holds {len(ids)} requests, fewer than {limit}")
    if not ids or not decode_tokens:
        raise ValueError(f"{path}: no recorded decode tokens")
    return {
        "num_requests": len(ids),
        "num_output_tokens": output_tokens,
        "num_decode_tokens": decode_tokens,
        "exit_counts": counts,
        "exit_fractions": [count / decode_tokens for count in counts],
        "mean_exit_depth": sum(depth * count for depth, count in enumerate(counts, 1))
        / decode_tokens,
    }, ids
