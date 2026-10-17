"""Analyze full-generation Nsight Systems SQLite exports for one GPU.

GPU busy time is the union of kernel, copy, memset and whole-graph intervals.
Whole-graph tracing treats each graph execution as busy, including its internal gaps.
Stage GPU time follows launch correlation IDs and can overlap across streams.
Stage ranges include sampling and output copies launched by the corresponding batch or phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any

GPU_ACTIVITY_TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
)
CUDA_API_TABLES = ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
MEASURED_LABEL = "benchmark.generate"
STAGE_LABELS = (
    "cb.compute.prefill",
    "cb.compute.decode",
    "cdb.compute.prefill",
    "cdb.compute.prelude",
    "cdb.compute.core",
    "cdb.compute.coda",
    "cdb.gate_wait",
)
Interval = tuple[int, int]
Correlation = tuple[int, int]


def process_id(serialized_id: int) -> int:
    """Extract the process identifier from Nsight's serialized globalPid or globalTid."""
    return (serialized_id >> 24) & 0xFFFFFF


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Combine overlapping or adjacent intervals without counting concurrent work twice."""
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(merged[-1][1], end)
        else:
            merged.append((start, end))
    return merged


def clip_intervals(intervals: list[Interval], window: Interval) -> list[Interval]:
    """Intersect intervals with the measured generation window."""
    return merge_intervals(
        [(max(start, window[0]), min(end, window[1])) for start, end in intervals]
    )


def summarize_intervals(window: Interval, busy: list[Interval]) -> dict[str, float | int]:
    """Measure GPU busy time and gaps inside a single generation window."""
    busy = clip_intervals(busy, window)
    gaps: list[int] = []
    cursor = window[0]
    for start, end in busy:
        if start > cursor:
            gaps.append(start - cursor)
        cursor = end
    if cursor < window[1]:
        gaps.append(window[1] - cursor)
    duration = window[1] - window[0]
    busy_ns = sum(end - start for start, end in busy)
    idle_ns = duration - busy_ns
    return {
        "duration_s": duration / 1e9,
        "gpu_busy_s": busy_ns / 1e9,
        "gpu_idle_s": idle_ns / 1e9,
        "gpu_busy_fraction": busy_ns / duration,
        "gpu_idle_fraction": idle_ns / duration,
        "idle_gap_count": len(gaps),
        "largest_idle_gap_ms": max(gaps, default=0) / 1e6,
    }


def read_nvtx_ranges(
    conn: sqlite3.Connection, strings: dict[int, str]
) -> dict[str, dict[int, list[Interval]]]:
    """Read completed generation and execution ranges, grouped by label and host thread."""
    ranges: dict[str, dict[int, list[Interval]]] = defaultdict(lambda: defaultdict(list))
    for row in conn.execute("SELECT * FROM NVTX_EVENTS"):
        if row["start"] is None or row["end"] is None or row["end"] <= row["start"]:
            continue
        name = row["text"] if "text" in row.keys() else None
        if name is None and "textId" in row.keys():
            name = strings.get(row["textId"])
        if name == MEASURED_LABEL or name in STAGE_LABELS:
            ranges[name][int(row["globalTid"])].append((int(row["start"]), int(row["end"])))
    return ranges


def read_gpu_activities(
    conn: sqlite3.Connection, tables: set[str], window: Interval
) -> dict[int, list[sqlite3.Row]]:
    """Collect GPU activities overlapping generation separately for each recorded device."""
    activities: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for table in GPU_ACTIVITY_TABLES:
        if table not in tables:
            continue
        for row in conn.execute(
            f"SELECT * FROM {table} WHERE start < ? AND end > ?", (window[1], window[0])
        ):
            activities[int(row["deviceId"])].append(row)
    return activities


def launched_gpu_ns_by_stage(
    conn: sqlite3.Connection,
    tables: set[str],
    strings: dict[int, str],
    ranges: dict[str, dict[int, list[Interval]]],
    activities: list[sqlite3.Row],
    window: Interval,
) -> tuple[dict[str, int], int]:
    """Attribute GPU durations to launching host ranges with process-scoped CUDA correlations."""
    gpu_ns: dict[Correlation, int] = defaultdict(int)
    for row in activities:
        if row["correlationId"] is not None and row["globalPid"] is not None:
            key = process_id(int(row["globalPid"])), int(row["correlationId"])
            gpu_ns[key] += int(row["end"]) - int(row["start"])
    launches: dict[int, list[tuple[int, int, Correlation]]] = defaultdict(list)
    graph_launches: set[tuple[int, int, int]] = set()
    processes = {process_id(int(row["globalPid"])) for row in activities}
    for table in CUDA_API_TABLES:
        if table not in tables:
            continue
        for row in conn.execute(
            f"SELECT * FROM {table} WHERE start < ? AND end > ?", (window[1], window[0])
        ):
            tid = int(row["globalTid"])
            name = strings.get(row["nameId"], "")
            if process_id(tid) in processes and (
                name.startswith("cudaGraphLaunch") or name.startswith("cuGraphLaunch")
            ):
                graph_launches.add((tid, int(row["start"]), int(row["end"])))
            if row["correlationId"] is not None:
                key = process_id(tid), int(row["correlationId"])
                if key in gpu_ns:
                    launches[tid].append((int(row["start"]), int(row["end"]), key))
    for entries in launches.values():
        entries.sort()
    totals: dict[str, int] = {}
    for label, threads in ranges.items():
        credited: set[Correlation] = set()
        for tid, intervals in threads.items():
            entries = launches.get(tid, [])
            starts = [entry[0] for entry in entries]
            for start, end in clip_intervals(intervals, window):
                index = bisect_left(starts, start)
                while index < len(entries) and entries[index][0] < end:
                    if entries[index][1] <= end:
                        credited.add(entries[index][2])
                    index += 1
        totals[label] = sum(gpu_ns[key] for key in credited)
    return totals, len(graph_launches)


def summarize_sqlite(path: Path, device_id: int | None = None) -> dict[str, Any]:
    """Summarize full-generation idle time and stage GPU work for one selected device."""
    if not path.is_file():
        raise ValueError(f"SQLite export does not exist: {path}")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "NVTX_EVENTS" not in tables:
            raise ValueError("No NVTX events found; capture with --trace=cuda,nvtx.")
        strings = (
            dict(conn.execute("SELECT id, value FROM StringIds")) if "StringIds" in tables else {}
        )
        ranges = read_nvtx_ranges(conn, strings)
        windows = [
            interval
            for intervals in ranges.get(MEASURED_LABEL, {}).values()
            for interval in intervals
        ]
        if len(windows) != 1:
            raise ValueError(f"Expected one {MEASURED_LABEL} range, found {len(windows)}.")
        window = windows[0]
        devices = read_gpu_activities(conn, tables, window)
        if not devices:
            raise ValueError(
                "No GPU activity was recorded inside benchmark.generate. "
                "Check Nsight/driver compatibility and scheduler worker tracing."
            )
        if device_id is None:
            if len(devices) != 1:
                raise ValueError(f"Select --device-id from recorded devices {sorted(devices)}.")
            device_id = next(iter(devices))
        if device_id not in devices:
            raise ValueError(f"Device {device_id} has no GPU activities during generation.")
        activities = devices[device_id]
        processes = {process_id(int(row["globalPid"])) for row in activities}
        busy = [(int(row["start"]), int(row["end"])) for row in activities]
        measured = summarize_intervals(window, busy)
        launched_ns, graph_launches = launched_gpu_ns_by_stage(
            conn, tables, strings, ranges, activities, window
        )
        stages = {}
        for label in STAGE_LABELS:
            if label not in ranges:
                continue
            intervals = clip_intervals(
                [
                    interval
                    for tid, entries in ranges[label].items()
                    if process_id(tid) in processes
                    for interval in entries
                ],
                window,
            )
            if intervals:
                stages[label] = {
                    "cpu_range_s": sum(end - start for start, end in intervals) / 1e9,
                    "gpu_launched_s": launched_ns.get(label, 0) / 1e9,
                }
        columns = {key for row in activities for key in row.keys()}
        has_graph_trace = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables and bool(
            conn.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
                "WHERE deviceId = ? AND start < ? AND end > ?",
                (device_id, window[1], window[0]),
            ).fetchone()[0]
        )
        has_graph_nodes = "graphNodeId" in columns and any(
            "graphNodeId" in row.keys() and row["graphNodeId"] not in (None, 0)
            for row in activities
        )
        warnings = []
        if graph_launches and not (has_graph_trace or has_graph_nodes):
            warnings.append(
                "CUDA graph launches have no recorded GPU graph activity; capture from process startup."
            )
        for label, stage in stages.items():
            if ".compute." in label and stage["gpu_launched_s"] == 0:
                warnings.append(f"No GPU work was correlated with executed range {label}.")
        prefill = stages.get("cb.compute.prefill", {}).get("gpu_launched_s", 0)
        decode = stages.get("cb.compute.decode", {}).get("gpu_launched_s", 0)
        return {
            "path": str(path),
            "device_id": device_id,
            "measured_label": MEASURED_LABEL,
            "measured": measured,
            "stages": stages,
            "prefill_fraction": prefill / (prefill + decode) if prefill + decode else None,
            "diagnostics": {
                "cuda_graph_launch_count": graph_launches,
                "has_graph_trace_activity": has_graph_trace,
                "has_graph_node_kernel_activity": has_graph_nodes,
                "warnings": warnings,
            },
        }


def main() -> None:
    """Write a JSON analysis and optional CSV from a full-generation Nsight SQLite export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize_sqlite(args.sqlite, args.device_id)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"label": MEASURED_LABEL, **summary["measured"]}]
        rows.extend({"label": label, **stage} for label, stage in summary["stages"].items())
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with args.output_csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    if summary["diagnostics"]["warnings"]:
        raise SystemExit("Nsight capture diagnostics failed; inspect the analysis warnings.")


if __name__ == "__main__":
    main()
