# Paper experiments

## Overview

### Data preparation

| Script | Purpose |
| --- | --- |
| `workloads/record_workload.sh` | Creates ShareGPT, Alpaca, and ArXiv workloads containing tokenized requests and recorded per-token exit depths for each threshold; all models' workloads preserve the same requests and replay order |

### Main experiments

| Script                             | Measures                                                                                              | Paper                                                                                                                |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `benchmark_decode_step_latency.sh` | Per-recurrent-step decode latency against batch size and context length                               | **Sec. 5, Fig. 4:** "Efficiency Limits of Adaptive Depth"; **Appendix C, Fig. 13:** "Recurrent-Step Latency Details" |
| `throughput_exit-sweep.sh`         | Offline throughput against the exit threshold (mean exit depth), at a fixed maximum decode batch size | **Sec. 6.2, Fig. 5 left:** "Offline Throughput"                                                                      |
| `throughput_width-sweep.sh`        | Offline throughput against the maximum decode batch size, at a fixed exit threshold (mean exit depth) | **Sec. 6.2, Fig. 5 right:** "Offline Throughput"; **Appendix B, Fig. 11:** "Last-exited KV Cache"                    |
| `serving_rate-sweep.sh`            | Normalized serving latency against different Poisson request rates                                    | **Sec. 6.3, Fig. 6:** "Online Serving"; **Appendix F, Figs. 17 and 18:** "Complete Online Serving Results"           |

### Ablations

| Script                                       | Measures                                                                                         | Paper                                                                              |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `ablations/benchmark_scheduling_overhead.sh` | Profiles GPU idle time under synchronous scheduling, asynchronous scheduling, and lookahead      | **Sec. 4.3, Fig. 3 right:** "Asynchronous Scheduling"                              |
| `ablations/min_coda_batch.sh`                | Ablates the minimum coda batch size for offline throughput                                       | **Sec. 6.4, Fig. 7 left three panels:** "Ablations"                                |
| `ablations/layer_split_huginn.sh`            | Ablates Huginn's prelude-core-coda layer split (0-4-0, 1-4-1)                                    | **Sec. 6.4, Fig. 7 right:** "Ablations"                                            |
| `ablations/benchmark_memory.sh`              | Offline throughput with concurrency (decode batch size) limited by fixed KV-cache memory budgets | **Sec. 6.4, Fig. 8:** "Ablations"; **Appendix B, Fig. 12:** "Last-exited KV Cache" |
| `ablations/accuracy_ouro.sh`                 | GSM8K accuracy sweep for Ouro: fixed depths and exit thresholds under different exit gates       | **Appendix E.2, Fig. 16 left:** "Early-exit Gating"                                |
| `ablations/accuracy_huginn.sh`               | GSM8K accuracy sweep for Huginn: fixed depths and exit thresholds                                | **Appendix E.2, Fig. 16 right:** "Early-exit Gating"                               |
| `ablations/profile_nsys.sh`                  | Profiles CB and CDB GPU activity, launch-site timing, and host waits                             | Supporting diagnostics                                                             |

## Details

### `workloads/record_workload.sh`

- Runs `workloads/prepare_workload.py` once per model and dataset to tokenize prompts and reference responses.
- Runs `workloads/record_workload.py` for every threshold.
  We use forced decoding on the reference response: the model decodes it token by token using KV cache, as in normal generation, but receives the reference token as the next input instead of its own prediction.
- Writes recordings to `outputs/workloads/<dataset>/<model>/`, one JSONL file per KV cache policy, exit gate, exit routing, and threshold.
  - The throughput and serving experiments replay these recorded exits.
- Follow-up: `exporters/export_workload_results.py` writes JSON files containing the data used to generate the workload figures and tables.
  - It exports prompt and output length distributions, exit-depth distributions, and mean exit depths (**Appendix D, Figs. 14 and 15, Tabs. 4 and 5**).

### `benchmark_decode_step_latency.sh`

- Runs `loopsgl.benchmark.one_batch` with Ouro 1.4B, Ouro 2.6B, or Huginn 3.5B.
- Writes a JSONL file with sweep results to `outputs/decode_step_latency/`.
  - Each batch-size and context-length point contains best-window latencies, a regression fit, and diagnostics.
- Follow-up: `exporters/export_decode_step_latency_results.py` writes a JSON file containing the data used to generate the paper figures.
  - It filters the data and computes the weight-reload cost, saturation batch `B*`, and per-sequence compute and KV-streaming slope.

### `throughput_exit-sweep.sh`

- Runs `loopsgl.benchmark.offline_throughput` for CB, CDB no-refill, and CDB with refill at multiple exit thresholds.
- Writes one JSONL file per model and dataset to `outputs/exit-sweep/`.
- Each row records whole-run throughput and CUDA event timings for prefill and decode.
- Follow-up: `exporters/export_exit_sweep_results.py` writes a JSON file containing the data used to generate the paper figure.
  - It exports throughput normalized to CB, the decode FLOP bound over the threshold range, and the prefill-adjusted end-to-end bound from the CB timings.

### `throughput_width-sweep.sh`

- Runs `loopsgl.benchmark.offline_throughput` for CB, CDB no-refill, and CDB with refill at multiple maximum decode batch sizes with a fixed exit threshold.
- Writes one JSONL file per model and dataset to `outputs/width-sweep/`.
- Follow-up: `exporters/export_width_sweep_results.py` writes a JSON file containing the data used to generate the paper figure.
  - It exports throughput normalized to CB, the decode FLOP bound, and the width-dependent prefill-adjusted end-to-end bound.
  - Passing `--b-star-from` also exports the saturation batch `B*` (requires running `benchmark_decode_step_latency.sh` first).

### `serving_rate-sweep.sh`

- Runs an open-loop serving benchmark with Poisson arrivals for CB, CDB no-refill, and CDB with refill.
- Writes one JSONL file per model and dataset to `outputs/serving-rate/`.
- Follow-up: `exporters/export_serving_results.py` combines one model's dataset summaries and writes the JSON file for the paper figure.
  - It exports normalized latency, TTFT (including P99), TPOT, queueing, throughput, batch-size, and residency metrics.
  - It also exports the mean exit depth of every replayed recording.

### `ablations/benchmark_scheduling_overhead.sh`

- Runs `loopsgl.benchmark.offline_throughput` under Nsight Systems for CDB with synchronous scheduling, asynchronous scheduling with immediate gate consumption, and asynchronous scheduling with lookahead.
- Writes Nsight `.nsys-rep` traces, SQLite exports, benchmark summaries, and analyzed GPU timing files to `outputs/ablations/scheduling-overhead/`.
- Follow-up: `exporters/export_scheduling_overhead_results.py` writes the validated idle fractions to a tex file.

### `ablations/profile_nsys.sh`

- Runs `loopsgl.benchmark.offline_throughput` under Nsight Systems for CB, CDB without refill, and CDB with refill at different maximum decode batch sizes.
- Writes Nsight `.nsys-rep` traces, SQLite exports, benchmark summaries, and analyzed GPU timing files to `outputs/nsys/`.

### `ablations/min_coda_batch.sh`

- Runs the ShareGPT experiment from `throughput_exit-sweep.sh` with different minimum coda batch sizes.
- Runs only CDB with refill.
- Sweeps minimum coda batch sizes and maximum decode batch sizes.
  1. The Ouro experiment uses a maximum decode batch size of 64.
  2. The Huginn experiment sweeps multiple maximum decode batch sizes.
- Writes one JSONL file per model to `outputs/ablations/min-coda-batch/`.
- Follow-up: `exporters/export_coda_ablation_results.py` combines the minimum-coda and layer-split results.

### `ablations/layer_split_huginn.sh`

- Runs the ShareGPT experiment from `throughput_exit-sweep.sh` with Huginn's boundary stages removed (0-4-0) or halved (1-4-1).
- Uses the same recorded workloads for all layer splits.
- Writes one JSONL file to `outputs/ablations/layer-split/`.
- Follow-up: `exporters/export_coda_ablation_results.py` combines the minimum-coda and layer-split results.

### `ablations/benchmark_memory.sh`

- Runs `loopsgl.benchmark.offline_throughput` for full-depth CB with shared and depth-indexed KV, and CDB with refill with shared and last-exited KV, at fixed KV-cache memory budgets.
- `RUNS` selects a subset of the backend and KV policy pairs.
- Writes one JSONL file per dataset and budget to `outputs/ablations/memory/`.
- Follow-up: `exporters/export_memory_results.py` writes a JSON file containing the data used to generate the paper figure.
  - It exports throughput normalized to depth-indexed CB and the median decode batch size.

### Last-exited KV cache comparison

- Compares CDB with refill under shared, last-exited, and copying last-exited KV caches (**Appendix B, Figs. 11 and 12**).
- Reuses the shared-KV runs of `throughput_width-sweep.sh` and `ablations/benchmark_memory.sh`.
- Runs `throughput_width-sweep.sh` with `CACHE_POLICY=last_exited`, and with `CACHE_POLICY=depth_indexed BACKENDS=cb` for the matching CB baseline.
  - The depth-indexed runs can only use a smaller cache than shared KV runs: `CACHE_TOKENS=100000`.
- The copying cache is implemented on the `feat/last-exited-copy` branch.
  - On that branch, run `throughput_width-sweep.sh` with `CACHE_POLICY=last_exited_copy BACKENDS=cdb-refill` and `ablations/benchmark_memory.sh` with `RUNS=cdb-refill:last_exited_copy`.
  - Copy the width summaries into `outputs/width-sweep/` and the memory summaries, with their `logs/` directory, into `outputs/ablations/memory/copy/`.
- Follow-up: `exporters/export_last_exited_results.py` writes a JSON file containing the data used to generate the paper figures.
  - It exports throughput normalized to depth-indexed CB for both last-exited caches and to shared CB for shared KV, together with median decode batch sizes under memory limits.

### `ablations/accuracy_ouro.sh` and `ablations/accuracy_huginn.sh`

- Run `loopsgl.evaluation` for GSM8K.
- Fixed-depth rows use CB, and early-exit rows use CDB.
- Write one result directory per configuration to `outputs/ablations/accuracy/`, with recorded exit depths beside each gated result.
- Follow-up: `exporters/export_gate_accuracy_results.py` writes a JSON file containing the data used to generate the paper figure.
  - It exports accuracy against the mean exit depth and the exit-depth counts for every gate and threshold.
