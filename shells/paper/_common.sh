#!/usr/bin/env bash
# Shared paper experiment configuration

set -euo pipefail

readonly OURO_1_4B_MODEL_ID="${OURO_1_4B_MODEL_ID:-KristianS7/Ouro-1.4B}"
readonly OURO_2_6B_MODEL_ID="KristianS7/Ouro-2.6B"
readonly HUGINN_MODEL_ID="KristianS7/huginn-0125"

# Throughput defaults are applied by the throughput scripts' CLI arguments.
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa}"
PAGE_SIZE="${PAGE_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
HUGINN_MODEL_OVERRIDE_ARGS="{\"max_position_embeddings\":$MAX_MODEL_LEN,\"state_init\":\"zero\"}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-$MAX_PREFILL_TOKENS}"
MEMORY_RATIO="${MEMORY_RATIO:-0.9}"
WATERMARK="${WATERMARK:-0.0}"
RETRACTION_MODE="${RETRACTION_MODE:-none}"
# Paper experiments disable cross-request prefix reuse by default.
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-true}"
case "${DISABLE_RADIX_CACHE,,}" in
    true | 1) DISABLE_RADIX_CACHE=True; RADIX_CACHE_ARGS=(--disable-radix-cache) ;;
    false | 0) DISABLE_RADIX_CACHE=False; RADIX_CACHE_ARGS=() ;;
    *) echo "DISABLE_RADIX_CACHE must be true or false." >&2; exit 1 ;;
esac

# Cache budgets and width ranges are sized for a single 80GB H100 GPU.
set_throughput_defaults() {
    local default_widths default_decode_width default_cache_tokens
    CACHE_POLICY="${CACHE_POLICY:-shared}"
    case "$CACHE_POLICY" in
        shared | last_exited | depth_indexed) ;;
        *) echo "unknown CACHE_POLICY '$CACHE_POLICY' (expected: shared, last_exited, depth_indexed)" >&2; return 1 ;;
    esac
    case "$DATASET" in
        sharegpt | alpaca)
            default_widths="8 16 32 64 128 256 512"
            default_decode_width=64
            ;;
        arxiv)
            default_widths="2 4 8 16 32"
            default_decode_width=8
            ;;
        *) echo "unknown dataset '$DATASET' (expected: sharegpt, alpaca, arxiv)" >&2; return 1 ;;
    esac
    default_cache_tokens=327680
    if [[ "$CACHE_POLICY" == last_exited ]]; then
        # Last-exited cache requires much more memory than shared KV, making some widths infeasible.
        PAGE_SIZE=1
        # Full-depth capacities sized to fit alongside model weights and runtime buffers.
        case "$MODEL" in
            ouro) default_cache_tokens=100000 ;;
            huginn) default_cache_tokens=45000 ;;
        esac
        case "$MODEL:$DATASET" in
            ouro:sharegpt) default_widths="8 16 32 64 128" ;;
            huginn:sharegpt) default_widths="8 16"; default_decode_width=16 ;;
            ouro:alpaca) default_widths="8 16 32 64 128 256 512" ;;
            huginn:alpaca) default_widths="8 16 32 64 128 256" ;;
            ouro:arxiv) default_widths="2 4 8" ;;
            huginn:arxiv) default_widths=""; default_decode_width=1 ;;
        esac
    fi
    WIDTHS="${WIDTHS:-$default_widths}"
    DECODE_WIDTH="${DECODE_WIDTH:-$default_decode_width}"
    # Token capacity assumes full depth; auto uses memory-based sizing.
    CACHE_TOKENS="${CACHE_TOKENS:-$default_cache_tokens}"
    KV_CACHE_ARGS=()
    if [[ "$CACHE_TOKENS" != auto ]]; then
        KV_CACHE_ARGS=(--max-total-tokens "$CACHE_TOKENS")
    fi
}

# Benchmark prefixes of the converted research workloads.
NUM_REQUESTS_SHAREGPT="${NUM_REQUESTS_SHAREGPT:-6000}"
NUM_REQUESTS_ALPACA="${NUM_REQUESTS_ALPACA:-10000}"
NUM_REQUESTS_ARXIV="${NUM_REQUESTS_ARXIV:-1000}"
num_requests_for() {
    case "$1" in
        sharegpt) echo "$NUM_REQUESTS_SHAREGPT" ;;
        alpaca)   echo "$NUM_REQUESTS_ALPACA" ;;
        arxiv)    echo "$NUM_REQUESTS_ARXIV" ;;
        *) echo "unknown dataset '$1' (expected: sharegpt, alpaca, arxiv)" >&2; return 1 ;;
    esac
}

# Each replay json file contains the tokens and their exit depths for one threshold.
# Exits are recorded by shells/paper/workloads/record_workload.sh.
replay_workload_for() {
    local recording
    case "$1" in
        ouro)   recording=last_exited_early_exit_lookahead ;; # last-exited KV + lookahead
        huginn) recording=shared_convergence_lookahead ;; # shared KV + lookahead
        *) echo "unknown model '$1' (expected: ouro, huginn)" >&2; return 1 ;;
    esac
    case "$2" in
        sharegpt | alpaca | arxiv) ;;
        *) echo "unknown dataset '$2' (expected: sharegpt, alpaca, arxiv)" >&2; return 1 ;;
    esac
    echo "outputs/workloads/$2/$1/${recording}_q$3.jsonl"
}

wait_for_server() {
    local server_pid="$1" base_url="$2" timeout="${3:-600}"
    uv run --frozen --project "$REPO_ROOT" python - "$server_pid" "$base_url" "$timeout" <<'READY'
import os
import sys
import time
import urllib.error
import urllib.request

pid, base_url, timeout = int(sys.argv[1]), sys.argv[2].rstrip("/"), float(sys.argv[3])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=1) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        pass
    time.sleep(0.2)
else:
    raise TimeoutError(f"Server was not ready after {timeout} seconds: {base_url}")
READY
}

# Use a custom log directory
log_experiment() {
    local output_path="$1" run_name="$2"
    shift 2
    local log_dir log_name run_id
    log_dir="$(dirname -- "$output_path")/logs"
    run_id="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
    log_name="$(basename -- "${output_path%.jsonl}")_${run_name}_$run_id"
    mkdir -p -- "$log_dir"
    echo "Logs: $log_dir/$log_name.out and $log_dir/$log_name.err"
    "$@" >"$log_dir/$log_name.out" 2>"$log_dir/$log_name.err"
}
