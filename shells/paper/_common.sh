#!/usr/bin/env bash
# Shared paper experiment configuration

set -euo pipefail

readonly OURO_1_4B_MODEL_ID="KristianS7/Ouro-1.4B"
readonly OURO_2_6B_MODEL_ID="KristianS7/Ouro-2.6B"
readonly HUGINN_MODEL_ID="KristianS7/huginn-0125"

# Throughput defaults are applied by the throughput scripts' CLI arguments.
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa}"
PAGE_SIZE="${PAGE_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_PREFILL_LENGTH="${MAX_PREFILL_LENGTH:-8192}"
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
            default_widths="2 4 8 16 32 64"
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
            huginn:sharegpt) default_widths="2 4 8 16"; default_decode_width=16 ;;
            ouro:alpaca) default_widths="8 16 32 64 128 256 512" ;;
            huginn:alpaca) default_widths="16 32 64 128 256" ;;
            ouro:arxiv) default_widths="2 4 8" ;;
            huginn:arxiv) default_widths="1"; default_decode_width=1 ;;
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

# Converted bundles retain the research filenames and full recurrent budgets.
workload_for() {
    local depth suffix
    case "$1" in
        ouro)   depth=4 ;;
        huginn) depth=16 ;;
        *) echo "unknown model '$1' (expected: ouro, huginn)" >&2; return 1 ;;
    esac
    case "$2" in
        sharegpt) suffix=_10k ;;
        alpaca)   suffix="" ;;
        arxiv)    suffix=_6k ;;
        *) echo "unknown dataset '$2' (expected: sharegpt, alpaca, arxiv)" >&2; return 1 ;;
    esac
    echo "outputs/workloads/${1}_${2}_recur${depth}${suffix}.jsonl"
}

# Each replay json file contains the final exit depths for one threshold.
replay_workload_for() {
    local workload
    workload="$(workload_for "$1" "$2")"
    echo "${workload%.jsonl}.q${3}.jsonl"
}
