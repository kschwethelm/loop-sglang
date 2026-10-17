#!/usr/bin/env bash
# Measures GPU idle time under CDB with synchronous scheduling, asynchronous scheduling, and lookahead.
# The analysis reports GPU busy/idle and per-NVTX-stage time.
#
# Submit with:
#   ./shells/_submit.sh shells/paper/ablations/benchmark_scheduling_overhead.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit GPU workloads through shells/_submit.sh." >&2; exit 1; }
cd "$REPO_ROOT"

NSYS_BIN="${NSYS_BIN:-nsys}"
command -v "$NSYS_BIN" >/dev/null || { echo "ERROR: nsys is required" >&2; exit 1; }
MODEL=ouro
DATASET=sharegpt
set_throughput_defaults
if [[ "$CACHE_POLICY" == depth_indexed ]]; then
    echo "depth_indexed KV only supports CB; select shared or last_exited." >&2
    exit 1
fi
MODEL_ID="$OURO_1_4B_MODEL_ID"
WORKLOAD="$(replay_workload_for ouro sharegpt 0.2)"
NUM_REQUESTS="$(num_requests_for "$DATASET")"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablations/scheduling-overhead}"
mkdir -p "$OUTPUT_DIR"
export LOOPSGL_ENABLE_NVTX=1

for variant in sync async lookahead; do
    flags=()
    export LOOPSGL_DISABLE_OVERLAP_SCHEDULING=0
    case "$variant" in
        sync)
            export LOOPSGL_DISABLE_OVERLAP_SCHEDULING=1
            flags+=(--disable-lookahead-exit)
            ;;
        async) flags+=(--disable-lookahead-exit) ;;
        lookahead) ;;
    esac
    rep="$OUTPUT_DIR/$variant.nsys-rep"
    sqlite="$OUTPUT_DIR/$variant.sqlite"
    for artifact in "$rep" "$sqlite" "$OUTPUT_DIR/$variant.jsonl" \
        "$OUTPUT_DIR/${variant}_analysis.json" "$OUTPUT_DIR/${variant}_analysis.csv"; do
        if [[ -e "$artifact" ]]; then
            echo "ERROR: refusing to overwrite $artifact" >&2
            exit 1
        fi
    done
    echo "Profiling $variant: graphs=all, predicted retirement, W=64, K=1, q=0.2"
    "$NSYS_BIN" profile \
        --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --cuda-graph-trace=graph --cuda-event-trace=false \
        --trace-fork-before-exec=true \
        --output="$rep" \
        uv run --frozen --project "$REPO_ROOT" python -m loopsgl.benchmark.offline_throughput \
        --scheduler cdb --refill --model-path "$MODEL_ID" \
        --dataset-name custom --dataset-path "$WORKLOAD" \
        --custom-input-format token-ids --no-shuffle --replay-exits \
        --exit-threshold 0.2 --loop-steps 4 --loop-cache-policy "$CACHE_POLICY" \
        --max-seq-len-override "$MAX_MODEL_LEN" --max-running-requests 64 \
        --min-coda-batch-size 1 --min-free-slots-delay 8 \
        --dtype "$DTYPE" --attention-backend "$ATTENTION_BACKEND" \
        "${NUM_PAGES_ARGS[@]}" --memory-ratio "$MEMORY_RATIO" \
        --page-size "$PAGE_SIZE" --max-prefill-length "$MAX_PREFILL_LENGTH" \
        "${RADIX_CACHE_ARGS[@]}" --watermark "$WATERMARK" --retraction-mode "$RETRACTION_MODE" \
        --cuda-graph-max-bs 64 --cuda-graph-backend-prefill breakable \
        --num-prompts "$NUM_REQUESTS" \
        --result-filename "$OUTPUT_DIR/$variant.jsonl" "${flags[@]}"
    "$NSYS_BIN" export --type=sqlite --output="$sqlite" "$rep"
    uv run --frozen --project "$REPO_ROOT" python -m loopsgl.benchmark.analyze_nsys \
        "$sqlite" \
        --output-json "$OUTPUT_DIR/${variant}_analysis.json" \
        --output-csv "$OUTPUT_DIR/${variant}_analysis.csv"
done

echo "Scheduling controls: $OUTPUT_DIR"
