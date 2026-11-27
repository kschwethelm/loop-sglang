#!/usr/bin/env bash
# Ablates the minimum coda batch size for offline throughput (ShareGPT); only relevant for CDB with refill.
#
# Sweeps the minimum coda batch size per maximum decode batch size.
# MIN_CODA_BATCHES accepts absolute sizes and the fractions 'eighth', 'quarter', and 'half'.
# Fractional sizes are clamped to one.
#
# Submit with:
#   MODEL=ouro   ./shells/_submit.sh shells/paper/ablations/min_coda_batch.sh
#   MODEL=huginn ./shells/_submit.sh shells/paper/ablations/min_coda_batch.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro or huginn}"
DATASET=sharegpt
set_throughput_defaults

MODEL_ARGS=()
case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        LOOP_STEPS=4
        THRESHOLD=0.0001 # earliest threshold from exit sweep
        DECODE_WIDTHS="${DECODE_WIDTHS:-64}"
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        MODEL_ARGS=(--json-model-override-args "$HUGINN_MODEL_OVERRIDE_ARGS")
        LOOP_STEPS=16
        THRESHOLD=0.14 # earliest threshold from exit sweep
        DECODE_WIDTHS="${DECODE_WIDTHS:-16 32 64 512}"
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

if [[ "$CACHE_POLICY" == depth_indexed ]]; then
    echo "depth_indexed KV only supports CB; select shared or last_exited." >&2
    exit 1
fi

WORKLOAD="$(replay_workload_for "$MODEL" "$DATASET" "$THRESHOLD")"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/ablations/min-coda-batch/min-coda-batch_${MODEL}_sharegpt.jsonl}"
NUM_REQUESTS="$(num_requests_for "$DATASET")"

MIN_CODA_BATCHES="${MIN_CODA_BATCHES:-1 eighth quarter half}"
REPEATS="${REPEATS:-1}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: minimum coda batch (offline throughput), $MODEL_ID, sharegpt, q=$THRESHOLD"
echo "Minimum coda batch sizes: $MIN_CODA_BATCHES"
echo "Max decode batch sizes: $DECODE_WIDTHS"
echo "KV cache: $CACHE_POLICY, page size $PAGE_SIZE, CACHE_TOKENS=$CACHE_TOKENS"
echo "Requests:   $NUM_REQUESTS"
echo "Output:     $OUTPUT_PATH"
echo ""

declare -A seen=()
for repeat in $(seq 0 $((REPEATS - 1))); do
    result_path="$OUTPUT_PATH"
    if ((REPEATS > 1)); then
        result_path="${OUTPUT_PATH%.jsonl}_repeat${repeat}.jsonl"
    fi
    seen=()
    for width in $DECODE_WIDTHS; do
        for min_coda in $MIN_CODA_BATCHES; do
            case "$min_coda" in
                eighth)  min_coda=$((width / 8 > 0 ? width / 8 : 1)) ;;
                quarter) min_coda=$((width / 4 > 0 ? width / 4 : 1)) ;;
                half)    min_coda=$((width / 2 > 0 ? width / 2 : 1)) ;;
            esac
            # Small widths can map multiple fractions to the same K.
            [[ -z "${seen[$width:$min_coda]:-}" ]] || continue
            seen[$width:$min_coda]=1

            echo "=== repeat: $repeat  width: $width  min coda batch: $min_coda ==="
            args=(
                --scheduler cdb --exit-threshold "$THRESHOLD"
                --model-path "$MODEL_ID"
                "${MODEL_ARGS[@]}"
                --dataset-name custom --dataset-path "$WORKLOAD"
                --custom-input-format token-ids --no-shuffle
                --replay-exits
                --cuda-event-timing
                --cuda-graph-max-bs "$width"
                --loop-steps "$LOOP_STEPS"
                --loop-cache-policy "$CACHE_POLICY"
                --max-seq-len-override "$MAX_MODEL_LEN"
                --max-running-requests "$width"
                --min-free-slots-delay "${MIN_FREE_SLOTS_DELAY:-$((width / 8 > 0 ? width / 8 : 1))}"
                --dtype "$DTYPE"
                --attention-backend "$ATTENTION_BACKEND"
                "${KV_CACHE_ARGS[@]}"
                --memory-ratio "$MEMORY_RATIO"
                --max-prefill-tokens "$MAX_PREFILL_TOKENS"
                --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
                --page-size "$PAGE_SIZE"
                "${RADIX_CACHE_ARGS[@]}"
                --watermark "$WATERMARK"
                --retraction-mode "$RETRACTION_MODE"
                --num-prompts "$NUM_REQUESTS"
                --result-filename "$result_path"
                --min-coda-batch-size "$min_coda"
            )
            # Use a custom log directory for each experiment
            log_experiment "$result_path" "default_w${width}_cdb-refill_k${min_coda}" \
                uv run --frozen --project "$REPO_ROOT" \
                python -u -m loopsgl.benchmark.offline_throughput "${args[@]}"
            echo ""
        done
    done
done
