#!/usr/bin/env bash
# Offline throughput against the exit threshold, at a fixed maximum decode batch size.
# By default compares CB, CDB-norefill, and CDB-refill.
#
# Exit thresholds are chosen to retain good GSM8k accuracy.
# Lower thresholds lead to fewer exits for Huginn and more for Ouro.
#
# Widths and cache sizes for each dataset and model are defined in _common.sh.
#
# Submit with:
# - MODEL=ouro DATASET=sharegpt ./shells/_submit.sh shells/paper/throughput_exit-sweep.sh
# - MODEL=ouro DATASET=sharegpt CACHE_POLICY=last_exited ./shells/_submit.sh shells/paper/throughput_exit-sweep.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro or huginn}"
DATASET="${DATASET:?set DATASET to sharegpt, alpaca, or arxiv}"
set_throughput_defaults

case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        LOOP_STEPS=4
        MIN_CODA_BATCH_SIZE=1
        THRESHOLDS="${THRESHOLDS:-0.2 0.4 0.5 0.7 1.0}"
        CONTROL=1.0
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        LOOP_STEPS=16
        MIN_CODA_BATCH_SIZE=$((DECODE_WIDTH / 4 > 0 ? DECODE_WIDTH / 4 : 1))
        THRESHOLDS="${THRESHOLDS:-0.28 0.16 0.1 0.0}"
        CONTROL=0.0
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

OUTPUT_PATH="${OUTPUT_PATH:-outputs/exit-sweep/exit-sweep_${MODEL}_${DATASET}_${CACHE_POLICY}.jsonl}"
NUM_REQUESTS="$(num_requests_for "$DATASET")"

BACKENDS="${BACKENDS:-cb cdb-norefill cdb-refill}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: drain throughput vs exit threshold, $MODEL_ID, $DATASET"
echo "Thresholds: $THRESHOLDS (no-exit control at $CONTROL)"
echo "Max decode batch size: $DECODE_WIDTH"
echo "KV cache: $CACHE_POLICY, page size $PAGE_SIZE, CACHE_TOKENS=$CACHE_TOKENS"
echo "Output:     $OUTPUT_PATH"
echo ""

for backend in $BACKENDS; do
    # cb decodes at full depth and ignores the threshold, so measure it once.
    if [ "$backend" = cb ]; then thresholds="${THRESHOLDS%% *}"; else thresholds="$THRESHOLDS"; fi
    for threshold in $thresholds; do
        WORKLOAD="$(replay_workload_for "$MODEL" "$DATASET" "$threshold")"
        echo "=== backend: $backend  threshold: $threshold ==="
        case "$backend" in
            cb) args=(--scheduler cb) ;;
            cdb-norefill) args=(--scheduler cdb --no-refill --exit-threshold "$threshold") ;;
            cdb-refill) args=(--scheduler cdb --exit-threshold "$threshold") ;;
            *) echo "unknown backend '$backend' (expected: cb, cdb-norefill, cdb-refill)" >&2; exit 1 ;;
        esac
        policy="$CACHE_POLICY"
        if [[ "$policy" == depth_indexed && "$backend" != cb ]]; then
            echo "depth_indexed KV only supports CB; set BACKENDS=cb or select shared or last_exited." >&2
            exit 1
        fi
        args+=(
            --model-path "$MODEL_ID"
            --dataset-name custom --dataset-path "$WORKLOAD"
            --custom-input-format token-ids --no-shuffle
            --replay-exits
            --loop-steps "$LOOP_STEPS"
            --loop-cache-policy "$policy"
            --max-seq-len-override "$MAX_MODEL_LEN"
            --max-running-requests "$DECODE_WIDTH"
            --min-free-slots-delay "${MIN_FREE_SLOTS_DELAY:-$((DECODE_WIDTH / 8 > 0 ? DECODE_WIDTH / 8 : 1))}"
            --dtype "$DTYPE"
            --attention-backend "$ATTENTION_BACKEND"
            "${NUM_PAGES_ARGS[@]}"
            --memory-ratio "$MEMORY_RATIO"
            --max-prefill-length "$MAX_PREFILL_LENGTH"
            --page-size "$PAGE_SIZE"
            "${RADIX_CACHE_ARGS[@]}"
            --watermark "$WATERMARK"
            --retraction-mode "$RETRACTION_MODE"
            --num-prompts "$NUM_REQUESTS"
            --result-filename "$OUTPUT_PATH"
        )
        [[ "$backend" == cdb-refill && -n "$MIN_CODA_BATCH_SIZE" ]] && args+=(--min-coda-batch-size "$MIN_CODA_BATCH_SIZE")
        uv run --frozen --project "$REPO_ROOT" \
            python -u -m loopsgl.benchmark.offline_throughput "${args[@]}"
        echo ""
    done
done
