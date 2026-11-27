#!/usr/bin/env bash
# Ablates Huginn's prelude-core-coda layer split for offline throughput (ShareGPT).
#
# For comparison, all layer splits use the exit depths recorded from the 2-4-2 model.
#
# Submit with: ./shells/_submit.sh shells/paper/ablations/layer_split_huginn.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL=huginn
DATASET=sharegpt
set_throughput_defaults

MODEL_ID="$HUGINN_MODEL_ID"
LOOP_STEPS=16
THRESHOLD=0.14 # earliest threshold from exit sweep
DECODE_WIDTH=64 # Maximum decode batch size used in the throughput sweeps

WORKLOAD="$(replay_workload_for "$MODEL" "$DATASET" "$THRESHOLD")"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/ablations/layer-split/layer-split_huginn_sharegpt.jsonl}"
NUM_REQUESTS="$(num_requests_for "$DATASET")"

SPLITS="${SPLITS:-0-4-0 1-4-1}"
BACKENDS="${BACKENDS:-cb cdb-norefill cdb-refill}"
MIN_CODA_BATCHES="${MIN_CODA_BATCHES:-1 8 16 32 64}"
REPEATS="${REPEATS:-1}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: prelude-core-coda layer split (offline throughput), $MODEL_ID, sharegpt, q=$THRESHOLD"
echo "Splits: $SPLITS (backends: $BACKENDS)"
echo "Max decode batch size: $DECODE_WIDTH"
echo "KV cache: $CACHE_POLICY, page size $PAGE_SIZE, CACHE_TOKENS=$CACHE_TOKENS"
echo "Requests:   $NUM_REQUESTS"
echo "Output:     $OUTPUT_PATH"
echo ""

for repeat in $(seq 0 $((REPEATS - 1))); do
    result_path="$OUTPUT_PATH"
    if ((REPEATS > 1)); then
        result_path="${OUTPUT_PATH%.jsonl}_repeat${repeat}.jsonl"
    fi
    for split in $SPLITS; do
        case "$split" in
            0-4-0|1-4-1) boundary="${split%%-*}" ;;
            *) echo "unknown SPLITS entry '$split' (expected: 0-4-0 or 1-4-1)" >&2; exit 1 ;;
        esac
        overrides="${HUGINN_MODEL_OVERRIDE_ARGS%\}},\"num_prelude_layers\":$boundary,\"num_core_layers\":4,\"num_coda_layers\":$boundary,\"num_hidden_layers\":$((4 + 2 * boundary))}"
        MODEL_ARGS=(--json-model-override-args "$overrides")
        for backend in $BACKENDS; do
            # Only cdb-refill batches codas.
            batches="-"
            [[ "$backend" == cdb-refill ]] && batches="$MIN_CODA_BATCHES"
            for min_coda in $batches; do
                echo "=== repeat: $repeat  split: $split  backend: $backend  min coda batch: $min_coda ==="
                case "$backend" in
                    cb) args=(--scheduler cb) ;;
                    cdb-norefill) args=(--scheduler cdb --no-refill --exit-threshold "$THRESHOLD") ;;
                    cdb-refill) args=(--scheduler cdb --exit-threshold "$THRESHOLD") ;;
                    *) echo "unknown backend '$backend' (expected: cb, cdb-norefill, cdb-refill)" >&2; exit 1 ;;
                esac
                if [[ "$CACHE_POLICY" == depth_indexed && "$backend" != cb ]]; then
                    echo "depth_indexed KV only supports CB; set BACKENDS=cb or select shared or last_exited." >&2
                    exit 1
                fi
                args+=(
                    --model-path "$MODEL_ID"
                    "${MODEL_ARGS[@]}"
                    --dataset-name custom --dataset-path "$WORKLOAD"
                    --custom-input-format token-ids --no-shuffle
                    --replay-exits
                    --cuda-event-timing
                    --cuda-graph-max-bs "$DECODE_WIDTH"
                    --loop-steps "$LOOP_STEPS"
                    --loop-cache-policy "$CACHE_POLICY"
                    --max-seq-len-override "$MAX_MODEL_LEN"
                    --max-running-requests "$DECODE_WIDTH"
                    --min-free-slots-delay "${MIN_FREE_SLOTS_DELAY:-$((DECODE_WIDTH / 8))}"
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
                )
                run_tag="${split}_w${DECODE_WIDTH}_${backend}"
                if [[ "$backend" == cdb-refill ]]; then
                    args+=(--min-coda-batch-size "$min_coda")
                    run_tag+="_k${min_coda}"
                fi
                # Use a custom log directory for each experiment
                log_experiment "$result_path" "$run_tag" \
                    uv run --frozen --project "$REPO_ROOT" \
                    python -u -m loopsgl.benchmark.offline_throughput "${args[@]}"
                echo ""
            done
        done
    done
done
