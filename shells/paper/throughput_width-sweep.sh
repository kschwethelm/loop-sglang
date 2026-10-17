#!/usr/bin/env bash
# Offline throughput against the maximum decode batch size, at a fixed exit threshold.
# By default compares CB, CDB-norefill, and CDB-refill.
#
# Maximum decode batch sizes are chosen based on saturation batch size B* for each dataset.
# All runs are on average close to the configured maximum decode batch size.
#
# Widths and cache sizes for each dataset and model are defined in _common.sh.
#
# Submit with:
# - MODEL=ouro DATASET=sharegpt ./shells/_submit.sh shells/paper/throughput_width-sweep.sh
# - MODEL=ouro DATASET=sharegpt CACHE_POLICY=last_exited ./shells/_submit.sh shells/paper/throughput_width-sweep.sh

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
        THRESHOLD=0.2
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        LOOP_STEPS=16
        MIN_CODA_BATCH_SIZE=quarter
        THRESHOLD=0.28
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

WORKLOAD="$(replay_workload_for "$MODEL" "$DATASET" "$THRESHOLD")"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/width-sweep/width-sweep_${MODEL}_${DATASET}_${CACHE_POLICY}.jsonl}"
NUM_REQUESTS="$(num_requests_for "$DATASET")"

BACKENDS="${BACKENDS:-cb cdb-norefill cdb-refill}"
REPEATS="${REPEATS:-1}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: drain throughput vs maximum decode batch size, $MODEL_ID, $DATASET, q=$THRESHOLD"
echo "Max decode batch sizes: $WIDTHS"
echo "KV cache: $CACHE_POLICY, page size $PAGE_SIZE, CACHE_TOKENS=$CACHE_TOKENS"
echo "Requests:   $NUM_REQUESTS"
echo "Output:     $OUTPUT_PATH"
echo ""

for repeat in $(seq 0 $((REPEATS - 1))); do
    result_path="$OUTPUT_PATH"
    if ((REPEATS > 1)); then
        result_path="${OUTPUT_PATH%.jsonl}_repeat${repeat}.jsonl"
    fi
    for width in $WIDTHS; do
        for backend in $BACKENDS; do
            echo "=== repeat: $repeat  width: $width  backend: $backend ==="
            case "$backend" in
                cb) args=(--scheduler cb) ;;
                cdb-norefill) args=(--scheduler cdb --no-refill --exit-threshold "$THRESHOLD") ;;
                cdb-refill) args=(--scheduler cdb --exit-threshold "$THRESHOLD") ;;
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
                --max-running-requests "$width"
                --min-free-slots-delay "${MIN_FREE_SLOTS_DELAY:-$((width / 8 > 0 ? width / 8 : 1))}"
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
                --result-filename "$result_path"
            )
            if [[ "$backend" == cdb-refill && -n "$MIN_CODA_BATCH_SIZE" ]]; then
                min_coda=$MIN_CODA_BATCH_SIZE
                if [[ "$min_coda" == quarter ]]; then
                    min_coda=$((width / 4))
                    if ((min_coda < 1)); then
                        min_coda=1
                    fi
                fi
                args+=(--min-coda-batch-size "$min_coda")
            fi
            uv run --frozen --project "$REPO_ROOT" \
                python -u -m loopsgl.benchmark.offline_throughput "${args[@]}"
            echo ""
        done
    done
done
