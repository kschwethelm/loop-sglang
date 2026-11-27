#!/usr/bin/env bash
# Offline throughput with concurrency limited by fixed KV-cache memory budgets.
# Compares full-depth CB with shared/depth-indexed KV and adaptive CDB-refill with shared/last-exited KV.
# RUNS lists backend:policy pairs and defaults to all four configurations.
# cdb-refill:last_exited_copy runs the copying last-exited cache, which keeps all depth slots
# and uses the depth-indexed capacity of each budget.
#
# Submit with:
# - MODEL=ouro DATASET=sharegpt ./shells/_submit.sh shells/paper/ablations/benchmark_memory.sh
# - MODEL=ouro DATASET=arxiv ./shells/_submit.sh shells/paper/ablations/benchmark_memory.sh
# - MODEL=ouro DATASET=sharegpt RUNS=cb:depth_indexed ./shells/_submit.sh shells/paper/ablations/benchmark_memory.sh
# - MODEL=ouro DATASET=sharegpt RUNS=cdb-refill:last_exited_copy ./shells/_submit.sh shells/paper/ablations/benchmark_memory.sh

set -euo pipefail
WATERMARK="${WATERMARK:-0.3}"
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro}"
DATASET="${DATASET:?set DATASET to sharegpt or arxiv}"
case "$MODEL" in
    ouro) MODEL_ID="$OURO_1_4B_MODEL_ID" ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro)" >&2; exit 1 ;;
esac
case "$DATASET" in
    sharegpt | arxiv) ;;
    *) echo "unknown DATASET '$DATASET' (expected: sharegpt, arxiv)" >&2; exit 1 ;;
esac

LOOP_STEPS=4
THRESHOLD=0.0001
WORKLOAD="$(replay_workload_for "$MODEL" "$DATASET" "$THRESHOLD")"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/ablations/memory/memory_${MODEL}_${DATASET}.jsonl}"
NUM_REQUESTS="$(num_requests_for "$DATASET")"
MEMORY_BUDGETS="${MEMORY_BUDGETS:-20 45 70}"
DECODE_WIDTH="${DECODE_WIDTH:-2048}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-1280}"
RUNS="${RUNS:-cb:shared cb:depth_indexed cdb-refill:shared cdb-refill:last_exited}"
REPEATS="${REPEATS:-1}"

for run in $RUNS; do
    case "$run" in
        cb:shared | cb:depth_indexed | cdb-refill:shared | cdb-refill:last_exited | cdb-refill:last_exited_copy) ;;
        *)
            echo "unknown run '$run' (expected: cb:shared, cb:depth_indexed, cdb-refill:shared, cdb-refill:last_exited, cdb-refill:last_exited_copy)" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: memory-constrained throughput, $MODEL_ID, $DATASET, q=$THRESHOLD"
echo "Memory budgets (GiB): $MEMORY_BUDGETS"
echo "Runs:       $RUNS"
echo "Request cap: $DECODE_WIDTH; graph cap: $CUDA_GRAPH_MAX_BS"
echo "Watermark:  $WATERMARK"
echo "Requests:   $NUM_REQUESTS"
echo "Output:     $OUTPUT_PATH"
echo ""

for repeat in $(seq 0 $((REPEATS - 1))); do
    for budget in $MEMORY_BUDGETS; do
        case "$budget" in
            20)
                shared_cache_tokens=109200
                depth_indexed_cache_tokens=27280
                last_exited_cache_tokens=27302
                ;;
            45)
                shared_cache_tokens=245728
                depth_indexed_cache_tokens=61408
                last_exited_cache_tokens=61431
                ;;
            70)
                shared_cache_tokens=382272
                depth_indexed_cache_tokens=95552
                last_exited_cache_tokens=95559
                ;;
            *) echo "unknown memory budget '$budget' (expected: 20, 45, 70 GiB)" >&2; exit 1 ;;
        esac
        result_path="${OUTPUT_PATH%.jsonl}_${budget}gib.jsonl"
        if ((REPEATS > 1)); then
            result_path="${OUTPUT_PATH%.jsonl}_${budget}gib_repeat${repeat}.jsonl"
        fi
        echo "KV budget: $budget GiB"
        echo "KV token capacities: shared=$shared_cache_tokens, depth_indexed/last_exited_copy=$depth_indexed_cache_tokens, last_exited=$last_exited_cache_tokens"
        echo "Output: $result_path"
        echo ""
        for run in $RUNS; do
            backend="${run%%:*}"
            policy="${run#*:}"
            echo "=== repeat: $repeat  budget: $budget GiB  backend: $backend  KV cache: $policy ==="
            case "$backend" in
                cb) args=(--scheduler cb) ;;
                cdb-refill) args=(--scheduler cdb --refill --exit-threshold "$THRESHOLD") ;;
            esac
            case "$policy" in
                shared) cache_tokens="$shared_cache_tokens"; page_size=16 ;;
                depth_indexed) cache_tokens="$depth_indexed_cache_tokens"; page_size=16 ;;
                last_exited) cache_tokens="$last_exited_cache_tokens"; page_size=1 ;;
                last_exited_copy) cache_tokens="$depth_indexed_cache_tokens"; page_size=16 ;;
            esac
            args+=(
                --model-path "$MODEL_ID"
                --dataset-name custom --dataset-path "$WORKLOAD"
                --custom-input-format token-ids --no-shuffle
                --replay-exits
                --cuda-event-timing
                --loop-steps "$LOOP_STEPS"
                --loop-cache-policy "$policy"
                --max-seq-len-override "$MAX_MODEL_LEN"
                --max-running-requests "$DECODE_WIDTH"
                --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
                --min-free-slots-delay 1
                --min-coda-batch-size 1
                --dtype "$DTYPE"
                --attention-backend "$ATTENTION_BACKEND"
                --max-total-tokens "$cache_tokens"
                --memory-ratio "$MEMORY_RATIO"
                --max-prefill-tokens "$MAX_PREFILL_TOKENS"
                --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
                --page-size "$page_size"
                "${RADIX_CACHE_ARGS[@]}"
                --watermark "$WATERMARK"
                --retraction-mode none
                --num-prompts "$NUM_REQUESTS"
                --result-filename "$result_path"
            )
            log_experiment "$result_path" "${backend}_${policy}" \
                uv run --frozen --project "$REPO_ROOT" \
                python -u -m loopsgl.benchmark.offline_throughput "${args[@]}"
            echo ""
        done
    done
done
