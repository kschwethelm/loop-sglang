#!/usr/bin/env bash
# Sweep static-batch decode latency over context length and recurrent depth.
# Usage: MODEL=ouro ./shells/_submit.sh shells/paper/benchmark_decode_step_latency.sh

set -euo pipefail
cache_tokens="${CACHE_TOKENS:-}"
source "${REPO_ROOT:?Submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?Set MODEL to ouro, ouro26, or huginn.}"
MODEL_ARGS=()
case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        LOOP_STEPS="${LOOP_STEPS:-1 2 3 4}"
        ;;
    ouro26)
        MODEL_ID="$OURO_2_6B_MODEL_ID"
        LOOP_STEPS="${LOOP_STEPS:-1 2 3 4}"
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        LOOP_STEPS="${LOOP_STEPS:-2 4 8 12 16}"
        # Extend Huginn's RoPE table to the configured benchmark sequence limit.
        MODEL_ARGS=(--json-model-override-args "{\"max_position_embeddings\":$MAX_MODEL_LEN}")
        ;;
    *) echo "Unknown MODEL '$MODEL' (expected: ouro, ouro26, huginn)." >&2; exit 1 ;;
esac

CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-128 512 2048 8192}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64 128 256 512}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-16}"
CACHE_POLICY="${CACHE_POLICY:-shared}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/decode_step_latency/${MODEL}_${CACHE_POLICY}.jsonl}"
mkdir -p -- "$(dirname -- "$OUTPUT_PATH")"
read -r -a BATCH_SIZE_ARGS <<< "$BATCH_SIZES"
KV_CACHE_ARGS=(--memory-ratio "${MEMORY_RATIO:-0.9}")
if [[ -n "$cache_tokens" ]]; then
    KV_CACHE_ARGS=(--num-pages "$((cache_tokens / PAGE_SIZE))")
fi

for context_length in $CONTEXT_LENGTHS; do
    for loop_steps in $LOOP_STEPS; do
        echo "Decode latency: model=$MODEL context=$context_length loop_steps=$loop_steps cache=$CACHE_POLICY"
        uv run --frozen --project "$REPO_ROOT" python -u -m loopsgl.benchmark.one_batch \
            --model-path "$MODEL_ID" \
            "${MODEL_ARGS[@]}" \
            --batch-size "${BATCH_SIZE_ARGS[@]}" \
            --input-len "$context_length" --output-len "$OUTPUT_LENGTH" \
            --loop-steps "$loop_steps" --loop-cache-policy "$CACHE_POLICY" \
            --dtype "$DTYPE" --attention-backend "$ATTENTION_BACKEND" \
            --page-size "$PAGE_SIZE" \
            --max-seq-len-override "$MAX_MODEL_LEN" \
            --max-prefill-length "$MAX_PREFILL_LENGTH" \
            "${KV_CACHE_ARGS[@]}" \
            --result-filename "$OUTPUT_PATH" \
            --profile-prefix "${OUTPUT_PATH%.jsonl}_steps${loop_steps}" \
            --run-name "${MODEL}_${CACHE_POLICY}_steps${loop_steps}_ctx${context_length}" \
            "$@"
    done
done
