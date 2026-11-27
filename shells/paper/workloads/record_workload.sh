#!/usr/bin/env bash
# Record per-token early-exits for different workloads by forced decoding of reference responses.
# The model decodes each reference token by token with KV cache, but receives the reference token as the next input instead of its own prediction.
# Tokenized prompts and references are shared across thresholds and cache policies.
#
# Submit with MODEL=ouro DATASET=sharegpt ./shells/_submit.sh shells/paper/workloads/record_workload.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro or huginn}"
DATASET="${DATASET:?set DATASET to sharegpt, alpaca, or arxiv}"
TOKENIZED_DIR="${TOKENIZED_DIR:-outputs/workloads/tokenized_base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/workloads}"
DATA_ROOT="${DATA_ROOT:-data}"
SEED="${SEED:-0}"

case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        LOOP_STEPS=4
        CACHE_POLICY="${CACHE_POLICY:-last_exited}"
        EXIT_GATE_TYPE="${EXIT_GATE_TYPE:-early_exit}"
        THRESHOLDS="${THRESHOLDS:-0.0001 0.06 0.2 0.4 1.0}"
        MODEL_ARGS=(--json-model-override-args "{\"exit_gate_type\":\"$EXIT_GATE_TYPE\"}")
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        LOOP_STEPS=16
        CACHE_POLICY="${CACHE_POLICY:-shared}"
        EXIT_GATE_TYPE=convergence
        THRESHOLDS="${THRESHOLDS:-0.14 0.06 0.03 0.0}"
        MODEL_ARGS=(--json-model-override-args "$HUGINN_MODEL_OVERRIDE_ARGS")
        ;;
    *) echo "unknown MODEL '$MODEL' (expected ouro or huginn)" >&2; exit 1 ;;
esac
set_throughput_defaults

LOOKAHEAD_ARGS=()
exit_mode=lookahead
if [[ "${DISABLE_LOOKAHEAD_EXIT:-false}" == true ]]; then
    exit_mode=immediate
    LOOKAHEAD_ARGS=(--disable-lookahead-exit)
fi
if [[ "$MODEL" == ouro && "$EXIT_GATE_TYPE" == early_exit && "$exit_mode" == immediate ]]; then
    MODEL_ARGS=(--json-model-override-args '{"exit_gate_type":"early_exit","min_loop_steps":2}')
fi

case "$DATASET" in
    sharegpt)
        DATA_PATH="${DATA_PATH:-$DATA_ROOT/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json}"
        NUM_REQUESTS="${NUM_REQUESTS:-10000}"
        ;;
    alpaca)
        DATA_PATH="${DATA_PATH:-$DATA_ROOT/alpaca_data.json}"
        NUM_REQUESTS="${NUM_REQUESTS:-}"
        ;;
    arxiv)
        DATA_PATH="${DATA_PATH:-$DATA_ROOT/arxiv-summarization/document-train-0.parquet}"
        NUM_REQUESTS="${NUM_REQUESTS:-6000}"
        ;;
esac
tokenized_path="$TOKENIZED_DIR/$DATASET/$MODEL.jsonl"
LIMIT_ARGS=()
if [[ -n "$NUM_REQUESTS" ]]; then LIMIT_ARGS=(--limit "$NUM_REQUESTS"); fi

if [[ ! -f "$tokenized_path" ]]; then
    uv run --frozen --extra dev --project "$REPO_ROOT" \
        python -u "$REPO_ROOT/shells/paper/workloads/prepare_workload.py" \
        --dataset "$DATASET" --data-path "$DATA_PATH" --tokenizer "$MODEL_ID" \
        --filter-tokenizers "$OURO_1_4B_MODEL_ID" "$HUGINN_MODEL_ID" \
        --seed "$SEED" --max-model-len "$MAX_MODEL_LEN" "${LIMIT_ARGS[@]}" \
        --output-path "$tokenized_path"
fi

for threshold in $THRESHOLDS; do
    output_path="$OUTPUT_DIR/$DATASET/$MODEL/${CACHE_POLICY}_${EXIT_GATE_TYPE}_${exit_mode}_q$threshold.jsonl"
    echo "Recording $MODEL $DATASET with $CACHE_POLICY KV, $EXIT_GATE_TYPE gate, $exit_mode routing, threshold $threshold"
    uv run --frozen --project "$REPO_ROOT" \
        python -u "$REPO_ROOT/shells/paper/workloads/record_workload.py" \
        --model-path "$MODEL_ID" "${MODEL_ARGS[@]}" "${LOOKAHEAD_ARGS[@]}" \
        --dataset-path "$tokenized_path" --output-path "$output_path" \
        --scheduler cdb --loop-steps "$LOOP_STEPS" \
        --exit-threshold "$threshold" --loop-cache-policy "$CACHE_POLICY" \
        --max-seq-len-override "$MAX_MODEL_LEN" \
        --max-running-requests "$DECODE_WIDTH" \
        --cuda-graph-max-bs "$DECODE_WIDTH" "${KV_CACHE_ARGS[@]}" \
        --max-prefill-tokens "$MAX_PREFILL_TOKENS" --chunked-prefill-size -1 \
        --disable-radix-cache --retraction-mode none \
        --page-size "$PAGE_SIZE" --dtype "$DTYPE" \
        --attention-backend "$ATTENTION_BACKEND" --memory-ratio "$MEMORY_RATIO" \
        "$@"
done
