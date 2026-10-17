#!/usr/bin/env bash
# GSM8K (gsm8k_cot) accuracy sweep for Huginn.
#
# Fixed-depth rows run on CB, otherwise on CDB.
#
# Usage: ./shells/_submit.sh shells/paper/ablations/accuracy_huginn.sh

set -euo pipefail

# Override _common.sh defaults
MIN_FREE_SLOTS_DELAY="${MIN_FREE_SLOTS_DELAY:-32}"
WATERMARK="${WATERMARK:-0.1}"
source "${REPO_ROOT:?Set REPO_ROOT to the repository root.}/shells/paper/_common.sh"

MODEL_ID="$HUGINN_MODEL_ID"
# 8-shot matches the Huginn paper.
NUM_FEWSHOT="${NUM_FEWSHOT:-8}"
# Below the checkpoint's trained 32: accuracy is flat within noise above 16.
LOOP_STEPS="${LOOP_STEPS:-16}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/ablations/accuracy/huginn}"

FIXED_STEPS="${FIXED_STEPS-6 7 8 9 10 12 14 16 20 24 32}"
THRESHOLDS="${THRESHOLDS-0.02 0.06 0.1 0.14 0.16 0.2 0.24 0.28 0.3 0.34}"
# Every fixed depth runs under every layout.
FIXED_LAYOUTS="${FIXED_LAYOUTS-depth_indexed shared}"
# Every gated row runs under every layout.
GATED_LAYOUTS="${GATED_LAYOUTS-shared last_exited}"

TASK="${TASK:-gsm8k_cot}"
# One quarter of the engine's default request cap of 256.
MIN_CODA_BATCH_SIZE="${MIN_CODA_BATCH_SIZE:-64}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-}"

eval_args=(
    --tasks "$TASK" --num_fewshot "$NUM_FEWSHOT" --batch_size auto
    --gen_kwargs do_sample=False --seed "$SEED" --log_samples
)
if [[ -n "$LIMIT" ]]; then
    eval_args+=(--limit "$LIMIT")
fi

run_eval() {
    local scheduler="$1" policy="$2" steps="$3" threshold="${4:-}" timing="${5:-delayed}"
    local page_size="$PAGE_SIZE" model_args result_path
    if [[ "$policy" == last_exited ]]; then
        page_size=1
    fi

    model_args="pretrained=$MODEL_ID,scheduler=$scheduler,loop_cache_policy=$policy,loop_steps=$steps"
    model_args+=",attention_backend=$ATTENTION_BACKEND"
    model_args+=",page_size=$page_size,max_model_len=$MAX_MODEL_LEN"
    model_args+=",min_free_slots_delay=$MIN_FREE_SLOTS_DELAY,watermark=$WATERMARK,retraction_mode=$RETRACTION_MODE"
    model_args+=",disable_radix_cache=$DISABLE_RADIX_CACHE"
    result_path="${OUTPUT_PATH}/${scheduler}_${policy}_steps_${steps}"
    if [[ "$scheduler" == cdb ]]; then
        model_args+=",exit_threshold=$threshold,min_coda_batch_size=$MIN_CODA_BATCH_SIZE"
        if [[ "$timing" == immediate ]]; then
            model_args+=",disable_lookahead_exit=True"
        fi
        result_path+="_${timing}_threshold_${threshold}"
    fi
    local -a model_override_args=()
    if [[ -n "${JSON_MODEL_OVERRIDE_ARGS:-}" ]]; then
        local override_json
        override_json="$(python3 -c 'import json, sys; print(json.dumps({"json_model_override_args": sys.argv[1]}))' "$JSON_MODEL_OVERRIDE_ARGS")"
        model_override_args=(--model_args "$override_json")
    fi

    mkdir -p -- "$result_path"
    echo "=== $MODEL_ID: $TASK, ${NUM_FEWSHOT}-shot, $scheduler, depth $steps, KV $policy${threshold:+, $timing, threshold $threshold} ==="
    uv run --frozen --extra eval --project "$REPO_ROOT" \
        python -u -m loopsgl.evaluation run --model loopsgl \
        --model_args "$model_args" "${model_override_args[@]}" \
        "${eval_args[@]}" --output_path "$result_path"
}

mkdir -p -- "$OUTPUT_PATH"
echo "Experiment: $TASK accuracy sweep, $MODEL_ID (${NUM_FEWSHOT}-shot)"
echo "Fixed:      ${FIXED_STEPS:-none} over layouts: ${FIXED_LAYOUTS:-none}"
echo "Thresholds: ${THRESHOLDS:-none} over layouts: ${GATED_LAYOUTS:-none} (at depth $LOOP_STEPS)"
echo "Limit:      ${LIMIT:-full test set}"
echo "Output:     $OUTPUT_PATH"

for layout in $FIXED_LAYOUTS; do
    for steps in $FIXED_STEPS; do
        run_eval cb "$layout" "$steps"
    done
done

for layout in $GATED_LAYOUTS; do
    for threshold in $THRESHOLDS; do
        run_eval cdb "$layout" "$LOOP_STEPS" "$threshold" immediate
        run_eval cdb "$layout" "$LOOP_STEPS" "$threshold" delayed
    done
done
