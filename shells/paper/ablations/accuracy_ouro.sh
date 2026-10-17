#!/usr/bin/env bash
# GSM8K (gsm8k_cot) accuracy sweep for Ouro.
#
# Fixed-depth rows run on CB, otherwise on CDB.
#
# Usage: ./shells/_submit.sh shells/paper/ablations/accuracy_ouro.sh

set -euo pipefail

# Override _common.sh defaults
MIN_FREE_SLOTS_DELAY="${MIN_FREE_SLOTS_DELAY:-32}"
WATERMARK="${WATERMARK:-0.1}"
source "${REPO_ROOT:?Set REPO_ROOT to the repository root.}/shells/paper/_common.sh"

MODEL_ID="$OURO_1_4B_MODEL_ID"
# 3-shot matches the Ouro paper.
NUM_FEWSHOT="${NUM_FEWSHOT:-3}"
LOOP_STEPS="${LOOP_STEPS:-4}"
MIN_LOOP_STEPS="${MIN_LOOP_STEPS:-2}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/ablations/accuracy/ouro}"

FIXED_STEPS="${FIXED_STEPS-1 2 3 4}"
THRESHOLDS="${THRESHOLDS-0.01 0.1 0.2 0.25 0.3 0.4 0.5 0.7 0.9 1.0}"
# Every fixed depth runs under every layout.
FIXED_LAYOUTS="${FIXED_LAYOUTS-depth_indexed shared}"
# Test multiple exit gate policies.
GATES="${GATES-early_exit_immediate lookahead}"

TASK="${TASK:-gsm8k_cot}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-}"

eval_args=(
    --tasks "$TASK" --num_fewshot "$NUM_FEWSHOT" --batch_size auto
    --gen_kwargs do_sample=False --seed "$SEED" --log_samples
)
if [[ -n "$LIMIT" ]]; then
    eval_args+=(--limit "$LIMIT")
fi
for gate in $GATES; do
    case "$gate" in
        early_exit_immediate | lookahead) ;;
        *) echo "unknown gate '$gate' (expected: early_exit_immediate, lookahead)" >&2; exit 1 ;;
    esac
done

run_eval() {
    local scheduler="$1" policy="$2" steps="$3" gate="${4:-}" threshold="${5:-}"
    local page_size="$PAGE_SIZE" model_args result_path gate_type override_json
    if [[ "$policy" == last_exited ]]; then
        page_size=1
    fi
    model_args="pretrained=$MODEL_ID,scheduler=$scheduler,loop_cache_policy=$policy,loop_steps=$steps"
    model_args+=",attention_backend=$ATTENTION_BACKEND"
    model_args+=",page_size=$page_size,max_model_len=$MAX_MODEL_LEN"
    model_args+=",min_free_slots_delay=$MIN_FREE_SLOTS_DELAY,watermark=$WATERMARK,retraction_mode=$RETRACTION_MODE"
    model_args+=",disable_radix_cache=$DISABLE_RADIX_CACHE"
    result_path="${OUTPUT_PATH}/${scheduler}_${policy}_steps_${steps}"
    gate_type=early_exit
    if [[ "$scheduler" == cdb ]]; then
        model_args+=",exit_threshold=$threshold"
        case "$gate" in
            early_exit_immediate)
                model_args+=",disable_lookahead_exit=True"
                ;;
            lookahead) gate_type=lookahead ;;
        esac
        result_path+="_${gate}_threshold_${threshold}"
    fi
    local -a model_override_args=()
    if [[ "$scheduler" == cdb ]]; then
        override_json="$(python3 -c 'import json, sys; print(json.dumps({"json_model_override_args": json.dumps({"exit_gate_type": sys.argv[1], "min_loop_steps": int(sys.argv[2])})}))' "$gate_type" "$MIN_LOOP_STEPS")"
        model_override_args=(--model_args "$override_json")
    fi

    mkdir -p -- "$result_path"
    echo "=== $MODEL_ID: $TASK, ${NUM_FEWSHOT}-shot, $scheduler, depth $steps, KV $policy${gate:+, $gate, threshold $threshold} ==="
    uv run --frozen --extra eval --project "$REPO_ROOT" \
        python -u -m loopsgl.evaluation run --model loopsgl \
        --model_args "$model_args" "${model_override_args[@]}" \
        "${eval_args[@]}" --output_path "$result_path"
}

for policy in $FIXED_LAYOUTS; do
    case "$policy" in
        depth_indexed | shared) ;;
        *) echo "unknown fixed layout '$policy' (expected: depth_indexed, shared)" >&2; exit 1 ;;
    esac
done

for policy in $FIXED_LAYOUTS; do
    for steps in $FIXED_STEPS; do
        run_eval cb "$policy" "$steps"
    done
done

for gate in $GATES; do
    for threshold in $THRESHOLDS; do
        run_eval cdb last_exited "$LOOP_STEPS" "$gate" "$threshold"
    done
done
