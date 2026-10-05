#!/usr/bin/env bash
# Evaluate 3-shot GSM8K CoT for Ouro
# Usage: ./shells/_submit.sh shells/paper/ablations/accuracy_ouro.sh

set -euo pipefail

source "${REPO_ROOT:?Set REPO_ROOT to the repository root.}/shells/paper/_common.sh"
MODEL_ID="$OURO_1_4B_MODEL_ID"
CACHE_POLICY="${CACHE_POLICY:-depth_indexed}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/ablations/accuracy/ouro}"
LOOP_STEPS="${LOOP_STEPS:-4}"

exec uv run --frozen --extra eval --project "$REPO_ROOT" \
    python -m loopsgl.evaluation run --model loopsgl \
    --model_args "pretrained=$MODEL_ID,loop_cache_policy=$CACHE_POLICY,max_gen_toks=256,loop_steps=$LOOP_STEPS" \
    --tasks gsm8k_cot --num_fewshot 3 --batch_size auto \
    --output_path "$OUTPUT_PATH"
