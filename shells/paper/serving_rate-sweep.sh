#!/usr/bin/env bash
# Open-loop serving latency against the offered request rate.
#
# Each point uses a seeded Poisson arrival trace shared across backends at that rate.
#
# The engine uses the same admission parameters as the offline throughput sweeps.
# The rate grids span the saturation knees of all backends on the H100 used for the paper.
#
# Submit with:
# - MODEL=ouro DATASET=sharegpt ./shells/_submit.sh shells/paper/serving_rate-sweep.sh
# - MODEL=ouro DATASET=sharegpt CACHE_POLICY=last_exited ./shells/_submit.sh shells/paper/serving_rate-sweep.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

MODEL="${MODEL:?set MODEL to ouro or huginn}"
DATASET="${DATASET:?set DATASET to sharegpt, alpaca, or arxiv}"
set_throughput_defaults

MODEL_ARGS=()
case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        LOOP_STEPS=4
        MIN_CODA_BATCH_SIZE=1
        THRESHOLDS="${THRESHOLDS:-0.0001 0.06 0.2 0.4}"
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        MODEL_ARGS=(--json-model-override-args "$HUGINN_MODEL_OVERRIDE_ARGS")
        LOOP_STEPS=16
        MIN_CODA_BATCH_SIZE=$((DECODE_WIDTH / 4 > 0 ? DECODE_WIDTH / 4 : 1))
        THRESHOLDS="${THRESHOLDS:-0.14 0.06 0.03}"
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

case "$MODEL/$DATASET" in
    ouro/sharegpt)
        DEFAULT_RATES="5 6 7 8 10 12"
        ;;
    ouro/alpaca)
        DEFAULT_RATES="45 55 65 75 90 105"
        ;;
    ouro/arxiv)
        DEFAULT_RATES="0.45 0.52 0.58 0.65 0.72 0.80"
        ;;
    huginn/sharegpt)
        DEFAULT_RATES="1.5 2 2.3 2.6 3 3.5"
        ;;
    huginn/alpaca)
        DEFAULT_RATES="12 16 18 21 24 28"
        ;;
    huginn/arxiv)
        DEFAULT_RATES="0.16 0.20 0.23 0.26 0.30 0.34"
        ;;
    *) echo "unknown DATASET '$DATASET' (expected: sharegpt, alpaca, arxiv)" >&2; exit 1 ;;
esac
RATES="${RATES:-$DEFAULT_RATES}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/serving-rate/serving-rate_${MODEL}_${DATASET}_${CACHE_POLICY}.jsonl}"

BACKENDS="${BACKENDS:-cb cdb-norefill cdb-refill}"
REPEATS="${REPEATS:-1}"
ARRIVAL_SEED="${ARRIVAL_SEED:-0}"
# Each trace contains ceil(rate * TRACE_DURATION_S) requests and drains fully.
TRACE_DURATION_S="${TRACE_DURATION_S:-600}"

server_pid=""
cleanup_server() {
    if [[ -n "$server_pid" ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for ((attempt=0; attempt<100; attempt++)); do
            kill -0 -- "-$server_pid" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
    fi
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "Experiment: open-loop serving vs request rate, $MODEL_ID, $DATASET"
echo "Thresholds: $THRESHOLDS"
echo "Rates:      $RATES req/s (${TRACE_DURATION_S}s target trace duration, arrival seed $ARRIVAL_SEED)"
echo "Max decode batch size: $DECODE_WIDTH"
echo "KV cache: $CACHE_POLICY, page size $PAGE_SIZE, CACHE_TOKENS=$CACHE_TOKENS"
echo "Output:     $OUTPUT_PATH"
echo ""

for repeat in $(seq 0 $((REPEATS - 1))); do
    for backend in $BACKENDS; do
        if [ "$backend" = cb ]; then thresholds="${THRESHOLDS%% *}"; else thresholds="$THRESHOLDS"; fi
        for threshold in $thresholds; do
            WORKLOAD="$(replay_workload_for "$MODEL" "$DATASET" "$threshold")"
            for rate in $RATES; do
                echo "=== repeat: $repeat  threshold: $threshold  rate: $rate req/s  backend: $backend ==="
                case "$backend" in
                    cb) args=(--scheduler cb) ;;
                    cdb-norefill) args=(--scheduler cdb --no-refill --exit-threshold "$threshold") ;;
                    cdb-refill) args=(--scheduler cdb --exit-threshold "$threshold") ;;
                    *) echo "unknown backend '$backend' (expected: cb, cdb-norefill, cdb-refill)" >&2; exit 1 ;;
                esac
                if [[ "$CACHE_POLICY" == depth_indexed && "$backend" != cb ]]; then
                    echo "depth_indexed KV only supports CB; set BACKENDS=cb or select shared or last_exited." >&2
                    exit 1
                fi
                port="${SERVER_PORT:-$(uv run --frozen --project "$REPO_ROOT" python -c 'from loopsgl.utils import get_free_port; print(get_free_port())')}"
                args+=(
                    --model-path "$MODEL_ID"
                    "${MODEL_ARGS[@]}"
                    --host 127.0.0.1 --port "$port"
                    --cuda-graph-max-bs "$DECODE_WIDTH"
                    --loop-steps "$LOOP_STEPS"
                    --loop-cache-policy "$CACHE_POLICY"
                    --max-seq-len-override "$MAX_MODEL_LEN"
                    --max-running-requests "$DECODE_WIDTH"
                    --min-free-slots-delay "${MIN_FREE_SLOTS_DELAY:-$((DECODE_WIDTH / 8 > 0 ? DECODE_WIDTH / 8 : 1))}"
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
                )
                [[ "$backend" == cdb-refill ]] && args+=(--min-coda-batch-size "$MIN_CODA_BATCH_SIZE")
                run_name="r${repeat}_${backend}_q${threshold}_rate${rate}"
                log_dir="$(dirname -- "$OUTPUT_PATH")/logs"
                run_id="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
                server_log="$log_dir/$(basename -- "${OUTPUT_PATH%.jsonl}")_${run_name}_server_$run_id"
                mkdir -p "$log_dir"
                echo "Server logs: $server_log.out and $server_log.err"
                setsid uv run --frozen --project "$REPO_ROOT" \
                    python -u -m loopsgl.launch_server "${args[@]}" >"$server_log.out" 2>"$server_log.err" &
                server_pid=$!
                wait_for_server "$server_pid" "http://127.0.0.1:$port" "${SERVER_STARTUP_TIMEOUT_S:-600}"
                log_experiment "$OUTPUT_PATH" "${run_name}_client" \
                    uv run --frozen --project "$REPO_ROOT" \
                    python -u -m loopsgl.benchmark.serving \
                    --base-url "http://127.0.0.1:$port" --model "$MODEL_ID" \
                    --dataset-name custom --dataset-path "$WORKLOAD" \
                    --custom-input-format token-ids --no-shuffle --replay-exits \
                    --request-rate "$rate" --trace-duration-s "$TRACE_DURATION_S" \
                    --arrival-seed "$((ARRIVAL_SEED + repeat))" --repeat-index "$repeat" \
                    --ready-check-timeout-sec 0 --flush-cache --output-file "$OUTPUT_PATH"
                if ! kill -0 "$server_pid" 2>/dev/null; then
                    wait "$server_pid"
                    echo "Server exited before point cleanup." >&2
                    exit 1
                fi
                cleanup_server
                echo ""
            done
        done
    done
done
