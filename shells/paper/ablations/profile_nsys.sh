#!/usr/bin/env bash
# Profiling CB and CDB via Nsight Systems trace of the offline throughput.
# The analysis reports GPU busy/idle and per-NVTX-stage time.
#
# Submit with:
#   MODEL=ouro   ./shells/_submit.sh shells/paper/ablations/profile_nsys.sh
#   MODEL=huginn ./shells/_submit.sh shells/paper/ablations/profile_nsys.sh
#
# DATASET picks the replayed workload (sharegpt by default):
#   MODEL=ouro DATASET=alpaca ./shells/_submit.sh shells/paper/ablations/profile_nsys.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit GPU workloads through shells/_submit.sh." >&2; exit 1; }
cd "$REPO_ROOT"

NSYS_BIN="${NSYS_BIN:-nsys}"
command -v "$NSYS_BIN" >/dev/null || { echo "ERROR: nsys not found on PATH (source the CUDA env or set NSYS_BIN)" >&2; exit 1; }

MODEL="${MODEL:?set MODEL to ouro or huginn}"
DATASET="${DATASET:-sharegpt}"
WIDTHS="${WIDTHS:-64 512}" # Ablation width plus a near-uncapped wide regime
set_throughput_defaults

case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        LOOP_STEPS=4
        THRESHOLD="${THRESHOLD:-0.2}" # earliest threshold from exit sweep
        MIN_CODA_BATCHES="${MIN_CODA_BATCHES:-1}" # immediate coda (the default), as in the throughput sweeps
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        LOOP_STEPS=16
        THRESHOLD="${THRESHOLD:-0.28}" # earliest threshold from exit sweep
        MIN_CODA_BATCHES="${MIN_CODA_BATCHES:-quarter}" # 'quarter' resolves against each maximum decode batch size
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

WORKLOAD="$(replay_workload_for "$MODEL" "$DATASET" "$THRESHOLD")"

BACKENDS="${BACKENDS:-cb cdb-norefill cdb-refill}"
LIMIT="${LIMIT:-$(num_requests_for "$DATASET")}"
MIN_FREE_SLOTS="${MIN_FREE_SLOTS:--}"
export LOOPSGL_ENABLE_NVTX=1

WORKLOAD_STEM="$(basename "${WORKLOAD%.jsonl}")"
THRESH_TAG="q$(printf '%s' "$THRESHOLD" | tr '.' 'p')"
POLICY_TAG="_kv${CACHE_POLICY}_pressure${RETRACTION_MODE}"

echo "Experiment: nsys trace (replay benchmark), $MODEL_ID, $DATASET, q=$THRESHOLD"
echo "Backends:   $BACKENDS (min coda batches: $MIN_CODA_BATCHES)"
echo "KV policy: $CACHE_POLICY (retraction mode: $RETRACTION_MODE)"
echo "Max decode batch sizes: $WIDTHS (min free slots: $MIN_FREE_SLOTS)"
echo "Analysis:   full window (benchmark.generate)"
echo "nsys:       $("$NSYS_BIN" --version | head -1)"
echo ""

for width in $WIDTHS; do
    OUTPUT_DIR="outputs/nsys/${WORKLOAD_STEM}_${THRESH_TAG}${POLICY_TAG}_w${width}"
    mkdir -p "$OUTPUT_DIR"
    limit="$LIMIT"

    for backend in $BACKENDS; do
        # Only cdb-refill batches codas; the other backends run once per width.
        min_codas="-"
        [[ "$backend" == cdb-refill ]] && min_codas="$MIN_CODA_BATCHES"
        for min_coda in $min_codas; do
            if [[ "$min_coda" == quarter ]]; then
                min_coda=$((width / 4))
            fi
            if [[ "$min_coda" != "-" ]] && ((min_coda < 1)); then
                min_coda=1
            fi
            for min_free in $MIN_FREE_SLOTS; do
                run_tag="$backend"
                [[ "$min_coda" != "-" ]] && run_tag="${backend}_k${min_coda}"
                [[ "$min_free" != "-" ]] && run_tag="${run_tag}_s${min_free}"
                echo "=== width: $width  backend: $backend  min coda batch: $min_coda  min free slots: $min_free  limit: $limit ==="
                analysis_stem="${run_tag}_full"
                rep="$OUTPUT_DIR/${analysis_stem}.nsys-rep"
                sqlite="$OUTPUT_DIR/${analysis_stem}.sqlite"
                analysis_json="$OUTPUT_DIR/${analysis_stem}_analysis.json"
                analysis_csv="$OUTPUT_DIR/${analysis_stem}_analysis.csv"
                summary="$OUTPUT_DIR/${analysis_stem}_summary.jsonl"
                for artifact in "$rep" "$sqlite" "$analysis_json" "$analysis_csv" "$summary"; do
                    if [[ -e "$artifact" ]]; then
                        echo "ERROR: refusing to overwrite $artifact" >&2
                        exit 1
                    fi
                done

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
                    --dataset-name custom --dataset-path "$WORKLOAD"
                    --custom-input-format token-ids --no-shuffle --replay-exits
                    --loop-steps "$LOOP_STEPS"
                    --loop-cache-policy "$CACHE_POLICY"
                    --max-seq-len-override "$MAX_MODEL_LEN"
                    --max-running-requests "$width"
                    --dtype "$DTYPE"
                    --attention-backend "$ATTENTION_BACKEND"
                    "${NUM_PAGES_ARGS[@]}"
                    --memory-ratio "$MEMORY_RATIO"
                    --max-prefill-length "$MAX_PREFILL_LENGTH"
                    --page-size "$PAGE_SIZE"
                    "${RADIX_CACHE_ARGS[@]}"
                    --watermark "$WATERMARK"
                    --retraction-mode "$RETRACTION_MODE"
                    --num-prompts "$limit"
                    --cuda-graph-max-bs "$width"
                    --cuda-graph-backend-prefill breakable
                    # Benchmark rows measured under profiling overhead are stored with
                    # their traces, outside the throughput sweep results.
                    --result-filename "$summary"
                )
                [[ "$min_coda" != "-" ]] && args+=(--min-coda-batch-size "$min_coda")
                [[ "$min_free" != "-" ]] && args+=(--min-free-slots-delay "$min_free")

                "$NSYS_BIN" profile \
                    --trace=cuda,nvtx \
                    --sample=none \
                    --cpuctxsw=none \
                    --cuda-graph-trace=graph \
                    --cuda-event-trace=false \
                    --trace-fork-before-exec=true \
                    --output="$rep" \
                    uv run --frozen --project "$REPO_ROOT" python -m loopsgl.benchmark.offline_throughput "${args[@]}"

                echo "--- exporting $sqlite"
                "$NSYS_BIN" export --type=sqlite --output="$sqlite" "$rep"

                echo "--- analyzing full window"
                uv run --frozen --project "$REPO_ROOT" python -m loopsgl.benchmark.analyze_nsys \
                    "$sqlite" \
                    --output-json "$analysis_json" \
                    --output-csv "$analysis_csv"
                echo ""
            done
        done
    done
done

echo "Done. Traces and analyses in outputs/nsys/"
