#!/usr/bin/env bash
# Usage: ./shells/_submit.sh shells/benchmark_online.sh <bench_simple.py|bench_qwen.py> [args...]
# Select the loop cache with --loop-cache-policy depth_indexed (default) or shared.
# Start Ouro-1.4B on localhost:1919, run the benchmark, and stop the server.

set -euo pipefail

BENCHMARK="${1:?Pass a benchmark path relative to benchmark/online/.}"
shift
: "${SLURM_JOB_ID:?Submit this script through shells/_submit.sh.}"
: "${REPO_ROOT:?}"
LOOP_CACHE_POLICY=depth_indexed
CLIENT_ARGS=()
while (( $# )); do
    case "$1" in
        --loop-cache-policy)
            LOOP_CACHE_POLICY="${2:?Pass depth_indexed or shared after --loop-cache-policy.}"
            shift 2
            ;;
        --loop-cache-policy=*)
            LOOP_CACHE_POLICY="${1#*=}"
            shift
            ;;
        *) CLIENT_ARGS+=("$1"); shift ;;
    esac
done
BENCHMARK_ARGS=()
case "$BENCHMARK" in
    bench_simple.py) ;;
    bench_qwen.py)
        DATA_DIR="${QWEN_DATA_DIR:-${DATA_ROOT:?}/qwen}"
        mkdir -p -- "$(dirname -- "$DATA_DIR")"
        BENCHMARK_ARGS=(--data-dir "$DATA_DIR")
        ;;
    *) echo "ERROR: Unknown online benchmark: $BENCHMARK" >&2; exit 1 ;;
esac

URL="http://127.0.0.1:1919/v1/models"
if curl --silent --output /dev/null --max-time 2 "$URL"; then
    echo "ERROR: A server is already listening on localhost:1919." >&2
    exit 1
fi

# A separate process group lets cleanup stop the server and all its workers.
setsid uv run --frozen --project "$REPO_ROOT" python -u -m loopsgl \
    --model KristianS7/Ouro-1.4B --loop-cache-policy "$LOOP_CACHE_POLICY" \
    --host 127.0.0.1 --port 1919 &
SERVER_PID=$!
cleanup() {
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for ((attempt = 0; attempt < 10; attempt++)); do
        kill -0 -- "-$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Waiting for Loop-SGLang on localhost:1919..."
DEADLINE=$((SECONDS + 1800))
while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Loop-SGLang exited before becoming ready." >&2
        wait "$SERVER_PID"
        exit 1
    fi
    if curl --fail --silent --output /dev/null --max-time 2 "$URL"; then
        break
    fi
    if ((SECONDS >= DEADLINE)); then
        echo "ERROR: Loop-SGLang did not become ready within 30 minutes." >&2
        exit 1
    fi
    sleep 1
done

echo "Loop-SGLang is ready. Running $BENCHMARK."
uv run --frozen --project "$REPO_ROOT" python -u "$REPO_ROOT/benchmark/online/$BENCHMARK" "${BENCHMARK_ARGS[@]}" "${CLIENT_ARGS[@]}"
