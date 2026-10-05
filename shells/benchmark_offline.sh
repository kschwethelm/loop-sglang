#!/usr/bin/env bash
# Usage: ./shells/_submit.sh shells/benchmark_offline.sh <bench.py|bench_wildchat.py> [args...]
# Select the loop cache with --loop-cache-policy depth_indexed (default) or shared.
# Select a model ID or local path with --model (default: KristianS7/Ouro-1.4B).

set -euo pipefail

BENCHMARK="${1:?Pass a benchmark path relative to benchmark/offline/.}"
shift
: "${SLURM_JOB_ID:?Submit this script through shells/_submit.sh.}"
: "${REPO_ROOT:?}"
EXTRAS=()
BENCHMARK_ARGS=()
case "$BENCHMARK" in
    bench.py) ;;
    bench_wildchat.py)
        EXTRAS=(--extra dev)
        DATA_DIR="${WILDCHAT_DATA_DIR:-${DATA_ROOT:?}/wildchat}"
        mkdir -p -- "$(dirname -- "$DATA_DIR")"
        BENCHMARK_ARGS=(--data-dir "$DATA_DIR")
        ;;
    *) echo "ERROR: Unknown benchmark: $BENCHMARK" >&2; exit 1 ;;
esac

exec uv run --frozen "${EXTRAS[@]}" --project "$REPO_ROOT" \
    python -u "$REPO_ROOT/benchmark/offline/$BENCHMARK" "${BENCHMARK_ARGS[@]}" "$@"
