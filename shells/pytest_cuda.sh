#!/usr/bin/env bash
# Run the CUDA-marked pytest suite on a GPU node.
# CPU tests run anywhere with: uv run pytest -m "not cuda"
#
# Usage:
#   ./shells/_submit.sh shells/pytest_cuda.sh
#   PYTEST_ARGS="tests/core -k cdb" ./shells/_submit.sh shells/pytest_cuda.sh

set -euo pipefail

cd "${REPO_ROOT:?submit with ./shells/_submit.sh}"

nvidia-smi --query-gpu=name --format=csv,noheader || true
# PYTEST_ARGS is split into separate pytest arguments.
# shellcheck disable=SC2086
uv run --frozen --extra dev pytest -m cuda -q ${PYTEST_ARGS:-}
