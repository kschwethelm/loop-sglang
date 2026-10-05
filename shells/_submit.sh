#!/usr/bin/env bash
# Submit a Bash GPU workload with an ignored, machine-specific configuration.
# GPU pinning is validated and applied inside the allocated batch job.
# Usage: ./shells/_submit.sh <script.sh> [script_args...] [-- sbatch_options...]

set -euo pipefail

SCRIPT_PATH="${1:?Usage: $0 <script.sh> [script_args...] [-- sbatch_options...]}"
shift
SCRIPT_PATH="$(realpath -e -- "$SCRIPT_PATH")"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "ERROR: Workload script is not a file: $SCRIPT_PATH" >&2
    exit 1
fi

SHELLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(realpath -e -- "${REPO_ROOT:-$SHELLS_DIR/..}")"
export REPO_ROOT
CONFIG_PATH="${MACHINE_CONFIG:-$SHELLS_DIR/_machine_config.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Machine config not found: $CONFIG_PATH" >&2
    echo "Copy $SHELLS_DIR/_machine_config.sh.template to _machine_config.sh and fill in this machine's settings." >&2
    exit 1
fi
MACHINE_CONFIG="$(realpath -e -- "$CONFIG_PATH")"
export MACHINE_CONFIG
# shellcheck source=/dev/null
source "$MACHINE_CONFIG"

for variable in REPO_ROOT HF_HOME UV_CACHE_DIR NUM_GPUS; do
    if [[ -z "${!variable:-}" ]]; then
        echo "ERROR: $variable must be set in $MACHINE_CONFIG or the environment." >&2
        exit 1
    fi
done
for variable in NUM_GPUS SLURM_CPUS_GPU; do
    if [[ "$variable" == SLURM_CPUS_GPU && -z "${!variable:-}" ]]; then
        continue
    fi
    if [[ ! "${!variable}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $variable must be a positive integer." >&2
        exit 1
    fi
done
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
if [[ ! -d "$REPO_ROOT" ]]; then
    echo "ERROR: Repository directory does not exist: $REPO_ROOT" >&2
    exit 1
fi
export REPO_ROOT

if [[ -n "${CUDA_ENV_SCRIPT:-}" ]]; then
    if [[ ! -f "$CUDA_ENV_SCRIPT" ]]; then
        echo "ERROR: CUDA environment script does not exist: $CUDA_ENV_SCRIPT" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$CUDA_ENV_SCRIPT"
fi
export PATH="$PATH:$HOME/.local/bin"

SCRIPT_ARGS=()
SBATCH_EXTRA_ARGS=()
FOUND_SEPARATOR=false
for argument in "$@"; do
    if [[ "$FOUND_SEPARATOR" == false && "$argument" == -- ]]; then
        FOUND_SEPARATOR=true
    elif [[ "$FOUND_SEPARATOR" == true ]]; then
        SBATCH_EXTRA_ARGS+=("$argument")
    else
        SCRIPT_ARGS+=("$argument")
    fi
done

SCRIPT_NAME="$(basename -- "$SCRIPT_PATH" .sh)"
LOG_DIR="$REPO_ROOT/logs/$SCRIPT_NAME"
mkdir -p -- "$LOG_DIR"
LOG_PATTERN="%j"
for argument in "${SBATCH_EXTRA_ARGS[@]}"; do
    case "$argument" in
        --array|--array=*|-a|-a?*) LOG_PATTERN="%A_%a" ;;
    esac
done

GRES="gpu:${SLURM_GPU_TYPE:+$SLURM_GPU_TYPE:}$NUM_GPUS"
if [[ -n "${SLURM_GPU_MEM:-}" ]]; then
    GRES+=",gpumem:$SLURM_GPU_MEM"
fi

SBATCH_CMD=(sbatch
    "--job-name=loop-sglang_$SCRIPT_NAME"
    "--output=$LOG_DIR/$LOG_PATTERN.out"
    "--error=$LOG_DIR/$LOG_PATTERN.err"
    "--chdir=$REPO_ROOT"
    --export=ALL
    --nodes=1 --ntasks=1
    "--gres=$GRES"
    "--time=${SLURM_TIME:-08:00:00}"
)
[[ -z "${SLURM_CPUS_GPU:-}" ]] || SBATCH_CMD+=("--cpus-per-task=$SLURM_CPUS_GPU")
[[ -z "${SLURM_PARTITION_GPU:-}" ]] || SBATCH_CMD+=("--partition=$SLURM_PARTITION_GPU")
[[ -z "${SLURM_QOS_GPU:-}" ]] || SBATCH_CMD+=("--qos=$SLURM_QOS_GPU")
[[ -z "${SLURM_ACCOUNT:-}" ]] || SBATCH_CMD+=("--account=$SLURM_ACCOUNT")
[[ -z "${SLURM_MEM_GPU:-}" ]] || SBATCH_CMD+=("--mem=$SLURM_MEM_GPU")
[[ -z "${SLURM_MAIL_USER:-}" ]] || SBATCH_CMD+=(--mail-type=FAIL,END "--mail-user=$SLURM_MAIL_USER")

SBATCH_CMD+=("${SBATCH_EXTRA_ARGS[@]}")
printf 'Workload: %s\n' "$SCRIPT_PATH"
printf 'Submitting:'
printf ' %q' "${SBATCH_CMD[@]}"
printf '\n'

# Preserve the workload's leading comments, including active #SBATCH directives.
# SLURM spools the generated bootstrap, which executes the original workload by absolute path.
{
    printf '#!/usr/bin/env bash\n'
    HEADER_PATTERN='^[[:space:]]*(#.*)?$'
    while IFS= read -r HEADER_LINE || [[ -n "$HEADER_LINE" ]]; do
        [[ "$HEADER_LINE" =~ $HEADER_PATTERN ]] || break
        printf '%s\n' "$HEADER_LINE"
    done < "$SCRIPT_PATH"

    cat <<'BATCH'
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: GPU workloads must run inside a SLURM allocation." >&2
    exit 1
fi
if [[ -n "${CUDA_ENV_SCRIPT:-}" ]]; then
    # shellcheck source=/dev/null
    source "$CUDA_ENV_SCRIPT"
fi
export PATH="$PATH:$HOME/.local/bin"

if [[ -n "${PIN_H100_UUID:-}" ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "ERROR: GPU pinning requires SLURM to set CUDA_VISIBLE_DEVICES." >&2
        exit 1
    fi
    # Query CUDA-visible UUIDs to respect SLURM's device allocation and renumbering.
    # This process releases its CUDA contexts before the workload starts.
    uv run --frozen --project "${REPO_ROOT:?}" python - <<'PYTHON'
import os
import torch

visible = {
    "GPU-" + str(torch.cuda.get_device_properties(index).uuid).removeprefix("GPU-")
    for index in range(torch.cuda.device_count())
}
if os.environ["PIN_H100_UUID"] not in visible:
    raise SystemExit("ERROR: The pinned GPU is not in this job's visible allocation.")
PYTHON
    export CUDA_VISIBLE_DEVICES="$PIN_H100_UUID"
fi
BATCH

    printf 'cd -- %q\n' "$REPO_ROOT"
    printf 'exec bash -- %q' "$SCRIPT_PATH"
    if (( ${#SCRIPT_ARGS[@]} )); then
        printf ' %q' "${SCRIPT_ARGS[@]}"
    fi
    printf '\n'
} | "${SBATCH_CMD[@]}"
