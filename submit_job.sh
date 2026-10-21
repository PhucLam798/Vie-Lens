#!/bin/bash
# =============================================================================
#  UIT HPC — Slurm job template with switchable GPU profile
#
#  Usage:
#    bash submit_job.sh <script.py> [small|normal|large|full] [args for your script...]
#
#  Examples:
#    bash submit_job.sh train.py                          # normal (default)
#    bash submit_job.sh scripts/eval.py small             # small
#    bash submit_job.sh train.py large --epochs 10 --lr 3e-4
#    bash submit_job.sh train.py full
#    CPUS=8 MEM=48G TIME=48:00:00 bash submit_job.sh train.py normal
#    JOB_NAME=exp1 bash submit_job.sh train.py            # custom job name
#
#  `sbatch submit_job.sh train.py [args]` also works, but always uses the
#  normal defaults below and needs logs/ to exist already.
#
#  Profiles (per-job limits enforced by the QOS, checked with sacctmgr):
#  ┌───────────┬───────────────┬──────────┬─────────┬─────────┬──────────────────┐
#  │ profile   │ GRES          │ QOS      │ max CPU │ max RAM │ default CPU/RAM  │
#  ├───────────┼───────────────┼──────────┼─────────┼─────────┼──────────────────┤
#  │ small     │ 1g.18gb       │ mig1-18  │ 4       │ 56G     │ 4  / 16G         │
#  │ normal    │ 2g.35gb       │ mig2-35  │ 8       │ 64G     │ 8  / 32G         │
#  │ large     │ 3g.71gb       │ mig3-71  │ 16      │ 128G    │ 12 / 64G         │
#  │ full      │ 7g.141gb      │ mig7-141 │ 32      │ 256G    │ 12 / 64G         │
#  └───────────┴───────────────┴──────────┴─────────┴─────────┴──────────────────┘
#  Max walltime 72:00:00. Max running jobs per user: 3 (5 for normal).
#  Max 50 submitted jobs per user.
# =============================================================================

# Complete defaults (normal), used as-is by plain `sbatch submit_job.sh`.
# When run via `bash submit_job.sh`, the flags passed to sbatch override these.
#SBATCH --job-name=normal
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# ---------------------------- User configuration -----------------------------
USER=${USER:-$(whoami)}
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
VENV="/datastore/luannt/.cache/sb"
DEFAULT_TIME=24:00:00
# -----------------------------------------------------------------------------

usage() { sed -n '5,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# =============================== LAUNCHER ====================================
# Not inside Slurm yet: check the script, pick the profile, submit this file.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]] && usage 1

    [[ -f "$1" ]] || { echo "ERROR: script not found: $1" >&2; exit 1; }
    SCRIPT=$(realpath "$1"); shift

    PROFILE=normal
    if [[ $# -gt 0 && "$1" =~ ^(small|normal|large|full)$ ]]; then
        PROFILE=$1; shift
    fi

    # GRES type, QOS and limits must match: each QOS allows only its own GPU type.
    case "$PROFILE" in
        small)  GRES=gpu:1g.18gb:1;  QOS=mig1-18;  MAX_CPU=4;  MAX_MEM=56;  DEF_CPU=4;  DEF_MEM=16G ;;
        normal) GRES=gpu:2g.35gb:1;  QOS=mig2-35;  MAX_CPU=8;  MAX_MEM=64;  DEF_CPU=8;  DEF_MEM=32G ;;
        large)  GRES=gpu:3g.71gb:1;  QOS=mig3-71;  MAX_CPU=16; MAX_MEM=128; DEF_CPU=12; DEF_MEM=64G ;;
        full)   GRES=gpu:7g.141gb:1; QOS=mig7-141; MAX_CPU=32; MAX_MEM=256; DEF_CPU=12; DEF_MEM=64G ;;
    esac

    CPUS=${CPUS:-$DEF_CPU}
    MEM=${MEM:-$DEF_MEM}
    TIME=${TIME:-$DEFAULT_TIME}
    JOB_NAME=${JOB_NAME:-$(basename "$SCRIPT" .py)}

    [[ "$CPUS" =~ ^[0-9]+$ ]] || { echo "ERROR: CPUS must be a whole number (got '$CPUS')." >&2; exit 1; }
    if (( CPUS > MAX_CPU )); then
        echo "ERROR: $PROFILE allows at most $MAX_CPU CPUs (asked for $CPUS)." >&2
        exit 1
    fi

    [[ "$MEM" =~ ^([0-9]+)[Gg]$ ]] || { echo "ERROR: give MEM in whole GB, e.g. 32G (got '$MEM')." >&2; exit 1; }
    if (( BASH_REMATCH[1] > MAX_MEM )); then
        echo "ERROR: $PROFILE allows at most ${MAX_MEM}G RAM (asked for $MEM)." >&2
        exit 1
    fi

    mkdir -p logs   # must exist before sbatch, or Slurm has nowhere to write

    echo "Submitting $SCRIPT"
    echo "  -> $PROFILE | $QOS | $GRES | ${CPUS} CPU | $MEM | $TIME | name: $JOB_NAME"
    sbatch \
        --job-name="$JOB_NAME" \
        --partition="$PROFILE" \
        --qos="$QOS" \
        --gres="$GRES" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --output="logs/%j_%x_${PROFILE}.out" \
        --error="logs/%j_%x_${PROFILE}.err" \
        "$0" "$SCRIPT" "$@"
    exit $?
fi

# ================================== JOB ======================================
# Runs on the GPU node once Slurm starts the job.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "ERROR: no python file given" >&2; exit 1; }
SCRIPT=$(realpath "$1"); shift

# Keep caches out of $HOME so it doesn't hit the 10 GB quota
export HF_HOME=/datastore/$USER/.cache/huggingface
export TORCH_HOME=/datastore/$USER/.cache/torch
export XDG_CACHE_HOME=/datastore/luannt/.cache
export TRITON_CACHE_DIR=/datastore/luannt/.cache/triton
export PIP_CACHE_DIR=/datastore/$USER/.cache/pip
export UV_CACHE_DIR=/datastore/$USER/.cache/uv
# Model-only Transformers cache. Keep HF_HOME/HF_DATASETS_CACHE and other
# runtime caches on their existing /datastore locations.
export HF_MODEL_CACHE=/datastore/shared/models
export HF_HUB_CACHE="$HF_MODEL_CACHE"
export TRANSFORMERS_CACHE="$HF_MODEL_CACHE"
export HF_DATASETS_CACHE=/datastore/luannt/.cache/huggingface/datasets
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), partition $SLURM_JOB_PARTITION"
echo "Script : $SCRIPT $*"
echo "Start  : $(date)"
nvidia-smi

[[ -f "$PROJECT_DIR/requirements.txt" ]] || { echo "ERROR: requirements.txt not found in $PROJECT_DIR" >&2; exit 1; }

VENV=/datastore/luannt/.cache/sb
[[ -x "$VENV/bin/python" ]] || { echo "ERROR: canonical venv missing at $VENV. Create it on the login node: uv venv --python 3.12 $VENV && uv pip install --python $VENV/bin/python -r $PROJECT_DIR/requirements.txt" >&2; exit 1; }
source "$VENV/bin/activate"
python - <<'PY' || exit 1
import sys
assert sys.prefix == "/datastore/luannt/.cache/sb", f"wrong interpreter: {sys.prefix}"
import torch, transformers
print(f"venv OK: python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}")
PY

cd "$(dirname "$SCRIPT")"
HF_TOKEN_FILE="$PROJECT_DIR/keys/hf.txt"
[[ -r "$HF_TOKEN_FILE" ]] || { echo "ERROR: Hugging Face token file is not readable: $HF_TOKEN_FILE" >&2; exit 1; }
# Disable tracing while reading the secret so it cannot appear in the job log.
case $- in *x*) HF_RESTORE_XTRACE=1; set +x ;; *) HF_RESTORE_XTRACE=0 ;; esac
HF_TOKEN=$(tr -d '\r\n' < "$HF_TOKEN_FILE")
if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: Hugging Face token file is empty" >&2
    exit 1
fi
export HF_TOKEN
if [[ "$HF_RESTORE_XTRACE" == 1 ]]; then set -x; fi
python "$SCRIPT" "$@"

echo "End    : $(date)"
