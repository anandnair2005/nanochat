#!/usr/bin/env bash

# Vast-friendly variant of runs/speedrun.sh.
# Preserves the default H100 training commands while allowing prewarmed storage
# and baked Docker dependencies to skip setup work.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
    if [ "${NANOCHAT_ALLOW_DEFAULT_BASE_DIR:-0}" = "1" ]; then
        export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
    else
        echo "ERROR: NANOCHAT_BASE_DIR must be set. Set NANOCHAT_ALLOW_DEFAULT_BASE_DIR=1 to use $HOME/.cache/nanochat." >&2
        exit 1
    fi
fi

export NANOCHAT_BASE_DIR
export NANOCHAT_RUN_ID="${NANOCHAT_RUN_ID:-speedrun-$(date -u +%Y%m%dT%H%M%SZ)}"
export WANDB_RUN="${WANDB_RUN:-dummy}"

LOG_DIR="$NANOCHAT_BASE_DIR/logs"
STATUS_DIR="$NANOCHAT_BASE_DIR/status"
LOG_FILE="${NANOCHAT_LOG_FILE:-$LOG_DIR/speedrun.log}"
SYNC_PID=""

mkdir -p "$NANOCHAT_BASE_DIR" "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR/speedrun.DONE" "$STATUS_DIR/speedrun.FAILED"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/speedrun.STARTED"

on_error() {
    local exit_code=$?
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/speedrun.FAILED"
    stop_background_sync || true
    final_gdrive_sync || true
    echo "Speedrun failed with exit code $exit_code. See $LOG_FILE" >&2
    exit "$exit_code"
}

require_prewarmed_dataset() {
    local data_dir="$NANOCHAT_BASE_DIR/base_data_climbmix"
    if [ ! -d "$data_dir" ]; then
        echo "ERROR: NANOCHAT_SKIP_DATASET=1 but $data_dir does not exist." >&2
        return 1
    fi
    if ! compgen -G "$data_dir/*.parquet" >/dev/null; then
        echo "ERROR: NANOCHAT_SKIP_DATASET=1 but no parquet files exist in $data_dir." >&2
        return 1
    fi
}

require_prewarmed_tokenizer() {
    local tokenizer_dir="$NANOCHAT_BASE_DIR/tokenizer"
    if [ ! -f "$tokenizer_dir/tokenizer.pkl" ]; then
        echo "ERROR: NANOCHAT_SKIP_TOKENIZER=1 but $tokenizer_dir/tokenizer.pkl does not exist." >&2
        return 1
    fi
    if [ ! -f "$tokenizer_dir/token_bytes.pt" ]; then
        echo "ERROR: NANOCHAT_SKIP_TOKENIZER=1 but $tokenizer_dir/token_bytes.pt does not exist." >&2
        return 1
    fi
}

find_latest_base_step() {
    local checkpoint_dir="$NANOCHAT_BASE_DIR/base_checkpoints/${NANOCHAT_BASE_MODEL_TAG:-d24}"
    local latest=""
    local file step
    if [ ! -d "$checkpoint_dir" ]; then
        return 0
    fi
    for file in "$checkpoint_dir"/model_*.pt; do
        [ -e "$file" ] || continue
        step="$(basename "$file")"
        step="${step#model_}"
        step="${step%.pt}"
        if [ -z "$latest" ] || [ "10#$step" -gt "10#$latest" ]; then
            latest="$step"
        fi
    done
    if [ -n "$latest" ]; then
        echo "$((10#$latest))"
    fi
}

start_background_sync() {
    if [ "${NANOCHAT_ENABLE_GDRIVE_SYNC:-0}" != "1" ]; then
        return 0
    fi
    if [ -z "${NANOCHAT_RCLONE_REMOTE:-}" ]; then
        echo "ERROR: NANOCHAT_ENABLE_GDRIVE_SYNC=1 but NANOCHAT_RCLONE_REMOTE is unset." >&2
        return 1
    fi
    rm -f "$STATUS_DIR/gdrive_sync.STOP"
    echo "Starting background Google Drive sync loop"
    NANOCHAT_SYNC_MODE=loop bash runs/sync_gdrive.sh &
    SYNC_PID=$!
}

stop_background_sync() {
    if [ -n "$SYNC_PID" ]; then
        echo "Stopping background Google Drive sync loop"
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/gdrive_sync.STOP"
        wait "$SYNC_PID" || true
        SYNC_PID=""
    fi
}

final_gdrive_sync() {
    if [ "${NANOCHAT_ENABLE_GDRIVE_SYNC:-0}" = "1" ] && [ -n "${NANOCHAT_RCLONE_REMOTE:-}" ]; then
        echo "Running final Google Drive sync"
        NANOCHAT_SYNC_MODE=once bash runs/sync_gdrive.sh
    fi
}

trap on_error ERR

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting nanochat Vast speedrun"
echo "Repo root: $REPO_ROOT"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "NANOCHAT_RUN_ID: $NANOCHAT_RUN_ID"
echo "WANDB_RUN: $WANDB_RUN"
echo "Log file: $LOG_FILE"

# Validate prewarmed artifacts before any Python work when setup phases are skipped.
if [ "${NANOCHAT_SKIP_DATASET:-0}" = "1" ]; then
    require_prewarmed_dataset
fi
if [ "${NANOCHAT_SKIP_TOKENIZER:-0}" = "1" ]; then
    require_prewarmed_tokenizer
fi

start_background_sync

# -----------------------------------------------------------------------------
# Python venv setup with uv

if [ "${NANOCHAT_SKIP_UV_SYNC:-0}" = "1" ]; then
    echo "Skipping uv dependency setup because NANOCHAT_SKIP_UV_SYNC=1"
    if [ -f ".venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
else
    if ! command -v uv >/dev/null 2>&1; then
        echo "Installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# -----------------------------------------------------------------------------
# Report reset

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer and pretraining data

if [ "${NANOCHAT_SKIP_DATASET:-0}" = "1" ]; then
    echo "Skipping dataset download because NANOCHAT_SKIP_DATASET=1"
else
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 170 &
    DATASET_DOWNLOAD_PID=$!
fi

if [ "${NANOCHAT_SKIP_TOKENIZER:-0}" = "1" ]; then
    echo "Skipping tokenizer train/eval because NANOCHAT_SKIP_TOKENIZER=1"
else
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi

if [ "${NANOCHAT_SKIP_DATASET:-0}" != "1" ]; then
    echo "Waiting for dataset download to complete..."
    wait "$DATASET_DOWNLOAD_PID"
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)

BASE_MODEL_TAG="${NANOCHAT_BASE_MODEL_TAG:-d24}"
BASE_SAVE_EVERY="${NANOCHAT_BASE_SAVE_EVERY:-200}"
BASE_RESUME_STEP=""
if [ "${NANOCHAT_RESUME_BASE_FROM_CHECKPOINT:-1}" = "1" ]; then
    BASE_RESUME_STEP="$(find_latest_base_step)"
fi

BASE_TRAIN_ARGS=(--depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run="$WANDB_RUN" --model-tag="$BASE_MODEL_TAG" --save-every="$BASE_SAVE_EVERY")
if [ -n "$BASE_RESUME_STEP" ]; then
    echo "Resuming base training from checkpoint step $BASE_RESUME_STEP"
    BASE_TRAIN_ARGS+=(--resume-from-step="$BASE_RESUME_STEP")
else
    echo "No base checkpoint found; starting base training from scratch"
fi

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- "${BASE_TRAIN_ARGS[@]}"
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16 --model-tag="$BASE_MODEL_TAG"

# -----------------------------------------------------------------------------
# SFT

IDENTITY_PATH="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ "${NANOCHAT_SKIP_IDENTITY_DOWNLOAD:-0}" = "1" ]; then
    if [ ! -f "$IDENTITY_PATH" ]; then
        echo "ERROR: NANOCHAT_SKIP_IDENTITY_DOWNLOAD=1 but $IDENTITY_PATH does not exist." >&2
        false
    fi
elif [ -f "$IDENTITY_PATH" ]; then
    echo "Using existing $IDENTITY_PATH"
else
    curl -L -o "$IDENTITY_PATH" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run="$WANDB_RUN" --model-tag="$BASE_MODEL_TAG"
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate the full report

python -m nanochat.report generate

stop_background_sync
final_gdrive_sync

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/speedrun.DONE"
trap - ERR

echo
echo "Speedrun complete"
echo "Base directory: $NANOCHAT_BASE_DIR"
echo "Run ID: $NANOCHAT_RUN_ID"
echo "Log file: $LOG_FILE"
echo "Status directory: $STATUS_DIR"
