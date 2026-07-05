#!/usr/bin/env bash

# Vast-friendly variant of runs/speedrun.sh.
# Runs entirely on the target host: restore durable artifacts from Drive, fetch
# source-downloadable data locally, train/evaluate, and continuously sync back.

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
NUM_GPUS="${NANOCHAT_NUM_GPUS:-}"
DEVICE_BATCH_SIZE="${NANOCHAT_DEVICE_BATCH_SIZE:-16}"
ENABLE_FP8="${NANOCHAT_ENABLE_FP8:-1}"
DEPTH="${NANOCHAT_DEPTH:-24}"
TARGET_PARAM_DATA_RATIO="${NANOCHAT_TARGET_PARAM_DATA_RATIO:-8}"
BASE_MODEL_TAG="${NANOCHAT_BASE_MODEL_TAG:-d$DEPTH}"
BASE_SAVE_EVERY="${NANOCHAT_BASE_SAVE_EVERY:-200}"
DATASET_SHARDS="${NANOCHAT_DATASET_SHARDS:-170}"
DATASET_BOOTSTRAP_SHARDS="${NANOCHAT_DATASET_BOOTSTRAP_SHARDS:-8}"
RUN_BASE_EVAL="${NANOCHAT_RUN_BASE_EVAL:-1}"
RUN_SFT="${NANOCHAT_RUN_SFT:-1}"
RUN_CHAT_EVAL="${NANOCHAT_RUN_CHAT_EVAL:-1}"
RUN_REPORT="${NANOCHAT_RUN_REPORT:-1}"
RESTORE_GDRIVE="${NANOCHAT_RESTORE_GDRIVE:-1}"

if [ -z "$NUM_GPUS" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    else
        NUM_GPUS="8"
    fi
fi

case "$NUM_GPUS" in
    ''|*[!0-9]*)
        echo "ERROR: NANOCHAT_NUM_GPUS must be a positive integer, got '$NUM_GPUS'." >&2
        exit 1
        ;;
esac
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: NANOCHAT_NUM_GPUS must be >= 1, got '$NUM_GPUS'." >&2
    exit 1
fi

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

dataset_present() {
    local data_dir="$NANOCHAT_BASE_DIR/base_data_climbmix"
    [ -d "$data_dir" ] && compgen -G "$data_dir/*.parquet" >/dev/null
}

tokenizer_present() {
    local tokenizer_dir="$NANOCHAT_BASE_DIR/tokenizer"
    [ -f "$tokenizer_dir/tokenizer.pkl" ] && [ -f "$tokenizer_dir/token_bytes.pt" ]
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

find_latest_checkpoint_step() {
    local checkpoint_dir="$1"
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
echo "NANOCHAT_NUM_GPUS: $NUM_GPUS"
echo "NANOCHAT_DEVICE_BATCH_SIZE: $DEVICE_BATCH_SIZE"
echo "NANOCHAT_ENABLE_FP8: $ENABLE_FP8"
echo "NANOCHAT_DEPTH: $DEPTH"
echo "NANOCHAT_BASE_MODEL_TAG: $BASE_MODEL_TAG"
echo "NANOCHAT_DATASET_SHARDS: $DATASET_SHARDS"
echo "NANOCHAT_RESTORE_GDRIVE: $RESTORE_GDRIVE"
echo "Log file: $LOG_FILE"

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
# Restore generated artifacts from Google Drive before deciding what to skip.

if [ "$RESTORE_GDRIVE" = "1" ]; then
    if [ -n "${NANOCHAT_RCLONE_REMOTE:-}" ] && command -v rclone >/dev/null 2>&1; then
        bash runs/restore_gdrive.sh
    else
        echo "Skipping Google Drive restore because rclone or NANOCHAT_RCLONE_REMOTE is unavailable"
    fi
fi

# -----------------------------------------------------------------------------
# Tokenizer and pretraining data

if [ "${NANOCHAT_SKIP_DATASET:-0}" = "1" ]; then
    echo "Skipping dataset download because NANOCHAT_SKIP_DATASET=1"
elif dataset_present; then
    echo "Using existing dataset at $NANOCHAT_BASE_DIR/base_data_climbmix"
else
    if [ "$DATASET_BOOTSTRAP_SHARDS" -gt 0 ]; then
        python -m nanochat.dataset -n "$DATASET_BOOTSTRAP_SHARDS"
    fi
    python -m nanochat.dataset -n "$DATASET_SHARDS" &
    DATASET_DOWNLOAD_PID=$!
fi

if [ "${NANOCHAT_SKIP_TOKENIZER:-0}" = "1" ]; then
    echo "Skipping tokenizer train/eval because NANOCHAT_SKIP_TOKENIZER=1"
elif tokenizer_present; then
    echo "Using existing tokenizer at $NANOCHAT_BASE_DIR/tokenizer"
else
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi

if [ -n "${DATASET_DOWNLOAD_PID:-}" ]; then
    echo "Waiting for dataset download to complete..."
    wait "$DATASET_DOWNLOAD_PID"
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)

BASE_RESUME_STEP=""
if [ "${NANOCHAT_RESUME_BASE_FROM_CHECKPOINT:-1}" = "1" ]; then
    BASE_RESUME_STEP="$(find_latest_base_step)"
fi

BASE_TRAIN_ARGS=(--depth="$DEPTH" --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" --device-batch-size="$DEVICE_BATCH_SIZE" --run="$WANDB_RUN" --model-tag="$BASE_MODEL_TAG" --save-every="$BASE_SAVE_EVERY")
if [ "$ENABLE_FP8" = "1" ]; then
    BASE_TRAIN_ARGS+=(--fp8)
fi
if [ -n "${NANOCHAT_MAX_SEQ_LEN:-}" ]; then
    BASE_TRAIN_ARGS+=(--max-seq-len="$NANOCHAT_MAX_SEQ_LEN")
fi
if [ -n "${NANOCHAT_TOTAL_BATCH_SIZE:-}" ]; then
    BASE_TRAIN_ARGS+=(--total-batch-size="$NANOCHAT_TOTAL_BATCH_SIZE")
fi
if [ -n "${NANOCHAT_NUM_ITERATIONS:-}" ]; then
    BASE_TRAIN_ARGS+=(--num-iterations="$NANOCHAT_NUM_ITERATIONS")
fi
if [ -n "${NANOCHAT_EVAL_TOKENS:-}" ]; then
    BASE_TRAIN_ARGS+=(--eval-tokens="$NANOCHAT_EVAL_TOKENS")
fi
if [ -n "${NANOCHAT_CORE_METRIC_EVERY:-}" ]; then
    BASE_TRAIN_ARGS+=(--core-metric-every="$NANOCHAT_CORE_METRIC_EVERY")
fi
if [ -n "$BASE_RESUME_STEP" ]; then
    echo "Resuming base training from checkpoint step $BASE_RESUME_STEP"
    BASE_TRAIN_ARGS+=(--resume-from-step="$BASE_RESUME_STEP")
else
    echo "No base checkpoint found; starting base training from scratch"
fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_train -- "${BASE_TRAIN_ARGS[@]}"
if [ "$RUN_BASE_EVAL" = "1" ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_eval -- --device-batch-size="$DEVICE_BATCH_SIZE" --model-tag="$BASE_MODEL_TAG"
else
    echo "Skipping base eval because NANOCHAT_RUN_BASE_EVAL=$RUN_BASE_EVAL"
fi

# -----------------------------------------------------------------------------
# SFT

SFT_STEP="$(find_latest_checkpoint_step "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$BASE_MODEL_TAG")"
if [ "$RUN_SFT" = "1" ] && [ -n "$SFT_STEP" ] && [ "${NANOCHAT_RERUN_SFT_IF_PRESENT:-0}" != "1" ]; then
    echo "Using existing SFT checkpoint step $SFT_STEP for model tag $BASE_MODEL_TAG"
elif [ "$RUN_SFT" = "1" ]; then
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

    torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.chat_sft -- --device-batch-size="$DEVICE_BATCH_SIZE" --run="$WANDB_RUN" --model-tag="$BASE_MODEL_TAG"
else
    echo "Skipping SFT because NANOCHAT_RUN_SFT=$RUN_SFT"
fi

SFT_STEP="$(find_latest_checkpoint_step "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$BASE_MODEL_TAG")"
if [ "$RUN_CHAT_EVAL" = "1" ] && [ -n "$SFT_STEP" ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.chat_eval -- -i sft
else
    echo "Skipping chat eval because NANOCHAT_RUN_CHAT_EVAL=$RUN_CHAT_EVAL or no SFT checkpoint exists"
fi

# -----------------------------------------------------------------------------
# Generate the full report

if [ "$RUN_REPORT" = "1" ]; then
    python -m nanochat.report generate
else
    echo "Skipping report generation because NANOCHAT_RUN_REPORT=$RUN_REPORT"
fi

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
