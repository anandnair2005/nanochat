#!/usr/bin/env bash

# Prepare a mounted Vast storage directory before renting the 8xH100 node.
# This downloads disposable ClimbMix data and restores durable generated
# artifacts from Google Drive. It intentionally does not train the tokenizer or
# restore source-downloadable eval/identity data.

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
LOG_DIR="$NANOCHAT_BASE_DIR/logs"
STATUS_DIR="$NANOCHAT_BASE_DIR/status"
LOG_FILE="$LOG_DIR/prewarm.log"
NUM_SHARDS="${NANOCHAT_NUM_SHARDS:-170}"

mkdir -p "$NANOCHAT_BASE_DIR" "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR/prewarm.DONE" "$STATUS_DIR/prewarm.FAILED"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/prewarm.STARTED"

on_error() {
    local exit_code=$?
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/prewarm.FAILED"
    echo "Prewarm failed with exit code $exit_code. See $LOG_FILE" >&2
    exit "$exit_code"
}
trap on_error ERR

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting nanochat Vast prewarm"
echo "Repo root: $REPO_ROOT"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "NANOCHAT_NUM_SHARDS: $NUM_SHARDS"
echo "Log file: $LOG_FILE"

if [ "${NANOCHAT_SKIP_UV_SYNC:-0}" = "1" ]; then
    echo "Skipping uv dependency setup because NANOCHAT_SKIP_UV_SYNC=1"
else
    if ! command -v uv >/dev/null 2>&1; then
        echo "Installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    [ -d ".venv" ] || uv venv
    uv sync --extra "${NANOCHAT_UV_EXTRA:-gpu}"
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

if [ "${NANOCHAT_SKIP_GDRIVE_RESTORE:-0}" = "1" ]; then
    echo "Skipping Google Drive artifact restore because NANOCHAT_SKIP_GDRIVE_RESTORE=1"
elif [ -n "${NANOCHAT_RCLONE_REMOTE:-}" ]; then
    bash runs/restore_gdrive.sh
else
    echo "Skipping Google Drive artifact restore because NANOCHAT_RCLONE_REMOTE is unset"
fi

if [ "$NUM_SHARDS" = "0" ]; then
    echo "Skipping dataset download because NANOCHAT_NUM_SHARDS=0"
else
    python -m nanochat.dataset -n "$NUM_SHARDS"
fi

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/prewarm.DONE"
trap - ERR

echo
echo "Prewarm complete"
echo "Base directory: $NANOCHAT_BASE_DIR"
echo "Dataset directory: $NANOCHAT_BASE_DIR/base_data_climbmix"
echo "Tokenizer directory: $NANOCHAT_BASE_DIR/tokenizer"
echo "Log file: $LOG_FILE"
echo "Status directory: $STATUS_DIR"
