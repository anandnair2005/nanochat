#!/usr/bin/env bash

# Restore durable generated nanochat artifacts from a restricted rclone root into
# NANOCHAT_BASE_DIR. This is intended for prewarm/retry flows before H100 work.
# Do not use this for source-downloadable data such as ClimbMix, eval bundles,
# Hugging Face datasets, word lists, or identity data.

set -euo pipefail

if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
    echo "ERROR: NANOCHAT_BASE_DIR must be set." >&2
    exit 1
fi

if [ -z "${NANOCHAT_RCLONE_REMOTE:-}" ]; then
    echo "ERROR: NANOCHAT_RCLONE_REMOTE must be set, for example nanochat_gdrive:nanochat." >&2
    exit 1
fi

if ! command -v rclone >/dev/null 2>&1; then
    echo "ERROR: rclone is required but was not found on PATH." >&2
    exit 1
fi

LOG_DIR="$NANOCHAT_BASE_DIR/logs"
STATUS_DIR="$NANOCHAT_BASE_DIR/status"
LOG_FILE="$LOG_DIR/gdrive_restore.log"
RUN_ID="${NANOCHAT_RUN_ID:-}"

SHARED_REMOTE="${NANOCHAT_SHARED_REMOTE:-$NANOCHAT_RCLONE_REMOTE/shared-artifacts}"
RUN_REMOTE="${NANOCHAT_RUN_REMOTE:-}"
if [ -z "$RUN_REMOTE" ] && [ -n "$RUN_ID" ]; then
    RUN_REMOTE="$NANOCHAT_RCLONE_REMOTE/runs/$RUN_ID"
fi

TOKENIZER_REMOTE="${NANOCHAT_TOKENIZER_REMOTE:-$SHARED_REMOTE/tokenizer}"

mkdir -p "$NANOCHAT_BASE_DIR" "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR/gdrive_restore.DONE" "$STATUS_DIR/gdrive_restore.FAILED"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/gdrive_restore.STARTED"

on_error() {
    local exit_code=$?
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/gdrive_restore.FAILED"
    echo "Google Drive restore failed with exit code $exit_code. See $LOG_FILE" >&2
    exit "$exit_code"
}
trap on_error ERR

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting nanochat artifact restore"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "NANOCHAT_RCLONE_REMOTE: $NANOCHAT_RCLONE_REMOTE"
echo "Shared remote: $SHARED_REMOTE"
echo "Run remote: ${RUN_REMOTE:-<unset>}"
echo "Log file: $LOG_FILE"

remote_exists() {
    local remote_path="$1"
    rclone lsjson "$remote_path" --stat >/dev/null 2>&1
}

restore_dir_if_present() {
    local remote_path="$1"
    local local_dir="$2"
    local label="$3"
    if remote_exists "$remote_path"; then
        mkdir -p "$local_dir"
        echo "Restoring $label: $remote_path -> $local_dir"
        rclone copy "$remote_path" "$local_dir" --exclude "*.tmp"
    else
        echo "Skipping missing remote $label: $remote_path"
    fi
}

restore_file_if_present() {
    local remote_path="$1"
    local local_file="$2"
    local label="$3"
    if remote_exists "$remote_path"; then
        mkdir -p "$(dirname "$local_file")"
        echo "Restoring $label: $remote_path -> $local_file"
        rclone copyto "$remote_path" "$local_file"
    else
        echo "Skipping missing remote $label: $remote_path"
    fi
}

restore_dir_if_present "$TOKENIZER_REMOTE" "$NANOCHAT_BASE_DIR/tokenizer" "tokenizer"

if [ -n "$RUN_REMOTE" ]; then
    restore_dir_if_present "$RUN_REMOTE/base_checkpoints" "$NANOCHAT_BASE_DIR/base_checkpoints" "base checkpoints"
    restore_dir_if_present "$RUN_REMOTE/chatsft_checkpoints" "$NANOCHAT_BASE_DIR/chatsft_checkpoints" "SFT checkpoints"
    restore_dir_if_present "$RUN_REMOTE/chatrl_checkpoints" "$NANOCHAT_BASE_DIR/chatrl_checkpoints" "RL checkpoints"
else
    echo "Skipping run-specific checkpoint restore because NANOCHAT_RUN_ID and NANOCHAT_RUN_REMOTE are unset."
fi

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/gdrive_restore.DONE"
trap - ERR

echo "Artifact restore complete"
