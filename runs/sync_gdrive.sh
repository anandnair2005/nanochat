#!/usr/bin/env bash

# Copy valuable generated nanochat run artifacts to a run-specific rclone destination.
# Despite the name, any rclone destination is supported via NANOCHAT_RCLONE_REMOTE.
# Do not use this for source-downloadable data such as ClimbMix, eval bundles,
# Hugging Face datasets, word lists, or identity data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
    echo "ERROR: NANOCHAT_BASE_DIR must be set." >&2
    exit 1
fi

if [ -z "${NANOCHAT_RUN_ID:-}" ]; then
    if [ "${NANOCHAT_ALLOW_DEFAULT_RUN_ID:-0}" = "1" ]; then
        export NANOCHAT_RUN_ID="sync-$(date -u +%Y%m%dT%H%M%SZ)"
    else
        echo "ERROR: NANOCHAT_RUN_ID must be set. Set NANOCHAT_ALLOW_DEFAULT_RUN_ID=1 to generate one." >&2
        exit 1
    fi
fi

if [ -z "${NANOCHAT_RCLONE_REMOTE:-}" ]; then
    echo "ERROR: NANOCHAT_RCLONE_REMOTE must be set, for example gdrive:nanochat-speedruns." >&2
    exit 1
fi

if ! command -v rclone >/dev/null 2>&1; then
    echo "ERROR: rclone is required but was not found on PATH." >&2
    exit 1
fi

LOG_DIR="$NANOCHAT_BASE_DIR/logs"
STATUS_DIR="$NANOCHAT_BASE_DIR/status"
LOG_FILE="$LOG_DIR/gdrive_sync.log"
SYNC_MODE="${NANOCHAT_SYNC_MODE:-once}"
SYNC_INTERVAL_SECONDS="${NANOCHAT_SYNC_INTERVAL_SECONDS:-600}"
DEST="${NANOCHAT_RUN_REMOTE:-$NANOCHAT_RCLONE_REMOTE/runs/$NANOCHAT_RUN_ID}"
SHARED_DEST="${NANOCHAT_SHARED_REMOTE:-$NANOCHAT_RCLONE_REMOTE/shared-artifacts}"
STOP_MARKER="$STATUS_DIR/gdrive_sync.STOP"
RETAIN_CHECKPOINTS="${NANOCHAT_RETAIN_CHECKPOINTS:-3}"

case "$RETAIN_CHECKPOINTS" in
    ''|*[!0-9]*)
        echo "ERROR: NANOCHAT_RETAIN_CHECKPOINTS must be a non-negative integer, got '$RETAIN_CHECKPOINTS'." >&2
        exit 1
        ;;
esac

mkdir -p "$LOG_DIR" "$STATUS_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting nanochat artifact sync"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "NANOCHAT_RUN_ID: $NANOCHAT_RUN_ID"
echo "Destination: $DEST"
echo "Shared destination: $SHARED_DEST"
echo "Sync mode: $SYNC_MODE"
echo "Retain checkpoints: $RETAIN_CHECKPOINTS"
echo "Log file: $LOG_FILE"

RCLONE_COPY_ARGS=(
    --exclude "*.tmp"
    --exclude ".venv/**"
    --exclude "wandb/**"
    --exclude "base_data_climbmix/**"
    --exclude "eval_bundle/**"
    --exclude "hf_cache/**"
    --exclude "words_alpha.txt"
    --exclude "identity_conversations.jsonl"
)

copy_dir_if_exists() {
    local source_dir="$1"
    local dest_name="$2"
    if [ -d "$source_dir" ]; then
        echo "Copying directory $source_dir -> $DEST/$dest_name"
        rclone copy "$source_dir" "$DEST/$dest_name" "${RCLONE_COPY_ARGS[@]}"
    else
        echo "Skipping missing directory $source_dir"
    fi
}

copy_file_if_exists() {
    local source_file="$1"
    local dest_name="$2"
    if [ -f "$source_file" ]; then
        echo "Copying file $source_file -> $DEST/$dest_name"
        rclone copyto "$source_file" "$DEST/$dest_name"
    else
        echo "Skipping missing file $source_file"
    fi
}

copy_logs_if_exists() {
    local source_dir="$NANOCHAT_BASE_DIR/logs"
    local snapshot_dir
    if [ ! -d "$source_dir" ]; then
        echo "Skipping missing directory $source_dir"
        return 0
    fi

    snapshot_dir="$(mktemp -d "$NANOCHAT_BASE_DIR/.gdrive-log-snapshot.XXXXXX")"
    if ! cp -a "$source_dir"/. "$snapshot_dir"/; then
        rm -rf "$snapshot_dir"
        return 1
    fi

    echo "Copying log snapshot $snapshot_dir -> $DEST/logs"
    if ! rclone copy "$snapshot_dir" "$DEST/logs" "${RCLONE_COPY_ARGS[@]}"; then
        rm -rf "$snapshot_dir"
        return 1
    fi
    rm -rf "$snapshot_dir"
}

checkpoint_steps_local() {
    local checkpoint_dir="$1"
    local file step
    [ -d "$checkpoint_dir" ] || return 0
    for file in "$checkpoint_dir"/model_*.pt; do
        [ -e "$file" ] || continue
        step="$(basename "$file")"
        step="${step#model_}"
        step="${step%.pt}"
        printf '%s\n' "$step"
    done | sort -n
}

checkpoint_steps_remote() {
    local remote_dir="$1"
    local file step
    rclone lsf "$remote_dir" --include 'model_*.pt' 2>/dev/null | while read -r file; do
        [ -n "$file" ] || continue
        file="$(basename "$file")"
        step="${file#model_}"
        step="${step%.pt}"
        printf '%s\n' "$step"
    done | sort -n
}

prune_local_checkpoint_dir() {
    local checkpoint_dir="$1"
    local label="$2"
    local steps count prune_count step
    [ "$RETAIN_CHECKPOINTS" -gt 0 ] || return 0
    steps="$(checkpoint_steps_local "$checkpoint_dir")"
    [ -n "$steps" ] || return 0
    count="$(printf '%s\n' "$steps" | wc -l | tr -d ' ')"
    prune_count=$((count - RETAIN_CHECKPOINTS))
    [ "$prune_count" -gt 0 ] || return 0
    echo "Pruning local $label checkpoints in $checkpoint_dir to latest $RETAIN_CHECKPOINTS"
    printf '%s\n' "$steps" | while read -r step; do
        [ "$prune_count" -gt 0 ] || break
        rm -f "$checkpoint_dir/model_${step}.pt" "$checkpoint_dir/meta_${step}.json" "$checkpoint_dir/optim_${step}_rank"*.pt
        prune_count=$((prune_count - 1))
    done
}

prune_remote_checkpoint_dir() {
    local remote_dir="$1"
    local label="$2"
    local steps count prune_count step file
    [ "$RETAIN_CHECKPOINTS" -gt 0 ] || return 0
    if ! rclone lsjson "$remote_dir" --stat >/dev/null 2>&1; then
        return 0
    fi
    steps="$(checkpoint_steps_remote "$remote_dir")"
    [ -n "$steps" ] || return 0
    count="$(printf '%s\n' "$steps" | wc -l | tr -d ' ')"
    prune_count=$((count - RETAIN_CHECKPOINTS))
    [ "$prune_count" -gt 0 ] || return 0
    echo "Pruning remote $label checkpoints in $remote_dir to latest $RETAIN_CHECKPOINTS"
    printf '%s\n' "$steps" | while read -r step; do
        [ "$prune_count" -gt 0 ] || break
        for file in "model_${step}.pt" "meta_${step}.json"; do
            rclone deletefile "$remote_dir/$file" >/dev/null 2>&1 || true
        done
        rclone lsf "$remote_dir" --include "optim_${step}_rank*.pt" 2>/dev/null | while read -r file; do
            [ -n "$file" ] || continue
            rclone deletefile "$remote_dir/$file" >/dev/null 2>&1 || true
        done
        prune_count=$((prune_count - 1))
    done
}

prune_checkpoint_tree() {
    local local_root="$1"
    local remote_root="$2"
    local label="$3"
    local dir tag
    if [ -d "$local_root" ]; then
        for dir in "$local_root"/*; do
            [ -d "$dir" ] || continue
            tag="$(basename "$dir")"
            prune_local_checkpoint_dir "$dir" "$label/$tag"
            prune_remote_checkpoint_dir "$remote_root/$tag" "$label/$tag"
        done
    elif rclone lsjson "$remote_root" --stat >/dev/null 2>&1; then
        rclone lsf "$remote_root" --dirs-only 2>/dev/null | while read -r tag; do
            [ -n "$tag" ] || continue
            tag="${tag%/}"
            prune_remote_checkpoint_dir "$remote_root/$tag" "$label/$tag"
        done
    fi
}

run_once() {
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/gdrive_sync.LAST_START"

    copy_dir_if_exists "$NANOCHAT_BASE_DIR/tokenizer" "tokenizer"
    copy_dir_if_exists "$NANOCHAT_BASE_DIR/base_checkpoints" "base_checkpoints"
    copy_dir_if_exists "$NANOCHAT_BASE_DIR/chatsft_checkpoints" "chatsft_checkpoints"
    copy_dir_if_exists "$NANOCHAT_BASE_DIR/chatrl_checkpoints" "chatrl_checkpoints"
    copy_dir_if_exists "$NANOCHAT_BASE_DIR/report" "report"

    copy_file_if_exists "$REPO_ROOT/report.md" "report.md"
    copy_file_if_exists "$NANOCHAT_BASE_DIR/report.md" "report.md"

    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATUS_DIR/gdrive_sync.LAST_DONE"
    copy_dir_if_exists "$NANOCHAT_BASE_DIR/status" "status"
    if [ "$SYNC_MODE" = "loop" ]; then
        echo "Skipping logs during background sync loop; final sync copies logs."
    else
        copy_logs_if_exists
    fi

    prune_checkpoint_tree "$NANOCHAT_BASE_DIR/base_checkpoints" "$DEST/base_checkpoints" "base_checkpoints"
    prune_checkpoint_tree "$NANOCHAT_BASE_DIR/chatsft_checkpoints" "$DEST/chatsft_checkpoints" "chatsft_checkpoints"
    prune_checkpoint_tree "$NANOCHAT_BASE_DIR/chatrl_checkpoints" "$DEST/chatrl_checkpoints" "chatrl_checkpoints"

    if [ "${NANOCHAT_SYNC_SHARED_ARTIFACTS:-0}" = "1" ]; then
        if [ -d "$NANOCHAT_BASE_DIR/tokenizer" ]; then
            echo "Copying shared tokenizer $NANOCHAT_BASE_DIR/tokenizer -> $SHARED_DEST/tokenizer"
            rclone copy "$NANOCHAT_BASE_DIR/tokenizer" "$SHARED_DEST/tokenizer" "${RCLONE_COPY_ARGS[@]}"
        fi
    fi
}

case "$SYNC_MODE" in
    once)
        run_once
        ;;
    loop)
        echo "Looping until $STOP_MARKER exists"
        while [ ! -f "$STOP_MARKER" ]; do
            run_once
            if [ -f "$STOP_MARKER" ]; then
                break
            fi
            slept=0
            while [ "$slept" -lt "$SYNC_INTERVAL_SECONDS" ] && [ ! -f "$STOP_MARKER" ]; do
                sleep 5
                slept=$((slept + 5))
            done
        done
        echo "Stop marker found; running one final sync"
        run_once
        ;;
    *)
        echo "ERROR: NANOCHAT_SYNC_MODE must be 'once' or 'loop', got '$SYNC_MODE'." >&2
        exit 1
        ;;
esac

echo "Artifact sync complete: $DEST"
