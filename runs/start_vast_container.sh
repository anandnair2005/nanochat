#!/usr/bin/env bash

# Minimal Vast container entrypoint. It prepares SSH access and keeps the
# container idle by default. Training is normally launched later by the local
# orchestrator inside an explicit detached tmux session.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NANOCHAT_START_SSHD="${NANOCHAT_START_SSHD:-1}"
NANOCHAT_AUTORUN="${NANOCHAT_AUTORUN:-0}"
NANOCHAT_USE_TMUX="${NANOCHAT_USE_TMUX:-0}"
NANOCHAT_RUN_ID="${NANOCHAT_RUN_ID:-d24}"
NANOCHAT_SPEEDRUN_SCRIPT="${NANOCHAT_SPEEDRUN_SCRIPT:-$REPO_ROOT/runs/speedrun_vast.sh}"

if [ -n "${NANOCHAT_COMPLETION_DIR:-}" ]; then
    COMPLETION_DIR="$NANOCHAT_COMPLETION_DIR"
elif [ -n "${NANOCHAT_BASE_DIR:-}" ]; then
    COMPLETION_DIR="$NANOCHAT_BASE_DIR/status"
else
    COMPLETION_DIR="/tmp/nanochat-status"
fi

if [ -n "${NANOCHAT_LOG_FILE:-}" ]; then
    LOG_FILE="$NANOCHAT_LOG_FILE"
elif [ -n "${NANOCHAT_BASE_DIR:-}" ]; then
    LOG_FILE="$NANOCHAT_BASE_DIR/logs/speedrun.log"
else
    LOG_FILE=""
fi

mkdir -p "$COMPLETION_DIR"
touch "$COMPLETION_DIR/container.STARTED"

child_pid=""

finish() {
    local status="$?"
    if [ "$status" -eq 0 ]; then
        touch "$COMPLETION_DIR/container.DONE"
    else
        touch "$COMPLETION_DIR/container.FAILED"
    fi
}

terminate() {
    if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
    exit 143
}

trap finish EXIT
trap terminate TERM INT

install_authorized_keys() {
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    touch /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys

    local key
    for var_name in NANOCHAT_SSH_PUBLIC_KEY PUBLIC_KEY SSH_PUBLIC_KEY VAST_SSH_PUBLIC_KEY AUTHORIZED_KEYS; do
        key="${!var_name:-}"
        if [ -n "$key" ]; then
            printf '%s\n' "$key" >> /root/.ssh/authorized_keys
        fi
    done

    sort -u /root/.ssh/authorized_keys -o /root/.ssh/authorized_keys
}

start_sshd() {
    if [ "$NANOCHAT_START_SSHD" != "1" ]; then
        return 0
    fi
    if ! command -v sshd >/dev/null 2>&1; then
        echo "sshd not installed; skipping SSH startup"
        return 0
    fi

    install_authorized_keys
    mkdir -p /var/run/sshd /etc/ssh/sshd_config.d
    ssh-keygen -A >/dev/null
    cat >/etc/ssh/sshd_config.d/nanochat.conf <<'SSHD_CONFIG'
PasswordAuthentication no
PermitRootLogin prohibit-password
PubkeyAuthentication yes
SSHD_CONFIG

    /usr/sbin/sshd
    echo "sshd started"
}

run_with_optional_log() {
    if [ -n "$LOG_FILE" ]; then
        mkdir -p "$(dirname "$LOG_FILE")"
        bash -lc "$*" 2>&1 | tee -a "$LOG_FILE"
    else
        bash -lc "$*"
    fi
}

run_autorun() {
    local session="nanochat-$NANOCHAT_RUN_ID"
    local command="cd '$REPO_ROOT' && bash '$NANOCHAT_SPEEDRUN_SCRIPT'"

    if [ "$NANOCHAT_USE_TMUX" = "1" ]; then
        if ! command -v tmux >/dev/null 2>&1; then
            echo "NANOCHAT_USE_TMUX=1 but tmux is not installed" >&2
            return 1
        fi
        if tmux has-session -t "$session" 2>/dev/null; then
            echo "tmux session already exists: $session" >&2
            return 1
        fi
        tmux new-session -d -s "$session" "$command"
        echo "started autorun tmux session: $session"
        echo "attach with: tmux attach -t $session"
        return 0
    fi

    run_with_optional_log "$command"
}

start_sshd

echo "nanochat Vast container ready"
echo "repo: $REPO_ROOT"
echo "run id: $NANOCHAT_RUN_ID"
echo "default tmux session for orchestrator: nanochat-$NANOCHAT_RUN_ID"

if [ "$NANOCHAT_AUTORUN" = "1" ]; then
    run_autorun
    exit $?
fi

echo "autorun disabled; waiting for orchestrator or manual SSH"
echo "orchestrator should launch training with an explicit detached tmux session"

if [ "$#" -eq 0 ]; then
    set -- sleep infinity
fi

"$@" &
child_pid="$!"
wait "$child_pid"
