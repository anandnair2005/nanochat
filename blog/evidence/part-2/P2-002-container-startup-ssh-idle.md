# P2-002: Container Startup, SSH, And Idle Mode

## Claim

The Vast container starts SSH and then idles until the local orchestrator or a human starts training explicitly.

## Evidence

- `runs/start_vast_container.sh` lines 3-5 describe the entrypoint: prepare SSH access, keep the container idle by default, and normally launch training later from the local orchestrator in a detached tmux session.
- Lines 12-16 define startup controls: `NANOCHAT_START_SSHD`, `NANOCHAT_AUTORUN`, `NANOCHAT_USE_TMUX`, `NANOCHAT_RUN_ID`, and `NANOCHAT_SPEEDRUN_SCRIPT`.
- Lines 59-74 install public keys from environment variables such as `NANOCHAT_SSH_PUBLIC_KEY`, `PUBLIC_KEY`, `SSH_PUBLIC_KEY`, `VAST_SSH_PUBLIC_KEY`, or `AUTHORIZED_KEYS`.
- Lines 76-96 start `sshd`, disable password auth, and allow root login only by public key.
- Lines 129-143 print readiness information, keep autorun disabled by default, and tell the user/orchestrator to launch training explicitly.
- Lines 144-150 default to `sleep infinity` when no command is supplied.

## Use In Draft

Use this to support the operational decision to separate instance startup from training launch.
