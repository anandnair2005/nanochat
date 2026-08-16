# P2-007: Vast And Rclone Docs

## Claim

The operational workflow was documented so future runs had explicit setup, launch, secret-handling, and cleanup steps.

## Evidence

- `docs/vast_launch.md` lines 3-9 describe the model: one manually-created Vast instance per run, generated artifacts restored from Google Drive, source-downloadable data fetched on target host, and credentials kept local.
- Lines 11-18 list defaults for run ID, image, base directory, rclone remote, save interval, and checkpoint retention.
- Lines 20-31 list secrets that must not be baked into images or committed files.
- Lines 33-51 list required Vast launch settings and explain that the image entrypoint starts `sshd` and waits for the orchestrator.
- Lines 53-75 show launch commands and describe the orchestrator's steps: find one instance, resolve SSH, rsync repo, copy rclone config, pass WandB key, and start tmux.
- Lines 77-101 summarize runtime restore/download/resume/sync behavior and generated-vs-source-downloadable artifact policy.
- Lines 103-131 provide a cheap smoke-test command and explicitly say it validates mechanics, not FA3/FP8/H100 throughput.
- Lines 133-142 describe manifest-driven cleanup.
- `docs/rclone_gdrive_setup.md` lines 14-27 describe recreating the `nanochat_gdrive_runner` remote with `drive.file` scope and a root folder ID.
- Lines 64-90 describe revoking access after the final sync and verifying `speedrun.DONE`, logs, and retained checkpoints before revocation.
- Lines 92-109 say full Drive scope is a fallback only if `drive.file` cannot access the folder.

## Use In Draft

Use this for secret-handling and documentation/reproducibility claims.
