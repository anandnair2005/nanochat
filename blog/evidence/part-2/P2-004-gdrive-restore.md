# P2-004: Google Drive Restore

## Claim

New Vast hosts could restore durable generated artifacts before deciding what work to skip or resume.

## Evidence

- `runs/restore_gdrive.sh` lines 3-6 state the restore policy and repeat the warning not to use this for source-downloadable data.
- Lines 25-40 set logs/status paths and write `gdrive_restore.STARTED`.
- Lines 30-37 derive shared/run remotes, with tokenizer defaulting to `$NANOCHAT_RCLONE_REMOTE/shared-artifacts/tokenizer`.
- Lines 64-88 define helper functions to restore directories/files only if remote paths exist.
- Line 90 restores the shared tokenizer if present.
- Lines 92-98 restore run-specific base checkpoints, SFT checkpoints, and RL checkpoints when a run remote is available.
- Line 100 writes `gdrive_restore.DONE` on success.

## Use In Draft

Use this to support the claim that instances were ephemeral but generated state did not have to be.
