# P2-006: Local Vast Orchestrator

## Claim

The local orchestrator made the cloud workflow agent-friendly while keeping cloud/resource selection and credentials local.

## Evidence

- `tasks/vast_orchestrate.py` lines 3-7 state that it is a local Vast.ai orchestrator, keeps credentials local, and expects Vast instances to be created manually.
- Lines 24-29 define defaults: image `anandnair2005/nanochat-vast:h100`, rclone remote `nanochat_gdrive_runner:`, mount path `/workspace/nanochat-cache`, base save interval `200`, and checkpoint retention `3`.
- Lines 40-51 load local `.env` values without committing them.
- Lines 114-133 load SSH public/private key and rclone config paths from local files/config.
- Lines 190-194 require exactly one running matching instance.
- Lines 295-330 rsync the repo while excluding `.git`, `.venv`, caches, checkpoints, tokenizer, datasets, and `.env`.
- Lines 333-346 wait for SSH readiness.
- Lines 366-370 copy the local rclone config into the remote container.
- Lines 403-424 build the remote training environment, including base dir, run ID, rclone remote, GDrive sync, checkpoint retention, GPU count, FP8/depth/batch/dataset settings, and stage toggles.
- Lines 436-439 require a local `WANDB_API_KEY` when WandB is enabled.
- Lines 441-445 start `runs/speedrun_vast.sh` in a detached tmux session and write a manifest event.
- Lines 456-465 refuse cleanup before `final_sync_complete` unless forced. Note: verify whether current launch path ever writes `final_sync_complete`; draft should avoid claiming this guard was fully exercised unless evidence exists.

## Use In Draft

Use this for the local orchestration and agent-assisted ops section.
