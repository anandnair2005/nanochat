# P2-003: Google Drive Artifact Policy

## Claim

The run synced generated artifacts to Google Drive while avoiding source-downloadable data and secrets.

## Evidence

- `runs/sync_gdrive.sh` lines 3-6 state the policy: copy valuable generated artifacts and do not use the sync for ClimbMix, eval bundles, Hugging Face datasets, word lists, or identity data.
- Lines 67-76 implement excludes for temporary files, `.venv`, `wandb`, `base_data_climbmix`, `eval_bundle`, `hf_cache`, `words_alpha.txt`, and `identity_conversations.jsonl`.
- Lines 217-225 copy tokenizer, base checkpoints, SFT checkpoints, RL checkpoints, report directory, and report files if present.
- Lines 226-232 copy status and logs, skipping logs during background loop sync but copying them in final/once mode.
- Lines 234-236 prune base/SFT/RL checkpoint trees according to retention settings.
- Lines 238-241 optionally copy the tokenizer to the shared artifacts destination.
- Lines 246-265 support one-shot and loop modes; loop mode watches for `gdrive_sync.STOP` then performs one final sync.

## Use In Draft

Use this to support the "generated state only" artifact strategy and the recovery story.
