# P2-005: `speedrun_vast.sh` Workflow

## Claim

`runs/speedrun_vast.sh` adapts NanoChat's normal speedrun for ephemeral Vast hosts by restoring artifacts, downloading data locally, resuming where possible, syncing during/after the run, and generating the final report.

## Evidence

- `runs/speedrun_vast.sh` lines 3-5 describe the target-host workflow: restore durable Drive artifacts, fetch source-downloadable data locally, train/evaluate, and continuously sync back.
- Lines 15-22 require `NANOCHAT_BASE_DIR` unless explicitly allowed to use the default cache.
- Lines 24-46 set run identity, logging, GPU count, FP8/depth defaults, dataset shard counts, evaluation stage toggles, and restore behavior.
- Lines 66-68 create base/log/status directories and write `speedrun.STARTED`.
- Lines 70-77 define error handling that writes `speedrun.FAILED`, stops background sync, and runs a final sync attempt.
- Lines 131-159 define background and final Google Drive sync behavior.
- Lines 210-216 run `restore_gdrive.sh` when configured.
- Lines 221-240 download dataset shards only when needed and reuse an existing tokenizer if present.
- Lines 250-293 resume base pretraining from the latest restored checkpoint if available, then launch `scripts.base_train` with `torchrun`.
- Lines 294-328 run base eval, SFT, and chat eval depending on flags and checkpoint availability.
- Lines 334-342 generate the report, stop background sync, write `speedrun.DONE`, and run final sync.

## Use In Draft

Use this as the core source for the Vast-friendly pipeline section.
