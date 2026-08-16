# P2-010: Stage Timing And Scaling Comparison

## Claim

Part 2 should frame the run as validation of NanoChat's speedrun claims without wasting rented GPUs, with GPU utilization and stage timing as central evidence.

## Evidence

- `README.md` lines 12-24 say NanoChat's main development focus is tuning pretraining, that the GPT-2 speedrun metric is wall-clock time to GPT-2-grade capability on an `8XH100` node, and that the leaderboard row for autoresearch round 2 is `1.65` hours with val bpb `0.71800` and CORE `0.2626`.
- `blog/evidence/part-1/report.md` lines 33-38 show the run report started at `2026-07-11 12:05:16` and the base training report section timestamped at `2026-07-11 15:49:09`.
- `blog/evidence/part-1/speedrun.log` lines 1-13 show final-run environment variables: run ID `d24-4xh100-full`, GPU count `4`, FP8 enabled, depth `24`, dataset shards `170`, and log file path.
- `blog/evidence/part-1/speedrun.log` lines 34-45 show Google Drive restore before dataset download, including restored shared tokenizer and missing run-specific checkpoints.
- `blog/evidence/part-1/speedrun.log` lines 46-68 show the bootstrap dataset download of `9` shards and reuse of the existing tokenizer.
- `blog/evidence/part-1/speedrun.log` lines 70-318 show the larger dataset download of `171` shards completing before base training.
- `blog/evidence/part-1/speedrun.log` lines 330-354 show base training initialization on CUDA with distributed world size `4`, H100 detection, FA3 enabled, and FP8 converting `145/158` linear layers.
- `blog/evidence/part-1/report.md` lines 73-85 report base training calculated `5568` iterations, `5,838,471,168` training tokens, DDP world size `4`, MFU `59.79%`, and total training time `196.49m`.
- `blog/evidence/part-1/speedrun.log` lines 6235-6268 show peak memory, total training time `196.49m`, W&B summary `train/mfu 59.86377`, `train/tok_per_sec 495936`, and W&B run URL.
- `blog/evidence/part-1/report.md` lines 89-93 show base evaluation timestamp `2026-07-11 16:00:27` and CORE metric `0.2561`.
- `blog/evidence/part-1/speedrun.log` lines 6393-6421 show SFT startup around `2026-07-11 16:00:44`, loading the base checkpoint, inherited training metadata, and base optimizer state load.
- `blog/evidence/part-1/speedrun.log` lines 7027-7034 show SFT checkpoint saves ending around `2026-07-11 16:28:09` and SFT total training time `19.63m`.
- `blog/evidence/part-1/speedrun.log` lines 7071-7074 show chat eval loading the SFT checkpoint around `2026-07-11 16:28:25`.
- `blog/evidence/part-1/report.md` lines 164-184 show chat evaluation timestamp `2026-07-11 16:41:51` and ChatCORE metric `0.3712`.
- `blog/evidence/part-1/report.md` line 205 reports total wall clock time `4h36m`.

## Derived Timing Notes

- Pretraining itself: `196.49m`, or about `3.27h`.
- Compared with README `8XH100` leaderboard time `1.65h`, the `4×H100` pretraining time is almost exactly `2×` the 8-GPU time, which supports the interpretation that this path is largely GPU-bound and scales roughly linearly for this comparison.
- Status marker metadata after reconnecting the correct Google Drive remote shows `speedrun.STARTED` at `2026-07-11T12:05:13Z` and `speedrun.DONE` at `2026-07-11T16:44:44Z`, a status-window total of about `4h39.5m`.
- Google Drive restore marker contents show `gdrive_restore.STARTED` at `2026-07-11T12:05:17Z` and `gdrive_restore.DONE` at `2026-07-11T12:06:58Z`, about `1.68m`.
- Setup/data restore/download/tokenizer reuse before base training appears to be approximately the gap between report start `12:05:16` and base train init `12:10:07`, about `4m51s`, but this is an approximate wall-clock inference from mixed report/log timestamps.
- Base eval wall time appears to be roughly `11m` from base train completion/start of base eval to the `16:00:27` base-eval report timestamp.
- SFT training time is directly reported as `19.63m`; wall time from SFT startup to final SFT checkpoint save is about `27m` including setup/evaluation/checkpointing.
- Chat eval wall time appears to be about `13m26s` from SFT checkpoint load at `16:28:25` to chat-eval report timestamp `16:41:51`.
- Final marker contents show report completion around `16:41:57`, `speedrun.DONE` at `16:44:44`, and `gdrive_sync.LAST_DONE` at `16:44:54`. The final one-shot sync pass took about `10s`, but this should be described as a final confirmation pass because the background loop had already copied large artifacts.

## Use In Draft

Use this to sharpen the validation narrative and stage timing section. Label inferred wall-clock deltas as approximate.
