# Evidence For P1-016

Draft item:
> TODO P1-016: Add final H100 report source for `59.79%` MFU.

Primary source:
- Local evidence copy: `blog/evidence/part-1/report.md`
- Original artifact source: GDrive run artifact `d24-4xh100-full/report.md`

Supporting sources:
- Local evidence copy: `blog/evidence/part-1/speedrun.log`
- Original artifact source: GDrive run artifact `d24-4xh100-full/logs/speedrun.log`
- Supporting image: `blog/evidence/part-1/WandbTrainMFU.png`
- Supporting image: `blog/evidence/part-1/WandBGPUUtilization.png`

Relevant excerpt from `report.md`:

```text
Hardware:
- GPUs: 4x NVIDIA H100 80GB HBM3
- CUDA Version: 12.8

Base model training:
- run: d24-4xh100-full
- fp8: True
- fp8_recipe: tensorwise
- depth: 24
- max_seq_len: 2048
- window_pattern: SSSL
- target_param_data_ratio: 8.0000
- device_batch_size: 16
- Number of training tokens: 5,838,471,168
- DDP world size: 4
- Minimum validation bpb: 0.7194
- Final validation bpb: 0.7194
- CORE metric estimate: 0.2607
- MFU %: 59.79%
- Total training flops: 2.788002e+19
- Total training time: 196.49m
```

Relevant excerpt from `speedrun.log`:

```text
Starting nanochat Vast speedrun
NANOCHAT_RUN_ID: d24-4xh100-full
WANDB_RUN: d24-4xh100-full
NANOCHAT_NUM_GPUS: 4
Destination: nanochat_gdrive_runner:/runs/d24-4xh100-full
```

How to use it:
- Use `report.md` as the primary evidence for the `59.79%` base-training MFU claim.
- Use `speedrun.log` to tie the report to run ID `d24-4xh100-full`, 4 GPUs, WandB run identity, and GDrive artifact destination.
- Use WandB images only as visual support for charts, not as the primary source for the exact `59.79%` number.

Caveats:
- The report's hardware section lists an hourly rate of `$12.00/hour`; cost claims should use Vast billing evidence in Part 2 instead.
- This is final H100-run evidence. Do not mix it with smoke-test MFU evidence.
