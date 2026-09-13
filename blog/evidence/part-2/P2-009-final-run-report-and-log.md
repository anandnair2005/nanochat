# P2-009: Final Run Report And Log

## Claim

The successful final run was `d24-4xh100-full` on `4× NVIDIA H100 80GB HBM3`, with FP8 enabled, depth `24`, DDP world size `4`, and a complete report/log artifact trail.

## Evidence

- `blog/evidence/part-1/report.md` lines 12-24 report Linux, 96 CPU cores / 192 logical, `4x NVIDIA H100 80GB HBM3`, `316.7 GB` total GPU memory, CUDA `12.8`, Python `3.10.20`, and PyTorch `2.9.1+cu128`.
- `blog/evidence/part-1/report.md` lines 37-87 report base training for run `d24-4xh100-full`: FP8 `True`, recipe `tensorwise`, depth `24`, max sequence length `2048`, window pattern `SSSL`, target param/data ratio `8`, DDP world size `4`, `5,838,471,168` training tokens, final/min validation bpb `0.7194`, MFU `59.79%`, total training time `196.49m`, and peak memory `54256.65MiB`.
- `blog/evidence/part-1/report.md` lines 89-117 report base evaluation with CORE metric `0.2561`.
- `blog/evidence/part-1/report.md` lines 164-184 report SFT chat evaluation with ChatCORE metric `0.3712`.
- `blog/evidence/part-1/report.md` lines 187-205 summarize metrics and total wall clock time `4h36m`.
- `blog/evidence/part-1/speedrun.log` lines 4-6 show `NANOCHAT_RUN_ID: d24-4xh100-full`, `WANDB_RUN: d24-4xh100-full`, and `NANOCHAT_NUM_GPUS: 4`.
- `blog/evidence/part-1/speedrun.log` line 332 and line 6395 show `GPU: NVIDIA H100 80GB HBM3 | Peak FLOPS (BF16): 9.89e+14`.
- `blog/evidence/part-1/speedrun.log` line 6268 records the base-training WandB run URL for `d24-4xh100-full`.
- `blog/evidence/part-1/speedrun.log` line 7066 records the SFT WandB run URL for `d24-4xh100-full`.
- `blog/evidence/part-1/speedrun.log` lines 7121-7131 show final copying of tokenizer, base checkpoints, SFT checkpoints, report, `report.md`, status, artifact sync completion, and a final Google Drive sync.

## Use In Draft

Use this as the primary evidence for final run metrics and artifact sync, while keeping cost claims tied to separate Vast billing evidence.
