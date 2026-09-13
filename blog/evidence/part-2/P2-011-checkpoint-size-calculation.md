# P2-011: Checkpoint Size Calculation

## Claim

Uploading weights and optimizer checkpoints is operationally meaningful because a final checkpoint consists of one large model file plus optimizer shards from each rank.

## Evidence

- `nanochat/checkpoint_manager.py` lines 42-59 show that rank 0 saves `model_<step>.pt` and `meta_<step>.json`, while every rank saves `optim_<step>_rank<rank>.pt` when optimizer data is present.
- `scripts/base_train.py` lines 477-499 call `save_checkpoint()` with `orig_model.state_dict()`, `optimizer.state_dict()`, metadata, and `rank=ddp_rank`.
- `scripts/chat_sft.py` lines 398-419 also save SFT checkpoints with model state, optimizer state, and metadata.
- `nanochat/optim.py` lines 326-343 describe ZeRO-2-style optimizer state sharding for AdamW and chunked/sharded Muon optimizer state.
- `nanochat/optim.py` lines 426-430 show AdamW state initialization with `exp_avg` and `exp_avg_sq` tensors for the relevant parameter slice.
- `nanochat/optim.py` lines 464-470 show Muon state initialization with `momentum_buffer` and `second_momentum_buffer` tensors for the owned chunk.
- `blog/evidence/part-1/report.md` line 71 reports `1,384,122,122` parameters for the final run.
- `blog/evidence/part-1/speedrun.log` lines 6225-6234 show final base checkpoint saves for optimizer shards `rank0` through `rank3`, model parameters, and metadata.
- `blog/evidence/part-1/speedrun.log` lines 7027-7034 show final SFT checkpoint saves for optimizer shards `rank0` through `rank3`, model parameters, and metadata.

## Calculation

Using `1,384,122,122` parameters:

- bf16/fp16 model weights at 2 bytes per parameter: `2,768,244,244` bytes, about `2.77 GB` or `2.58 GiB` before PyTorch serialization overhead.
- fp32 model weights at 4 bytes per parameter: `5,536,488,488` bytes, about `5.54 GB` or `5.16 GiB` before overhead.
- A simple two-buffer fp32 optimizer-state upper-bound for all parameters is `11,072,976,976` bytes, about `11.07 GB` or `10.31 GiB`.
- If that optimizer state were perfectly sharded across 4 ranks, the rough per-rank optimizer-state budget would be about `2.77 GB` or `2.58 GiB` before overhead.
- The actual optimizer checkpoint size can differ because AdamW and Muon state shapes/dtypes differ, some state is sharded by parameter slice or chunk, metadata adds overhead, and PyTorch serialization is not raw tensor bytes.

## Actual Remote Sizes

After reconnecting `nanochat_gdrive_runner:` to the correct folder, remote metadata showed:

- Base checkpoint set: `9.283 GiB` / `9,967,588,584` bytes across 6 objects.
- SFT checkpoint set: `9.283 GiB` / `9,967,588,054` bytes across 6 objects.
- Base model file: `model_005568.pt`, `4,227,935,530` bytes.
- SFT model file: `model_000485.pt`, `4,227,935,530` bytes.
- Each base optimizer shard: `1,434,912,917` bytes.
- Each SFT optimizer shard: `1,434,912,917` bytes.

The rough raw-parameter estimate undercounted the actual model file because the real checkpoint includes saved tensors beyond only naive parameter bytes and PyTorch serialization overhead. The operational conclusion is stronger with real metadata: each base/SFT checkpoint set is roughly `9.3 GiB`, so retaining and syncing checkpoints materially affects storage and transfer planning.

## Use In Draft

Use this for a compact section explaining why final artifact upload time/storage mattered and why checkpoint retention was part of the operational design.
