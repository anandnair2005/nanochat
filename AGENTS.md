# AGENTS.md

## Project Shape
- This is the active `nanochat` repo, not legacy `nanoGPT`; expected top-level code is `nanochat/`, `scripts/`, `tasks/`, `runs/`, and `tests/`.
- Main executable pipeline is `runs/speedrun.sh`: dataset download, tokenizer train/eval, base pretrain/eval, SFT, chat eval, report generation.
- `--depth` is the main model-size dial in pretraining; `scripts/base_train.py` derives width, heads, batch size, LR scaling, horizon, and weight decay from it unless explicitly overridden.

## Setup And Artifacts
- Use `uv`: `uv sync --extra gpu --group dev` for CUDA development, or `uv sync --extra cpu --group dev` for CPU/MPS development.
- Python is pinned by `.python-version` to `3.10`; `pyproject.toml` requires `>=3.10` and pins `torch==2.9.1` through uv CPU/CUDA indexes.
- Intermediate data, tokenizers, checkpoints, reports, and run outputs default to `~/.cache/nanochat`; override with `NANOCHAT_BASE_DIR` when tests/runs must not touch shared cache.
- `.env`, `wandb/`, `report.md`, and cache-like artifacts are intentionally ignored; do not commit generated checkpoints, tokenizer files, datasets, or reports.

## Useful Commands
- Install dev deps: `uv sync --extra gpu --group dev` or `uv sync --extra cpu --group dev`.
- Run all tests: `python -m pytest`.
- Run focused engine tests: `python -m pytest tests/test_engine.py -v`.
- Run attention fallback tests: `python -m pytest tests/test_attention_fallback.py -v -s`; FA3 comparison cases skip unless Hopper/SM90 FA3 is available.
- CPU/MPS demo pipeline: `bash runs/runcpu.sh`.
- Full 8xH100 reference pipeline: `bash runs/speedrun.sh`.
- Quick tiny pretrain smoke test after tokenizer/data exist: `python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20`.

## Data And Tokenizer Order
- Base training requires parquet shards plus a trained tokenizer; the usual order is `python -m nanochat.dataset -n <shards>` then `python -m scripts.tok_train` then `python -m scripts.tok_eval`.
- The current pretraining data is ClimbMix under `$NANOCHAT_BASE_DIR/base_data_climbmix`; `nanochat.dataset` always downloads the final shard as validation in addition to requested train shards.
- `scripts.tok_train` saves tokenizer artifacts to `$NANOCHAT_BASE_DIR/tokenizer`; model/eval scripts assume that location via `get_base_dir()`.

## Training And Distributed Notes
- Launch distributed training with `torchrun --standalone --nproc_per_node=N -m scripts.base_train -- ...`; single GPU/CPU runs omit `torchrun` and use gradient accumulation to preserve total batch semantics.
- Distributed setup is keyed off `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` in `nanochat.common.compute_init()` and uses NCCL only for CUDA.
- Multi-GPU parameter synchronization is not ordinary PyTorch DDP wrapping; `nanochat.optim.DistMuonAdamW` performs async `reduce_scatter`/`all_gather` communication and shards optimizer state ZeRO-2 style.
- Global batch math in train scripts is `device_batch_size * max_seq_len * world_size * grad_accum_steps`; `total_batch_size` must divide the per-forward/backward token count exactly.
- If FA3 is unavailable, SDPA fallback works but sliding-window attention (`--window-pattern` other than `L`) is warned as very inefficient; use `--window-pattern=L` for CPU/MPS or non-Hopper quick checks.
- Precision is controlled by `nanochat.common.COMPUTE_DTYPE`, not `torch.amp.autocast`; override with `NANOCHAT_DTYPE=float32|float16|bfloat16` when needed.
- `--fp8` in `scripts.base_train` requires CUDA/H100-class hardware; do not enable it for CPU/MPS smoke tests.

## Checkpoints And Chat
- Base checkpoints live under `$NANOCHAT_BASE_DIR/base_checkpoints/<model-tag>`; SFT and RL checkpoints use `chatsft_checkpoints` and `chatrl_checkpoints`.
- `scripts.chat_sft` loads a base checkpoint and inherits many training hyperparameters from its metadata unless CLI values override them.
- Before SFT in the reference flow, download identity data to `$NANOCHAT_BASE_DIR/identity_conversations.jsonl`; `runs/speedrun.sh` and `runs/runcpu.sh` show the exact URL.
- CLI chat: `python -m scripts.chat_cli -p "Why is the sky blue?"`; web chat: `python -m scripts.chat_web`, optionally `--num-gpus N` for replicated inference workers.

## Style And Workflow
- There is no configured lint, formatter, typecheck, CI, or pre-commit in this checkout; verification is pytest plus the smallest relevant training/inference smoke command.
- Keep changes minimal and compatible with the repo’s deliberately flat script-based design; avoid introducing framework-style config systems or broad abstractions.
- README asks PR authors to disclose substantial LLM contribution they do not fully understand; preserve that expectation in contribution-facing edits.


<!-- headroom:rtk-instructions -->
# RTK (Rust Token Killer) - Token-Optimized Commands

OpenCode has the RTK plugin installed globally. For ordinary shell commands,
do **not** manually prefix with `rtk`; the plugin rewrites supported commands
automatically before execution for token savings.

## Manual RTK Usage

Use explicit `rtk` only when you need a specific RTK mode, when running outside
OpenCode, or when bypassing/filtering behavior intentionally.

```bash
rtk proxy <cmd>     # run without filtering but track usage
rtk err <cmd>       # show only errors/warnings
rtk summary <cmd>   # show heuristic summary
rtk test <cmd>      # show test failures only
```

## Rules
- Prefer normal commands in OpenCode, e.g. `git status`, `pytest`, `npm run build`.
- For debugging RTK itself, use raw commands or `rtk proxy <cmd>` as appropriate.
- The OpenCode plugin is global for this user and applies across repos on this host.
<!-- /headroom:rtk-instructions -->
