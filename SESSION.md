# SESSION.md

## Conversation Context Before `/init`

The user said they want to understand Andrej Karpathy's active `nanochat` repo thoroughly. They have basic PyTorch knowledge, are a professional Python developer, and understand AI/ML fundamentals, math, and transformer architecture. Their main goals are:

- Understand the repo's overall capabilities.
- Learn how the codebase is structured and how to read it effectively.
- Improve understanding of PyTorch distributed training across multiple GPUs.

## Repo Overview Given

nanochat was described as a compact end-to-end LLM training stack, covering:

- Tokenizer training/eval: `scripts/tok_train.py`, `scripts/tok_eval.py`
- Pretraining/base model: `scripts/base_train.py`, `scripts/base_eval.py`
- GPT model definition: `nanochat/gpt.py`
- Distributed data loading: `nanochat/dataloader.py`
- Custom distributed optimizer: `nanochat/optim.py`
- SFT/chat tuning: `scripts/chat_sft.py`, `tasks/*`
- RL on GSM8K: `scripts/chat_rl.py`
- Inference engine with KV cache/tool use: `nanochat/engine.py`
- Web/CLI chat: `scripts/chat_web.py`, `scripts/chat_cli.py`

The best orienting file is `runs/speedrun.sh`, because it shows the intended full pipeline from an empty machine to a usable chat model.

## Recommended Reading Path

1. Start with `README.md`.
   Focus on “Getting started”, “Precision / dtype”, and “File structure”. The key design idea is that `--depth` is the main scale dial and many other hyperparameters are derived from it.

2. Read `runs/speedrun.sh`.
   It is the repo’s executable table of contents:

   ```text
   dataset download -> tokenizer train/eval -> base pretrain -> base eval -> SFT -> chat eval -> web/CLI
   ```

   Important distributed launch examples:

   ```bash
   torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- ...
   torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- ...
   ```

3. Read `nanochat/common.py`.
   Focus on:

   - `compute_init()`
   - `get_dist_info()`
   - `print0()`
   - `COMPUTE_DTYPE`

   The repo uses `torchrun` environment variables: `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`. `compute_init()` sets the local CUDA device and initializes NCCL.

4. Read `nanochat/gpt.py`.
   Suggested order:

   - `GPTConfig`
   - `Linear`
   - `CausalSelfAttention`
   - `MLP`
   - `Block`
   - `GPT.__init__`
   - `GPT.init_weights`
   - `GPT.forward`
   - `GPT.setup_optimizer`

   Important model features:

   - RoPE, no learned positional embeddings
   - RMSNorm
   - QK norm
   - ReLU squared MLP
   - Untied input/output embeddings
   - Sliding-window attention pattern
   - Value embeddings / ResFormer-style additions
   - Explicit dtype control instead of `torch.amp.autocast`
   - `torch.compile` in training scripts

5. Read `scripts/base_train.py`.
   This is the central training file. Read it in sections:

   - CLI args
   - `compute_init`
   - tokenizer/model creation
   - `torch.compile`
   - scaling-law hyperparameter derivation
   - optimizer creation
   - dataloader creation
   - gradient accumulation calculation
   - training loop
   - eval/checkpoint/logging

   Important distributed batch math:

   ```python
   tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
   world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
   grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
   ```

   This means global batch size is achieved by:

   ```text
   per-GPU microbatch tokens * number of GPUs * gradient accumulation steps
   ```

6. Read `nanochat/dataloader.py`.
   This shows how data is sharded across ranks. Important idea:

   ```python
   rg_idx = ddp_rank
   ...
   rg_idx += ddp_world_size
   ```

   Each rank reads different parquet row groups. Data distribution is handled manually by rank, not through PyTorch `DistributedSampler`.

7. Read `nanochat/optim.py`.
   This is the most important file for understanding multi-GPU training in nanochat.

   nanochat does not use standard PyTorch `DistributedDataParallel` wrapping for parameter synchronization. Instead, it uses a custom optimizer, `DistMuonAdamW`, which performs gradient communication itself.

   Core ideas:

   - AdamW params use `all_reduce` for small params.
   - AdamW large params use `reduce_scatter_tensor` and `all_gather_into_tensor`.
   - Muon matrix params are stacked by shape.
   - Each rank owns a chunk of optimizer work.
   - Optimizer state is sharded, ZeRO-2 style.
   - Communication is launched asynchronously to overlap with compute.

   Suggested reading order:

   - `DistMuonAdamW.step()`
   - `_reduce_adamw`
   - `_compute_adamw`
   - `_reduce_muon`
   - `_compute_muon`
   - `_finish_gathers`

8. Read `nanochat/flash_attention.py`.
   This abstracts Flash Attention 3 vs PyTorch SDPA fallback. It is useful for understanding tensor layouts:

   ```text
   FA3 layout: (B, T, H, D)
   PyTorch SDPA layout: (B, H, T, D)
   ```

9. Read `nanochat/engine.py`.
   This is inference. Focus on:

   - `KVCache`
   - `Engine.generate`
   - `Engine.generate_batch`
   - calculator/tool-use state machine

   This explains efficient chat generation after training.

10. Then read SFT/RL files:

   - `scripts/chat_sft.py`
   - `tasks/common.py`
   - `tasks/smoltalk.py`
   - `tasks/gsm8k.py`
   - `scripts/chat_rl.py`

   SFT is structurally similar to pretraining but uses task mixtures and chat-formatted data. RL is a simpler REINFORCE-like loop over GSM8K samples.

## Distributed Training Mental Model

Typical PyTorch DDP looks like:

```python
model = DistributedDataParallel(model)
loss.backward()
optimizer.step()
```

nanochat instead does roughly:

```python
loss.backward()
optimizer.step()  # optimizer performs distributed gradient reduction/update
```

The model itself is not wrapped in ordinary DDP for synchronization. Gradients are local after backward, then `DistMuonAdamW` communicates gradients and updates parameters.

One training step can be traced as:

1. Each rank gets a different batch from `dataloader.py`.
2. Each rank runs forward/backward locally.
3. Gradients accumulate for `grad_accum_steps`.
4. `DistMuonAdamW.step()` launches async distributed reductions.
5. Each rank updates its shard/chunk of optimizer state.
6. Updated params are gathered back so all ranks have the same model.

## Suggested Hands-On Exercises

Run a tiny CPU smoke test after tokenizer/data exist:

```bash
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
```

If multiple GPUs are available, run a tiny distributed job:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- --depth=4 --max-seq-len=512 --device-batch-size=1 --total-batch-size=2048 --num-iterations=20 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1
```

Then change only `--nproc_per_node` and observe how `grad_accum_steps` changes.

## Files To Prioritize For PyTorch Distributed Learning

```text
nanochat/common.py       # process group setup
scripts/base_train.py    # global batch / grad accumulation / training loop
nanochat/dataloader.py   # rank-aware data sharding
nanochat/optim.py        # custom reduce_scatter/all_gather optimizer
nanochat/gpt.py          # optimizer grouping and model forward
```

The strongest recommendation was to spend the most time on `nanochat/optim.py`, because it exposes the distributed communication primitives directly and is more educational than a standard DDP wrapper example.
