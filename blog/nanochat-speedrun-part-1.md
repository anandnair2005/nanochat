# NanoChat Speedrun, Part 1: The Code That Makes H100s Go Brrrr

*A code walkthrough of NanoChat's single-node H100 training stack and the single-knob `--depth` centered design.*

![Cover Image (Replace with your image in Medium)](https://via.placeholder.com/800x400?text=Insert+Hero+Image+Here)

> **Disclosure:** I used AI assistance to edit and refine this post, but the ideas, interpretations, and conclusions are mine.

In Jan-Feb 2026, I learned about Andrej Karpathy's [NanoChat](https://github.com/karpathy/nanochat) repo and his experiments with AutoResearch. I decided to jump in and see what the speedrun was really doing.

NanoChat describes itself as a simple experimental harness for training LLMs on a single GPU node. It covers tokenization, pretraining, fine-tuning, evaluation, inference, and a chat UI. I wanted to understand how far the speedrun could be pushed as a practical end-to-end training run on rented GPUs.

> The headline result belongs in Part 2: my final successful run used a `4× NVIDIA H100 80GB HBM3` Vast.ai instance, depth `24`, FP8, DDP world size `4`, and `5,838,471,168` training tokens. The generated report says base training reached **59.79% MFU**. The final instance cost was roughly **$41**, or about **$48** including related setup and smoke-test instances from that day.

But before Docker images, Vast.ai, SSH issues, Google Drive backups, and billing details, I want to start with the code.

NanoChat is interesting because it is not just a pile of scripts. It is a compact training recipe. The repo is small enough for a motivated beginner to read, but opinionated enough to encode serious systems knowledge: single-node distributed training, H100-specific acceleration paths, batch-size arithmetic, optimizer communication, checkpointing, SFT, evals, reporting, and a simple command-line interface over the whole thing.

The central design choice is that NanoChat tries to make the speedrun understandable. Its main user-facing model-size knob is `--depth`. From that one integer, the pretraining script derives much of the model and training configuration: width, heads, token horizon, global batch size, learning-rate scaling, and weight decay.

I am not independently validating those scaling-law choices here. That deserves its own experiment. This post is about how the code is put together and why, after reading it, I expected NanoChat to make good use of a small H100 node.

## Related Speedrun Lineage

NanoChat sits in a lineage of speedrun-style language-model training work. Its README says the GPT-2 speedrun leaderboard was inspired by [`modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt), and later notes say NanoChat borrows ideas and some implementation from that project.

`modded-nanogpt` is a narrower benchmark: train on `8×H100` as fast as possible until a model reaches a target FineWeb validation loss. NanoChat's speedrun is broader. It includes tokenizer work, base pretraining, base eval, SFT, chat eval, and report generation. So I am treating `modded-nanogpt` as lineage, not as an apples-to-apples comparison.

***

## The Shape Of NanoChat

The main executable spine is [`runs/speedrun.sh`](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh). The comments say it is configured to train a GPT-2-grade LLM, including pretraining and fine-tuning, on a blank `8×H100` node.

The script is refreshingly direct. In order, it sets up the Python environment with `uv`, downloads data, trains and evaluates the tokenizer, runs base pretraining with `torchrun`, runs base evaluation, downloads synthetic identity conversations, runs SFT, runs chat evaluation, and generates the final report.

That matters because the final runtime and cost are not just measuring raw pretraining throughput. They include the operational reality of producing a usable artifact and a report at the end.

One scope note: the reference speedrun path runs SFT and chat eval, but not RL. NanoChat does include [`scripts/chat_rl.py`](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py), a simpler GRPO/REINFORCE-like GSM8K loop, but RL is part of the broader repo rather than the Part 1 path I am focusing on.

## The Model NanoChat Builds

The model lives in [`nanochat/gpt.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py). It is a compact GPT-style transformer with rotary embeddings, QK norm, untied token embedding and `lm_head`, ReLU-squared MLPs, RMSNorm, bias-free linear layers, optional GQA support, sliding-window attention, and FA3 integration.

The high-level flow is:

```text
token embedding → previous-token smear → transformer blocks → optional mid-layer backout → final norm → lm_head
```

`--depth` controls the number of repeated [`Block`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L142-L151) modules. The rest of the model shape is derived around that.

| Model piece | Parameter / default | What it affects | Source |
| --- | --- | --- | --- |
| Context length | `sequence_len`, default `2048` through `--max-seq-len` | Maximum training context and rotary cache sizing. | [`GPTConfig`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L28-L40), [`base_train.py`](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L49-L58) |
| Vocabulary | `vocab_size`, loaded from tokenizer | Embedding table and `lm_head`; padded internally for efficiency. | [`GPT.__init__`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L154-L200) |
| Number of blocks | `n_layer = depth` | How many transformer blocks are stacked. | [`build_model_meta`](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L130-L144) |
| Width | `n_embd = model_dim` | Residual-stream width and MLP/attention matrix sizes. | [`build_model_meta`](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L130-L144) |
| Query heads | `n_head = num_heads` | Number of query attention heads. | [`CausalSelfAttention`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L65-L126) |
| KV heads | `n_kv_head`, equal to `n_head` in base training | Supports GQA in the model definition, though base training sets KV heads equal to query heads. | [`GPTConfig`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L28-L40) |
| Attention window | `window_pattern`, default `SSSL` | Tiles short/full-context attention windows across layers; final layer always gets full context. | [`_compute_window_sizes`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L285-L312) |
| Attention block | Q/K/V projections, RoPE, QK norm, FA3/SDPA call | The main sequence-mixing path. | [`CausalSelfAttention.forward`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L82-L126) |
| MLP block | `n_embd → 4 × n_embd → n_embd`, ReLU squared | The tokenwise feed-forward path inside each block. | [`MLP`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L129-L139) |
| Residual extras | `resid_lambdas`, `x0_lambdas`, smear, backout, value embeddings | Small architectural additions inspired by speedrun-style experiments. | [`GPT.__init__`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L154-L200), [`GPT.forward`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L416-L481) |

This is one of the reasons NanoChat is so readable. The architecture is not trivial, but the code does not hide it behind a framework stack.

## The One-Knob Design: `--depth`

The beginner-friendly design choice is that NanoChat does not ask the user to tune dozens of hyperparameters before doing anything useful.

The README says NanoChat is configured around one single complexity dial: the depth of the transformer. The code in [`scripts/base_train.py`](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py) implements that idea using a reference-model strategy.

The `d12` model is the anchor point. Its compute-optimal horizon and batch size are treated as empirically measured values from NanoChat experimentation. Other depths are extrapolated from that reference using scaling laws plus practical hardware constraints.

The dependency chain is roughly:

```text
depth
  → base_dim = depth × aspect_ratio
  → model_dim, rounded to head_dim
  → num_heads
  → scaling_params
  → target_tokens
  → total_batch_size
  → learning-rate and weight-decay scaling
```

The most important defaults and formulas are split into two groups.

Inputs and reference constants:

| Name | Role / default | Formula or value | Paper / evidence |
| --- | --- | --- | --- |
| `depth` | Main model-size knob. Default: `20`. | User input. | NanoChat design choice. |
| `aspect_ratio` | Width-per-layer scaling constant. Default: `64`. | User input. | Architecture convention in [`scripts/base_train.py`](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L49-L58). |
| `head_dim` | Target attention head dimension. Default: `128`. | User input. | Hardware/layout convention; FA3 wants compatible head dimensions. |
| `target_param_data_ratio` | Token-to-parameter ratio. Default: `12`. | Exposed as a user input. The CLI help notes `Chinchilla=20`; NanoChat's default is `12`. | [Chinchilla](https://arxiv.org/abs/2203.15556) is the reference idea; NanoChat's chosen default comes from its own sweeps in [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md). |
| `total_batch_size` | Global token batch-size override. Default: `-1`. | User input. `-1` means auto-compute it from the scaling rule below. | Practical override in [`scripts/base_train.py`](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L59-L69). |
| `weight_decay` | Reference weight decay. Default: `0.28`. | User input. | Tuned NanoChat optimizer setting. |
| `d12_ref` | Reference model used for extrapolation. | `build_model_meta(12)` | Reference-model transfer / [muP](https://arxiv.org/abs/2203.03466)-style extrapolation in [`scripts/base_train.py`](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L272-L275). |
| `B_REF` | Reference batch size for `d12`. | `2**19` tokens | Empirically measured optimum for the `d12` reference model. |

Derived formulas and scaling rules:

| Name | Role / default | Formula or value | Paper / evidence |
| --- | --- | --- | --- |
| `base_dim` | Unrounded model width. | `depth × aspect_ratio` | Architecture convention. |
| `model_dim` | Rounded model width. | `ceil(base_dim / head_dim) × head_dim` | Practical head-division and FA3 compatibility constraint. |
| `num_heads` | Number of attention heads. | `model_dim / head_dim` | Follows from `model_dim` and `head_dim`. |
| `scaling_params` | Parameter count used for scaling laws. | `transformer_matrices + lm_head` | [Scaling Laws](https://arxiv.org/abs/2001.08361) motivate parameter/data scaling; [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md) says this subset gave cleaner NanoChat sweeps. |
| `target_tokens` | Training horizon for the current model. | `target_param_data_ratio × scaling_params` | [Scaling Laws](https://arxiv.org/abs/2001.08361) and [Chinchilla](https://arxiv.org/abs/2203.15556)-style compute-optimal token budgeting. |
| `D_REF` | Reference training horizon for `d12`. | `target_param_data_ratio × scaling_params(d12_ref)` | Same token-budget rule, applied to the reference model. |
| auto `total_batch_size` | Global token batch size when the override is `-1`. | `B_REF × (target_tokens / D_REF)^0.383`, rounded to a power of two | [Power Lines](https://arxiv.org/abs/2505.13738): optimal batch size scales approximately as `D^0.383`. |
| `batch_lr_scale` | Learning-rate multiplier. | `η ∝ √(total_batch_size / B_REF)` | Standard square-root batch-size scaling for AdamW; NanoChat also applies it to Muon as a practical assumption. |
| `weight_decay_scaled` | Depth-adjusted weight decay. | `λ = λ_ref × √(total_batch_size / B_REF) × (D_REF / target_tokens)` | [T_epoch framework](https://arxiv.org/abs/2405.13698): keep `B / (η × λ × D)` roughly constant. |

The papers in the background are the usual scaling-law family: [Scaling Laws](https://arxiv.org/abs/2001.08361), [Chinchilla](https://arxiv.org/abs/2203.15556), [muP](https://arxiv.org/abs/2203.03466), [Power Lines](https://arxiv.org/abs/2505.13738), and the [T_epoch framework](https://arxiv.org/abs/2405.13698). [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md) adds the NanoChat-specific history: batch-size scaling experiments, parameter-count choices, and depth/token/batch tables.

Again, I am not validating those choices in this post. The point is interface design. NanoChat hides a lot of complexity behind one knob without pretending the complexity does not exist.

## Base Training: Where Most Of The Speedrun Lives

Base pretraining is the expensive core of the speedrun. It is also where most of the H100 utilization story lives.

[`scripts/base_train.py`](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py) is the clearest expression of NanoChat's training recipe. It starts with CLI arguments for runtime, FP8, model shape, optimization, evaluation, and output. It initializes compute with [`compute_init()`](https://github.com/karpathy/nanochat/blob/master/nanochat/common.py#L173-L208), detects GPU metadata, loads the tokenizer, builds the model on the `meta` device, moves it to the target device with `to_empty()`, initializes weights, optionally resumes from checkpoint, optionally converts eligible linear layers to FP8, and compiles the model.

The compile call is intentionally fixed-shape:

```python
model = torch.compile(model, dynamic=False)
```

That choice fits the rest of the design. NanoChat wants stable training batches so `torch.compile` can specialize rather than handle constantly changing shapes.

After that, the script calculates scaling parameters, target tokens, batch size, learning-rate scale, and weight-decay scale. It sets up the combined Muon/AdamW optimizer, initializes dataloaders, computes the number of iterations, defines schedules, and enters the training loop.

The batch math is explicit:

```python
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
```

This is what lets the same script work on one GPU or multiple GPUs while preserving total token batch semantics. World size changes the per-step token count, and gradient accumulation adjusts to hit the configured global batch exactly.

The file also contains the unglamorous speed details: fixed shapes, explicit synchronization points, throughput/MFU logging, and even manual Python garbage-collection management after the first step. That last detail is not flashy, but it is exactly the kind of thing that matters when wall-clock time is the target.

## Single-Node Distributed Training

NanoChat is not trying to solve every distributed training topology. The target is narrower: one machine with `1-8` GPUs, especially H100s.

That constraint keeps the code direct while still scaling across a rented node. Distributed setup is keyed off standard `torchrun` environment variables: `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`. [`compute_init()`](https://github.com/karpathy/nanochat/blob/master/nanochat/common.py#L173-L208) detects distributed execution, sets the CUDA device, initializes NCCL on CUDA, and returns rank/world/device metadata.

One nuance matters. NanoChat initializes a distributed process group, but the base training path does not simply wrap the model in ordinary PyTorch `DistributedDataParallel` and stop there. [`GPT.setup_optimizer()`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L374-L414) switches to [`DistMuonAdamW`](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py) when distributed execution is active. That optimizer handles synchronization and sharded optimizer state explicitly.

This is one of the places where NanoChat feels educational rather than framework-heavy. You can see where the distributed boundary is.

## Making H100s Go Brrrr

The final H100 run reported base-training MFU of `59.79%`. I want to be careful with that number: I am using the generated final H100 `report.md` as the source, not earlier smoke tests. Some smoke runs had unreliable MFU because the GPU peak-FLOPS mapping was missing for those devices.

The final report says:

- **Hardware:** `4× NVIDIA H100 80GB HBM3`.
- **CUDA:** `12.8`.
- **Run:** `d24-4xh100-full`.
- **FP8:** `True`, recipe `tensorwise`.
- **Depth:** `24`.
- **Max sequence length:** `2048`.
- **Window pattern:** `SSSL`.
- **DDP world size:** `4`.
- **Training tokens:** `5,838,471,168`.
- **Final validation bpb:** `0.7194`.
- **MFU:** `59.79%`.
- **Total base-training time:** `196.49m`.

The code paths that likely matter most are H100-aware precision, Flash Attention 3, fixed-shape compiled training, explicit batch math, the data feeder, and the distributed optimizer.

The attention path lives in [`nanochat/gpt.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py) and [`nanochat/flash_attention.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/flash_attention.py). The wrapper tries to load Flash Attention 3 only on Hopper/sm90 GPUs, exposes a small API-compatible `flash_attn` object, and falls back to PyTorch SDPA on non-Hopper GPUs, MPS, CPU, or incompatible dtype paths. The training script warns that this fallback is less efficient, especially with sliding windows.

The FP8 path is similarly explicit. [`base_train.py`](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py#L166-L193) exposes `--fp8` and `--fp8-recipe`. If `--fp8` is set on CUDA, it converts eligible `nn.Linear` modules into FP8 training modules. The filter skips layers whose dimensions are not divisible by `16` and skips very small layers, reflecting hardware constraints of FP8 tensor-core paths.

[`nanochat/fp8.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/fp8.py) is intentionally small: tensorwise dynamic scaling, FP8 quantization, `torch._scaled_mm`, and dequantization. Inputs and weights use `float8_e4m3fn`; gradients use `float8_e5m2`. [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md) records that tensorwise scaling beat rowwise for this scale and that FP8 needs careful layer filtering. In other words, FP8 is not magic. It is a hardware-specific tradeoff that helps on the H100 path when used carefully.

## The Optimizer: `DistMuonAdamW`

One of the most educational files is [`nanochat/optim.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py).

NanoChat uses a combined Muon/AdamW optimizer. AdamW handles embeddings, unembedding, and scalar-ish parameters. Muon handles matrix parameters. On one GPU, that is `MuonAdamW`. In distributed training, `GPT.setup_optimizer()` switches to `DistMuonAdamW`.

The optimizer has three clear goals: overlap communication with computation using async ops, reduce memory by sharding optimizer state ZeRO-2 style, and batch small tensors into fewer communication operations.

The distributed step has a three-phase structure:

```text
launch async reductions
  → wait, compute local updates, launch gathers
  → wait for gathers and copy updated parameters back
```

For large AdamW parameters, each rank receives a gradient slice with `reduce_scatter`, updates its slice, and uses `all_gather` to reconstruct the updated parameter. For Muon parameters, the code groups tensors by shape, stacks them, reduce-scatters the stacked gradients, computes updates for the owned chunk, and gathers updated parameters back.

I would not attribute `59.79%` MFU to the optimizer alone. I do not have an ablation. The safe claim is that the optimizer is designed to reduce memory and move distributed communication into an explicit async optimizer phase. The final utilization number comes from the whole stack: model, attention, FP8, data feeder, optimizer, compilation, and hardware.

## Data Loading And Tokenization

Tokenizer and dataloader code is less flashy than FP8 or distributed optimization, but it matters. If the GPUs are going to go brrrr, the CPU side has to keep feeding them.

NanoChat's speedrun includes tokenizer training and evaluation. [`nanochat/dataset.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/dataset.py) downloads ClimbMix parquet shards into `$NANOCHAT_BASE_DIR/base_data_climbmix`. [`runs/speedrun.sh`](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh) first downloads `8` shards so tokenizer training can begin, then starts the larger `170` shard download in the background while tokenizer work proceeds.

This was not the original data choice. [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md#L107-L126) records a March 2026 upgrade from FineWeb-EDU 100B to ClimbMix 400B. Karpathy describes it as the single biggest improvement to NanoChat's GPT-2 speedrun time, reducing the run from `2 hours 46 minutes` to `2 hours 1 minute`, a `27%` reduction. The same entry says ClimbMix trained more efficiently, letting the target move from `d26` to `d24` while still reaching GPT-2 capability.

The runtime dataloader is in [`nanochat/dataloader.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/dataloader.py). The file comment explains the core idea: BOS-aligned best-fit packing. Every row starts with a BOS token. Documents are packed to minimize cropping. If no document fits the remaining row capacity, the loader crops a document to fill the row exactly. The result is `100%` utilization with no padding, at the cost of cropping some tokens.

The base dataloader choices all point in the same direction: parquet row groups as the batching unit, DDP-rank sharding, batched tokenization, BOS-aligned rows, best-fit packing, no padding, pinned CPU staging when using CUDA, a persistent GPU buffer, and one non-blocking host-to-device copy per batch.

The training loop then asks for the next batch immediately after each backward call, with a comment saying it is prefetching while the GPU is busy. I would not overstate kernel-level overlap without a profiler trace, but the code structure is clear: fixed-shape batches, dense packing, pinned staging, persistent GPU buffers, non-blocking copies, and early fetching.

There is a tradeoff. The BOS-aligned best-fit loader can crop documents when no buffered document fits the remaining row capacity. [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md) records BestFit-Crop at `100%` utilization and about `34.6%` crop waste at `T=2048`. The loader chooses dense, fixed-shape training batches over preserving every token boundary perfectly.

SFT makes a different choice. [`scripts/chat_sft.py`](https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py) also uses BOS-aligned best-fit packing, but when no conversation fits, it pads the row instead of cropping. Padding positions are masked out of cross-entropy. That makes sense: base pretraining has abundant web tokens; SFT conversations are structured, and discarding conversation tails would be a worse tradeoff.

## SFT And RL: Smaller But Part Of The Story

Base pretraining dominates the cost and utilization story, but NanoChat is not only a pretraining script.

For the reference speedrun path, SFT is the relevant next stage. [`runs/speedrun.sh`](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh) downloads synthetic identity conversations, runs [`scripts.chat_sft`](https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py), and then runs [`scripts.chat_eval -i sft`](https://github.com/karpathy/nanochat/blob/master/scripts/chat_eval.py).

`chat_sft.py` loads a base checkpoint, inherits useful pretraining metadata by default, compiles the model, and uses the same optimizer setup pattern as base training. Its data mixture includes SmolTalk, custom identity conversations, MMLU, GSM8K, SimpleSpelling, and SpellingBee.

RL exists, but it should not take over this post. [`scripts/chat_rl.py`](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py) is a simpler GRPO/REINFORCE-like GSM8K loop, not a heavyweight PPO/RLHF setup with separate actor, critic, and reference-policy copies. For Part 1, the important scope point is simple: the speedrun path I am discussing is tokenizer → base pretraining → base eval → SFT → chat eval → report.

## What I Expected Before Running It

After reading the code, my expectation was not "this will be magically cheap." It was more specific.

The repo is designed for exactly the hardware shape I planned to rent: one node with multiple H100s. The `--depth` interface keeps experiment setup simple. The training code contains explicit machinery for distributed execution. The data feeder is designed around dense fixed-shape batches. The H100 path has FA3 and FP8 support. The pipeline produces artifacts and evaluation reports, not just a training loss curve.

The remaining risk was operational: cloud GPU setup, Docker, secrets, artifact backup, failed instances, and not wasting paid H100 time.

After reading the code, my question changed from "can this repo run fast?" to "can I make the cloud workflow boring enough that the code gets a fair shot?"

That is the operational story of Part 2.

## Important But Not Covered Here

Some parts of NanoChat matter, but I am leaving them out to keep this post focused: CPU/MPS precision behavior, inference and KV-cache internals, tokenizer quality evaluation, CORE/chat eval methodology, and a full validation of the scaling-law choices in `dev/LOG.md`.

***

## Part 2 Preview

In Part 2, I will cover what happened when this code met rented GPUs: building a standalone Vast.ai Docker image, keeping credentials out of the image, backing up artifacts with `rclone`, smoke testing before H100 training, failed H100 attempts, the final `4×H100` run, and the billing numbers.

> *The short version is that the code was ready before the cloud workflow was ready. The rest of the work was making the expensive part boring.*

## References

- Andrej Karpathy, [`nanochat`](https://github.com/karpathy/nanochat).
- Keller Jordan et al., [`modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt).
- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html). This inspired the title's "go brrrr" phrasing.
- Jared Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
- Jordan Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), also known as Chinchilla.
- Greg Yang et al., [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466), the muP paper.
- [Power Lines: Scaling Laws for Batch Size in Language Model Pretraining](https://arxiv.org/abs/2505.13738).
- Keller Jordan, [Tuning language model hyperparameters via the T_epoch framework](https://arxiv.org/abs/2405.13698).
