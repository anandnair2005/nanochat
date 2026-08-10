# Evidence For P1-029

Draft item:
> TODO P1-029: In the optimizer section, phrase this as optimizer-phase async communication, not backward-phase DDP overlap, unless a different code path or profiler trace proves otherwise.

Source:
- Local file: `nanochat/optim.py`, lines 371-535.
- Local file: `nanochat/gpt.py`, lines 410-414.
- Grep search found no `DistributedDataParallel`, `register_hook`, or `register_post_accumulate_grad_hook` training synchronization path.

Relevant excerpt:

```text
GPT.setup_optimizer chooses DistMuonAdamW when DDP env vars are present. DistMuonAdamW launches async collectives inside optimizer.step(), after local gradients exist.
```

How to use it:
- Keep the current wording: optimizer-phase async communication, not DDP-style backward bucket overlap.

Caveats:
- If a profiler trace later shows NCCL during backward, this can be revisited.
