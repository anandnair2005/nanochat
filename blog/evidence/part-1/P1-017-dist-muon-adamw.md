# Evidence For P1-017

Draft item:
> TODO P1-017: Read `nanochat/optim.py` closely and explain `DistMuonAdamW` accurately.

Source:
- Local file: `nanochat/optim.py`, lines 299-356 and 371-535.
- Local file: `nanochat/gpt.py`, lines 374-414.

Relevant excerpt:

```text
DistMuonAdamW design goals:
- overlap communication with computation using async ops
- shard optimizer states ZeRO-2 style
- batch small tensors where possible

Communication pattern:
1. launch async reduce ops
2. wait for reduce, compute update, launch gather group by group
3. wait for gathers and copy updated Muon params back

AdamW large params use reduce_scatter gradient slices and all_gather updated slices. Muon params are grouped by shape, stacked, reduce_scattered by group, updated only on owned chunks, then all_gathered back.
```

How to use it:
- Describe optimizer-phase distributed synchronization and state sharding.

Caveats:
- The model is not wrapped in PyTorch DDP. Do not describe this as backward-bucket overlap unless profiler evidence proves it.
