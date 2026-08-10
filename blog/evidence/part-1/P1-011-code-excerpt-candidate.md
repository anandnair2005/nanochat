# Evidence For P1-011

Draft item:
> TODO P1-011: Include a small code excerpt only if it teaches the batch math or depth-derived config clearly.

Source:
- Local file: `scripts/base_train.py`, lines 130-140 and 407-414.

Candidate excerpts:

```python
base_dim = depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
```

```python
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
```

How to use it:
- Prefer one or both short snippets over long code blocks.

Caveats:
- Medium code blocks should stay short; link to GitHub for the full file.
