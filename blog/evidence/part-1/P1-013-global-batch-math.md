# Evidence For P1-013

Draft item:
> TODO P1-013: Cite the global batch-size calculation in `scripts/base_train.py`.

Source:
- Local file: `scripts/base_train.py`, lines 407-414.

Relevant excerpt:

```python
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
```

How to use it:
- Explain how the same script works on one GPU or multiple GPUs by changing gradient accumulation to hit the configured total token batch.

Caveats:
- The configured total batch size must divide the per-forward/backward token count exactly.
