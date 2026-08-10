# Evidence For P1-008

Draft item:
> TODO P1-008: Inspect `scripts/base_train.py` and cite the exact formulas derived from `--depth`.

Source:
- Local file: `scripts/base_train.py`, lines 130-140, 261-305, 407-414.
- Local file: `README.md`, lines 6 and 97.
- Local file: `dev/LOG.md`, lines 223-269 and 502-543.

Relevant excerpt:

```text
Model shape:
base_dim = depth * args.aspect_ratio
model_dim = ceil_to_multiple(base_dim, args.head_dim)
num_heads = model_dim // args.head_dim

Training horizon:
target_tokens = int(args.target_param_data_ratio * num_scaling_params)
num_iterations = target_tokens // total_batch_size

Batch size:
predicted_batch_size = B_REF * (target_tokens / D_REF) ** 0.383
total_batch_size = nearest power of 2

LR scaling:
batch_lr_scale = sqrt(total_batch_size / B_REF)

Weight decay:
weight_decay_scaled = args.weight_decay * sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
```

How to use it:
- Use these formulas in the `--depth` section and optionally include a short code excerpt.

Caveats:
- The exact default target ratio has changed over time in `dev/LOG.md`; cite current code for current behavior and `dev/LOG.md` for rationale/history.
