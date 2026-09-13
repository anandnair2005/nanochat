# Evidence For P1-023

Draft item:
> TODO P1-023: Cite `scripts/base_train.py` for the training-loop placement of `next(train_loader)`.

Source:
- Local file: `scripts/base_train.py`, lines 511-520.

Relevant excerpt:

```python
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    ...
    loss.backward()
    x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
```

How to use it:
- Support the claim that the training loop requests the next batch immediately after backward in each micro-step.

Caveats:
- This is Python-level generator work; exact overlap should be measured if making a strong profiler-level claim.
