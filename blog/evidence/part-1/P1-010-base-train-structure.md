# Evidence For P1-010

Draft item:
> TODO P1-010: Read `scripts/base_train.py` and write this section around the actual code structure.

Source:
- Local file: `scripts/base_train.py`, full file; key lines 84-119, 120-153, 166-247, 250-317, 330-335, 337-388, 407-545, 548-627.

Relevant structure:

```text
- compute init and WandB setup
- Flash Attention status/warnings
- tokenizer load and vocab size
- build model on meta device, move to target device, initialize weights
- optional checkpoint resume
- optional FP8 conversion
- torch.compile(dynamic=False)
- scaling-law-derived horizon, batch size, LR, weight decay
- optimizer setup
- dataloader initialization
- iteration count and schedulers
- train loop with eval/core/sample/checkpoint branches
- micro-step forward/backward accumulation
- optimizer step
- throughput/MFU logging
- report logging and cleanup
```

How to use it:
- Turn the base-training section into a narrative walkthrough of this structure.

Caveats:
- Avoid line-by-line explanation.
