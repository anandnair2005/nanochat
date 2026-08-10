# Evidence For P1-014

Draft item:
> TODO P1-014: Verify the exact attention implementation and fallback behavior in `nanochat/model.py` or related files.

Source:
- Local file: `nanochat/gpt.py`, lines 65-126.
- Local file: `nanochat/flash_attention.py`, lines 1-15, 20-63, 107-128.
- Local file: `scripts/base_train.py`, lines 103-118.
- Local file: `dev/LOG.md`, lines 645-686.

Relevant excerpt:

```text
The model code imports `flash_attn` from `nanochat.flash_attention` and calls `flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)` during training.

`flash_attention.py` provides a unified FA3/SDPA interface. FA3 is loaded only on Hopper/sm90 and only used for bf16 compute dtype. Otherwise it falls back to PyTorch SDPA and transposes between FA3 layout `(B, T, H, D)` and SDPA layout `(B, H, T, D)`.

base_train.py prints whether FA3 or SDPA is used and warns that SDPA plus sliding-window attention is inefficient.
```

How to use it:
- Correct the draft to reference `nanochat/gpt.py` and `nanochat/flash_attention.py`, not `nanochat/model.py`.

Caveats:
- FA3 is hardware/dtype-dependent; do not imply all devices use it.
