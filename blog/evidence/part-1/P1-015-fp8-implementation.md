# Evidence For P1-015

Draft item:
> TODO P1-015: Verify where FP8 is enabled and what hardware assumptions are made.

Source:
- Local file: `scripts/base_train.py`, lines 46-48 and 166-194.
- Local file: `nanochat/fp8.py`, lines 1-70 and 195-213.
- Local file: `dev/LOG.md`, lines 317-382.

Relevant excerpt:

```text
base_train.py exposes `--fp8` and `--fp8-recipe`, checks CUDA before enabling FP8, filters Linear modules whose dimensions are divisible by 16 and large enough, then converts eligible Linear layers to Float8 training modules.

fp8.py implements minimal tensorwise dynamic scaling around torch._scaled_mm, using float8_e4m3fn for inputs/weights and float8_e5m2 for gradients.

dev/LOG.md records that FP8 is H100-oriented, tensorwise beat rowwise, small layers must be skipped due to FP8 hardware requirements, and capability-matched speedup was lower than raw matmul speedup.
```

How to use it:
- Explain FP8 as an H100 path, not a portable default for every GPU.

Caveats:
- The current code uses custom `nanochat.fp8`, while the log entry discusses a torchao integration path; use current code for implementation details and the log for experimental background.
