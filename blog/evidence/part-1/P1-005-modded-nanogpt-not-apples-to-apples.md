# Evidence For P1-005

Draft item:
> TODO P1-005: Avoid wording that makes NanoChat and `modded-nanogpt` sound like apples-to-apples benchmarks.

Source:
- `README.md`, lines 6, 12, 24, 48, 60, 200.
- `modded-nanogpt` README.

Relevant contrast:

```text
NanoChat README: covers tokenization, pretraining, finetuning, evaluation, inference, and a chat UI; speedrun.sh is the reference way to train a GPT-2 grade model and talk to it.

modded-nanogpt README: target is reaching 3.28 cross-entropy loss on FineWeb validation set using 8xH100.
```

How to use it:
- Explicitly say `modded-nanogpt` is related speedrun lineage, not a direct baseline for NanoChat's full end-to-end run.

Caveats:
- No caveat beyond keeping the comparison bounded.
