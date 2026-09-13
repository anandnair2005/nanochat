# Evidence For P1-004

Draft item:
> TODO P1-004: Verify exact `modded-nanogpt` claim and link README.

Source:
- Public README: `https://github.com/KellerJordan/modded-nanogpt`
- Raw README fetched from: `https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/README.md`

Relevant excerpt:

```text
This repository hosts NanoGPT speedrun, in which contributors search for the fastest algorithm to use 8 NVIDIA H100 GPUs to train a language model that attains 3.28 cross-entropy loss on the FineWeb validation set.

The README states the repository contains a training algorithm that attains the target performance in "Under 75 seconds on 8xH100" and under 400M tokens.
```

How to use it:
- Phrase the claim as a specific benchmark target, not generic "GPT-2 pretraining." A safer phrase is: `modded-nanogpt reports reaching its FineWeb/GPT-2-replication-inspired target in under 75 seconds on 8xH100`.

Caveats:
- This is not apples-to-apples with NanoChat's end-to-end pipeline.
- The target is a specific validation-loss objective, not a full tokenizer-pretrain-SFT-eval-chat pipeline.
