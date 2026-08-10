# Evidence For P1-030

Draft item:
> TODO P1-030: Inspect `scripts/chat_sft.py`, RL scripts, and `runs/speedrun.sh` to state exactly what runs in the speedrun path.

Source:
- Local file: `runs/speedrun.sh`, lines 77-97.
- Local file: `scripts/chat_sft.py`, lines 1-10, 95-134, 163-180.
- Local file: `scripts/chat_rl.py`, lines 1-17.

Relevant excerpt:

```text
The reference speedrun runs chat_sft and chat_eval after base pretraining/eval. It does not invoke chat_rl.

chat_sft loads the base model, inherits relevant pretraining hyperparameters by default, compiles the model, reuses optimizer setup, and trains on a mixture including SmolTalk, identity conversations, MMLU, GSM8K, SimpleSpelling, and SpellingBee.

chat_rl exists as a GSM8K RL script with a simpler GRPO/REINFORCE-like loop, but is outside speedrun.sh.
```

How to use it:
- State that SFT is part of the reference speedrun; RL is part of the broader repo, not this specific speedrun path.

Caveats:
- If Part 2's custom Vast path ever ran RL, cite that separately. Current `runs/speedrun.sh` does not.
