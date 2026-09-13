# Evidence For P1-007

Draft item:
> TODO P1-007: Decide whether RL is part of the specific speedrun path I am discussing or part of NanoChat's broader repo design.

Source:
- Local file: `runs/speedrun.sh`, lines 77-97.
- Local file: `scripts/chat_rl.py`, lines 1-17.
- Local file: `README.md`, lines 166-173.

Relevant excerpt:

```text
runs/speedrun.sh runs SFT and chat eval after base pretraining; it does not invoke scripts.chat_rl.

scripts/chat_rl.py exists and describes reinforcement learning on GSM8K via a simpler GRPO/REINFORCE-like loop.
```

How to use it:
- Say RL belongs to NanoChat's broader repo design, but not to the reference speedrun path analyzed here.

Caveats:
- Part 1 can mention RL briefly as scope, but should not imply it was part of the final speedrun run unless a custom script used it.
