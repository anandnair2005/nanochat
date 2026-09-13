# Evidence For P1-006

Draft item:
> TODO P1-006: Read and quote the relevant stages from `runs/speedrun.sh`.

Source:
- Local file: `runs/speedrun.sh`, lines 3-4, 42-97.
- Local file: `README.md`, lines 48-60.

Relevant excerpt:

```text
runs/speedrun.sh: configured to train your own GPT-2 grade LLM (pretraining + finetuning), designed for a blank 8XH100 GPU node.

Pipeline stages:
- report reset
- dataset download
- tokenizer train/eval
- base pretraining
- base eval
- identity data download
- SFT
- chat eval
- CLI/web chat notes
- report generation
```

How to use it:
- Replace the generic pipeline bullet list with a code-backed description of the exact speedrun stages.

Caveats:
- `runs/speedrun.sh` does not run RL.
