# Evidence For P1-024

Draft item:
> TODO P1-024: Keep the SFT dataloader distinct: SFT uses best-fit padding/masking rather than base-pretraining cropping because conversation tokens should not be discarded casually.

Source:
- Local file: `scripts/chat_sft.py`, lines 181-305.

Relevant excerpt:

```text
SFT data generator comment: BOS-aligned dataloader for SFT with bestfit-pad packing. Conversations are packed using best-fit algorithm. When no conversation fits, the row is padded instead of cropped to ensure no tokens are ever discarded. Padding positions have targets masked with -1.
```

How to use it:
- Add a caveat in the data section that base pretraining crops to maintain dense batches, while SFT pads/masks to preserve conversation data.

Caveats:
- This distinction matters because SFT data has structured assistant/user/token masks.
