# Evidence For P1-020

Draft item:
> TODO P1-020: Explain the base data path from `nanochat/dataset.py` to `nanochat/dataloader.py`.

Source:
- Local file: `nanochat/dataset.py`, lines 1-8, 20-28, 32-81, 136-160.
- Local file: `nanochat/dataloader.py`, lines 25-72 and 74-162.
- Local file: `runs/speedrun.sh`, lines 51-65 and 69-73.

Relevant excerpt:

```text
dataset.py downloads ClimbMix parquet shards into `$NANOCHAT_BASE_DIR/base_data_climbmix`, always adding the final shard as validation.

dataloader.py lists parquet files, uses all but the last for train and the last for val, shards row groups by DDP rank, tokenizes document batches, and yields fixed-shape input/target tensors.
```

How to use it:
- Explain the data feeder as: Hugging Face hosted parquet shards -> local cache -> DDP row-group iteration -> tokenizer -> BOS-aligned packed tensors -> GPU batch.

Caveats:
- Keep source-downloadable dataset behavior separate from Part 2's GDrive artifact strategy.
