# Evidence For P1-022

Draft item:
> TODO P1-022: Cite `nanochat/dataloader.py` for BOS-aligned best-fit packing, DDP row-group sharding, pinned CPU staging buffer, persistent GPU buffer, and non-blocking copy.

Source:
- Local file: `nanochat/dataloader.py`, lines 1-17, 25-72, 74-162.
- Local file: `dev/LOG.md`, lines 745-805.

Relevant excerpt:

```text
dataloader.py comments: BOS-aligned best-fit packing; every row starts with BOS; 100% utilization/no padding; documents are packed using best-fit and cropped when needed.

Runtime details:
- parquet row groups start at ddp_rank and step by ddp_world_size
- tokenizer.encode batches document lists
- row_buffer, pinned cpu_buffer, and gpu_buffer are preallocated
- inputs/targets are views into persistent buffers
- gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
```

How to use it:
- Use this as the main evidence for the data feeder throughput section.

Caveats:
- The loader trades off token preservation for dense no-padding training batches.
