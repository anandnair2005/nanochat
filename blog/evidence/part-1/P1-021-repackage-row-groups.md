# Evidence For P1-021

Draft item:
> TODO P1-021: Cite `dev/repackage_data_reference.py` for the parquet shard and row-group preparation rationale.

Source:
- Local file: `dev/repackage_data_reference.py`, lines 1-12 and 75-102.

Relevant excerpt:

```text
The reference script repackages data into ~100MB zstd-compressed parquet shards with row group size 1024. Its comment says the point is that the dataloader can stream the data and cache it on disk, decreasing training latency.
```

How to use it:
- Support the claim that data layout was prepared for streaming/caching behavior and row-group-level reading.

Caveats:
- This file is documentation/reference and not used at runtime.
