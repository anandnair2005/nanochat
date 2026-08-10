# Evidence For P1-012

Draft item:
> TODO P1-012: Cite `nanochat.common.compute_init()` or equivalent distributed setup code.

Source:
- Local file: `nanochat/common.py`, lines 173-208.

Relevant excerpt:

```text
compute_init() autodetects or accepts device type, sets CUDA device for `LOCAL_RANK`, initializes `dist.init_process_group(backend="nccl", device_id=device)` when DDP is requested on CUDA, and returns ddp rank/world-size/device metadata.
```

How to use it:
- Cite this in the single-node distributed training section.

Caveats:
- The process group is initialized, but the model is not wrapped in PyTorch DDP; synchronization is handled by the distributed optimizer.
