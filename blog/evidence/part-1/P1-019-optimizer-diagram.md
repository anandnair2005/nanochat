# Evidence For P1-019

Draft item:
> TODO P1-019: Consider a small diagram: gradients/shards/optimizer state/all-gather.

Source:
- Local file: `nanochat/optim.py`, lines 311-348 and 514-535.

Suggested diagram content:

```text
local grads on each rank
  -> async reduce_scatter / all_reduce
  -> each rank owns gradient slice or Muon param chunk
  -> local optimizer update with sharded optimizer state
  -> async all_gather updated parameter slices/chunks
  -> all ranks hold updated parameters
```

How to use it:
- Either make a simple text diagram in the blog or save for a figure if the optimizer section feels dense.

Caveats:
- Keep it conceptual; exact tensor shapes vary by parameter group.
