# Evidence For P1-018

Draft item:
> TODO P1-018: Avoid overstating causality unless backed by comments, docs, or measured ablations.

Source:
- Local file: `nanochat/optim.py`, lines 306-348.
- Local file: `scripts/base_train.py`, lines 553-555.
- Current evidence state: no ablation isolating `DistMuonAdamW` contribution for the final run.

How to use it:
- Say the optimizer is designed to reduce/overlap distributed communication and shard state. Do not say it is responsible for exactly X% of MFU or cost savings unless an ablation exists.

Caveats:
- This is a wording guardrail, not a performance measurement.
