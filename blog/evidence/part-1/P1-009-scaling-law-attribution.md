# Evidence For P1-009

Draft item:
> TODO P1-009: Be careful with attribution: say "appears to be" or cite Karpathy's own wording where available.

Source:
- Local file: `README.md`, lines 6 and 97.
- Local file: `scripts/base_train.py`, lines 261-305.
- Local file: `dev/LOG.md`, lines 223-269 and 502-543.

Relevant excerpt:

```text
README.md: one single complexity dial, `--depth`, automatically determines other hyperparameters so the trained model comes out compute optimal.

base_train.py: comments cite scaling laws, target data:param ratio, Power Lines batch-size law, learning-rate scaling, and T_epoch-style weight-decay scaling.

dev/LOG.md: records empirical sweeps and rationale for selecting stable parameter/token ratios after architecture changes.
```

How to use it:
- Attribute strong claims to the README and code comments. Use `Karpathy's README says...` or `the code comments describe...` rather than unsupported speculation.

Caveats:
- Do not independently validate the scaling laws in this post.
