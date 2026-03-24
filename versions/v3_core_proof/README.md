# v3: The Core Proof

**Result:** +33.7pp accuracy recovery. Backbone never touched. Recall=14%

## What Was Different

- FIDI threshold dropped to 1.0 (from 2.0)
- FIDI window reduced to 10 (from 20)
- Real labels used (fixed from v1)
- Semi-supervised loss: 0.70*BCE(real) + 0.30*constraint
- No pos_weight

## What Happened

The mechanism worked. 20/20 batches healed. Accuracy recovered from 44.6%
to 78.3%. This is the core proof that the architecture is sound.

But recall collapsed to 14%. The model became extremely conservative.
With no class weighting for fraud, the 70/30 loss ratio caused the
ReflexiveLayer to reduce false positives so aggressively that it also
stopped catching most real fraud.

## Key Output

```
Clean Baseline          : 92.9%
Under Drift (no heal)   : 44.6%
Self-Healed             : 78.3%  (+33.7pp)
Recall                  : 14%
Healing Events          : 20/20
Backbone modified       : NEVER
```

## The Lesson

The architecture works. The loss formula works.
But without fraud class weighting, the model trades recall for precision
too aggressively under class imbalance.
See v4 for the (unsuccessful) attempt to fix recall with fixed pos_weight.
See production_final for the working conditional pos_weight solution.
