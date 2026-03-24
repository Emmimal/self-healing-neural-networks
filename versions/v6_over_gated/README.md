# v6: Over-Gated

**Result:** 1/20 healing events. Gate blocked everything.

## What Was Different

Added a strict healing gate requiring:
- n_conflicts >= 5 AND
- RealFraudFrac >= 0.10 (at least 10% real fraud in conflict samples)

Intent: prevent healing on zero-fraud batches that caused v4 and v5 to fail.

## What Happened

The gate was correct in principle. But conflict batches under covariate
shift are structurally normal-dominated. V14 shifts pushed 77% of normal
transactions below the -1.5 threshold. Most conflict batches had 0%
real fraud — not because fraud disappeared, but because the drifted
normals outnumbered real fraud in those samples.

With RealFraudFrac >= 0.10 required, the gate blocked healing on 19 of
20 batches. Only 1 healing event fired.

## Key Output

```
Healing Events          : 1/20
Low-fraud batches skipped: 19
Accuracy recovery       : -5.50%
```

## The Lesson

The fraud gate logic was right. The threshold was wrong.
A gate based on drift detection OR conflict count (without requiring
real fraud) works better. Conditional pos_weight handles the zero-fraud
batch problem more cleanly than a gate.
