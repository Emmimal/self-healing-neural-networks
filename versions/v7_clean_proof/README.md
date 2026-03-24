# v7: The Clean Proof

**Result:** +33.7pp accuracy recovery. 20/20 batches healed. Backbone untouched.

## What Was Different

- Returned to v3 loss formula (0.70*BCE + 0.30*constraint)
- No pos_weight (removed entirely — root cause of v4/v5)
- No fraud fraction gate (removed — root cause of v6)
- Simple gate: drift detected OR n_conflicts >= 5
- 8 separate PNG plots (not one canvas)
- Full per-batch metrics logged

## What Happened

The v3 core proof reproduced cleanly. This is the research version.
+33.7pp accuracy recovery. 20/20 batches healed. Backbone frozen throughout.

Recall is still 14% — the conservative trade-off from the v3 formula.
The production_final version adds conditional pos_weight and entropy
minimization to lift recall to 34% while maintaining the accuracy recovery.

## Key Output

```
Clean Baseline           : 92.9%
Under Drift (no heal)    : 44.6%
Self-Healed              : 78.3%  (+33.7pp)
Recall                   : 14%
Healing Events           : 20/20
Backbone modified        : NEVER
8 separate plots saved   : YES
```

## The Difference From v3

v3 was an intermediate experiment during iteration.
v7 is the clean, fully documented research version with:
- 8 separate PNG plots
- Full per-batch logging
- Version history plot
- Properly structured for reproducibility

## The Remaining Issue

Recall at 14% is usable in some contexts (where FP cost >> FN cost)
but not in standard fraud detection. See production_final for the
conditional pos_weight solution that lifts recall to 34%.
