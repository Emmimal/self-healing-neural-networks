# v4: The Overshoot

**Result:** Accuracy crashed to 38%. Worse than no healing.

## What Was Different

Added pos_weight=6.0 to BCEWithLogitsLoss during healing.
Intent: recover the recall that collapsed in v3.

## What Happened

Most conflict batches contain 0% real fraud. Under covariate shift, the
conflict region is filled with normal transactions that drifted into fraud
territory — not actual fraud.

Applying pos_weight=6x on batches with zero real fraud teaches the model:
"everything in the conflict region is fraud, and fraud is 6x more important."

The ReflexiveLayer learned to push predictions aggressively toward fraud
for any sample near the symbolic rule boundary. False positives exploded.
Accuracy: 38%.

## Key Output

```
Accuracy recovery    : -6.50%
Recall               : 88%  (was 14% in v3)
FP                   : 601  (was 88 in v3)
Accuracy             : 38%  (was 78% in v3)
```

## The Lesson

Fixed pos_weight on zero-fraud batches is the wrong solution. The weight
needs to be conditional on actual fraud fraction in the batch.
See production_final for conditional pos_weight implementation.
