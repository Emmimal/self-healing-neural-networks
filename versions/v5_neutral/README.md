# v5: The Neutral Outcome

**Result:** No meaningful recovery. Neutral outcome.

## What Was Different

Reduced pos_weight from 6.0 to 2.0.
Changed semi_sup_alpha from 0.70 to 0.85 (more weight on real labels).
Changed lambda from 0.80 to 0.40 (weaker constraint).

## What Happened

The structural problem from v4 was still present. Any fixed pos_weight
on zero-fraud batches still biases predictions upward, just less
aggressively.

Result: slight over-prediction of fraud on all batches, partially
cancelling the accuracy recovery from the constraint term.
Net: neutral.

## Key Output

```
Accuracy recovery    : -1.60%
Recall               : 81%
F1                   : 0.300
Healing Events       : 19/20
```

## The Lesson

The problem is not the magnitude of pos_weight. The problem is applying
ANY fixed pos_weight to batches with zero real fraud.
The solution is conditional pos_weight based on actual fraud fraction.
