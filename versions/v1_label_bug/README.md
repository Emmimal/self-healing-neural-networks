# v1: The Label Bug

**Result:** Accuracy collapsed to 17%

## What Was Different

First implementation. Conflict detection logic was correct. Loss function was not.

```python
# THE BUG
y_conflict = torch.ones(X_conflict.shape[0])  # forced fraud on ALL conflict samples
```

## What Happened

The model was told that every sample in the conflict region is fraud,
regardless of the real label. Under covariate shift, most conflict samples
are normal transactions that drifted into fraud territory.

Result: the model learned to predict fraud on everything in that region.
Accuracy: 17%.

## The Lesson

Real labels are not optional. torch.ones() as a target is not a symbolic
constraint. It is misinformation.

## Key Output

```
Under Drift — Before Healing  : 0.2005
Under Drift — After Healing   : 0.1680
Recovery Delta                : -0.3150 over baseline
Healing Triggers              : 2/10 batches
```
