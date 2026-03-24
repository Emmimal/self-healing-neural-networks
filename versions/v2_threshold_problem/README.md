# v2: The Threshold Problem

**Result:** 0 healing events across 15 batches

## What Was Different

Fixed the torch.ones() label bug from v1. Used real labels.
Set FIDI threshold to 2.0.

## What Happened

The maximum Z-Score the monitor ever produced was 1.21.
FIDI threshold was 2.0. The monitor never crossed its own alert threshold.

Zero healing events. The model sat completely dormant while accuracy
languished at 58.9%.

## The Bug

```python
# v2 config
"fidi_threshold": 2.0    # PROBLEM: max Z observed was only 1.21
"fidi_window"   : 20     # PROBLEM: too slow to accumulate signal
```

## The Lesson

Calibrate your drift detector against your actual data. A threshold that
sounds conservative (2.0 sigma) can be completely unreachable in practice
depending on the drift magnitude and feature variance.

## Key Output

```
Healing events triggered    : 0/15
Under Drift — No Healing    : 58.9%
Under Drift — Self-Healed   : 58.9%   (same — healing never fired)
```
