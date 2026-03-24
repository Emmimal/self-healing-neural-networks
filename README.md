# self-healing-neural-networks
A production-ready PyTorch system that detects model drift, injects targeted weight updates via a frozen-backbone Reflexive Layer, and recovers performance autonomously — without retraining or downtime.

# Self-Healing Neural Networks in PyTorch

> Fix Model Drift in Real Time Without Retraining

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A production-ready PyTorch system that detects model drift, injects targeted
weight updates, and recovers performance autonomously — without retraining or
downtime.

**Article:** [Self-Healing Neural Networks in PyTorch: Fix Model Drift in Real Time Without Retraining](https://towardsdatascience.com) *(link once published)*

---

## What This Does

When your deployed model drifts, the standard answer is to retrain it. But
retraining takes time. Your broken model keeps running while the pipeline
executes.

This system gives your model a different option: adapt in real time.

A small **ReflexiveLayer** sits between the backbone and the output head. It
is the only component that ever changes after deployment. When drift is
detected via a FIDI Z-Score monitor, and when the model's predictions conflict
with a symbolic domain rule, the ReflexiveLayer runs a local optimization loop
guided by real labels and a Lagrangian constraint.

The backbone never moves. The output head never moves. Only the adapter adapts.

### Results (from the experiment documented in the article)

| Stage | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Clean baseline | 92.9% | 0.784 | 0.727 | 0.754 |
| Under drift, no healing | 44.6% | 0.194 | 0.853 | 0.316 |
| Frozen shadow model | 44.6% | 0.194 | 0.853 | 0.316 |
| **Production self-healed** | **72.4%** | **0.224** | **0.340** | **0.270** |

- **+27.8pp accuracy recovery**
- **67% false positive reduction** (532 to 177)
- **Backbone weights: never modified**

---

## Repository Structure

```
self-healing-neural-networks/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── versions/                          # All 7 iterations documented
│   ├── v1_label_bug/
│   │   ├── app.py                     # The torch.ones() disaster
│   │   └── README.md                  # What went wrong and why
│   │
│   ├── v2_threshold_problem/
│   │   ├── app.py                     # FIDI threshold too high, 0 events
│   │   └── README.md
│   │
│   ├── v3_core_proof/
│   │   ├── app.py                     # +33.7pp recovery, recall=14%
│   │   └── README.md
│   │
│   ├── v4_overshoot/
│   │   ├── app.py                     # pos_weight=6x destroyed accuracy
│   │   └── README.md
│   │
│   ├── v5_neutral/
│   │   ├── app.py                     # pos_weight=2x, neutral outcome
│   │   └── README.md
│   │
│   ├── v6_over_gated/
│   │   ├── app.py                     # Fraud gate blocked everything
│   │   └── README.md
│   │
│   └── v7_clean_proof/
│       ├── app.py                     # The working research version
│       └── README.md
│
├── production/
│   └── self_healing_production_final.py  # Full production system
│
├── plots/                             # Output plots (generated on run)
│   └── .gitkeep
│
├── monitoring_export/                 # CSV + JSON monitoring output
│   └── .gitkeep
│
└── results/
    └── version_comparison.md          # All version results in one table
```

---

## Quick Start

### 1. Clone and install

```bash
git clone [https://github.com/Emmimal/self-healing-neural-networks.git](https://github.com/Emmimal/self-healing-neural-networks.git)
cd self-healing-neural-networks
pip install -r requirements.txt
```

### 2. Run the production system

```bash
python production/self_healing_production_final.py
```

This will:
- Generate synthetic fraud data
- Train the SelfHealingMLP and BaselineMLP on clean data
- Simulate severe covariate drift
- Run the full production streaming loop (25 batches)
- Export monitoring data to `monitoring_export/`
- Save 8 separate plots to the current directory

### 3. Run a specific version

```bash
# See the core proof (v7 — the working research version)
python versions/v7_clean_proof/app.py

# See the original failure (v1 — torch.ones() disaster)
python versions/v1_label_bug/app.py
```

---

## Architecture

```
Input
  |
  v
Backbone (frozen after training)
  - Linear(10, 64) + ReLU
  - Linear(64, 64) + ReLU
  |
  v
[ReflexiveLayer]  <-- only this adapts
  - Residual: x + scale * adapter(x)
  - adapter: Linear(64,64) -> Tanh -> Linear(64,64)
  - scale: learnable scalar, init=0.1
  |
  v
Output Head (frozen after training)
  - Linear(64, 1) + Sigmoid
```

### Production Components

```
AsyncHealingEngine     threading.Thread + RLock
                       Inference never blocked by healing

ModelRegistry          Versioned ReflexiveLayer snapshots
                       Rollback targets best post-heal F1

HealthMonitor          Rolling F1/accuracy window
                       Detects degradation automatically

RollbackEngine         Auto-reverts on degradation
                       Targets best post-heal snapshot, not clean weights

MonitoringExporter     JSON + CSV + threshold config
                       Grafana-ready output

FIDIMonitor            Rolling Z-Score drift detection
                       Fires at batch 3, stays active

SymbolicRuleEngine     V14 < -1.5 = Fraud
                       Conflict detection, Type-A only
```

---

## The Seven Iterations

Every version is preserved and documented. This is the honest research
history.

| Version | Key Change | Result | What Broke |
|---|---|---|---|
| v1 | First implementation | Acc: 17% | torch.ones() as conflict target label |
| v2 | Fixed labels | 0 healing events | FIDI threshold=2.0, max Z was 1.21 |
| v3 | Recalibrated thresholds | +33.7pp acc | Recall collapsed to 14%, no class weighting |
| v4 | pos_weight=6x | Acc crashed to 38% | 6x on zero-fraud batches = predict everything fraud |
| v5 | pos_weight=2x | Neutral outcome | Same structural problem, smaller scale |
| v6 | Added fraud gate | 1/20 healed | Gate too strict, conflict batches structurally normal-dominated |
| **v7** | **Conditional pos_weight** | **+33.7pp acc** | Working. Recall still 14% |
| **Production Final** | **Full production stack** | **+27.8pp acc, recall=34%** | Working. Balanced. |

---

## Production System Features

### Async Healing

```python
async_engine.request_heal(xb, yb, symbolic, i+1, fraud_frac=frac)
# returns immediately — inference continues
```

Healing runs in a background thread. Inference never blocks.

### Smart Rollback

```python
# Rollback targets best POST-HEAL snapshot, not clean weights
snap = self.registry.rollback(model)
```

Rolling back to clean weights on drifted data recreates the original problem.
The registry tracks post-heal snapshots separately.

### Tunable Thresholds

```python
PROD_CFG = {
    "rollback_f1_drop"  : 0.08,   # 0.05=strict  0.10=lenient
    "rollback_acc_drop" : 0.10,
    "health_window"     : 5,
}
```

Change risk tolerance without touching model code.

### Shadow Comparison

A frozen model runs alongside the self-healing model every batch. Lift is
tracked per batch. Exported to CSV.

### Monitoring Export

```
monitoring_export/
  metrics.csv            # 25 rows, one per batch
  events.json            # every heal and rollback event
  threshold_config.json  # current threshold settings
```

---

## Healing Loss Function

```python
# Semi-supervised loss — proven in v3, refined in production
loss = 0.70 * BCE(probs, real_labels)
     + 0.30 * 0.80 * BCE(probs, sym_labels)

# Conditional fraud weight
if fraud_frac >= 0.10:
    fraud_weight = min(2.0 + (fraud_frac - 0.10) * 10.0, 4.0)
else:
    fraud_weight = 1.0   # zero bias on normal-dominated batches

# Entropy minimization (sharpens predictions)
entropy = -(probs * log(probs) + (1-probs) * log(1-probs)).mean()
loss = loss + 0.03 * entropy
```

The conditional fraud weight is the detail that took four failed versions to
figure out. Under covariate shift, most conflict batches contain zero real
fraud. A fixed pos_weight on zero-fraud batches teaches the model that
everything is fraud.

---

## Output Plots

Running the production system generates 8 separate plots:

| File | Shows |
|---|---|
| `prod_plot1_accuracy.png` | Batch accuracy: baseline vs shadow vs self-healed |
| `prod_plot2_shadow_lift.png` | Per-batch lift over frozen shadow model |
| `prod_plot3_fidi.png` | FIDI Z-Score drift detection timeline |
| `prod_plot4_system_state.png` | State machine: HEALTHY / DRIFTING / ROLLED_BACK |
| `prod_plot5_rollback_f1.png` | F1 with rollback events annotated |
| `prod_plot6_registry.png` | Model registry: versioned snapshots |
| `prod_plot7_metrics.png` | Full 4-metric comparison (all stages) |
| `prod_plot8_monitoring_dashboard.png` | Grafana-ready dashboard preview |

---

## The Honest Trade-off

Recall lands at 34% in the production final. The drifted baseline has 85%.

The baseline achieves 85% recall by flagging 532 out of every 1,000
transactions as fraud. Most are legitimate customers being blocked. The healed
model flags 177. It catches fewer fraudsters but causes far less collateral
damage.

The right configuration depends on your cost structure. The architecture
supports both directions through the tunable pos_weight and constraint
coefficients in PROD_CFG.

---

## Related Work

This experiment connects to the neuro-symbolic training literature:

- [Hybrid Neuro-Symbolic Fraud Detection](https://towardsdatascience.com/hybrid-neuro-symbolic-fraud-detection-guiding-neural-networks-with-domain-rules/) — encoding analyst rules into training loss
- [How a Neural Network Learned Its Own Fraud Rules](https://towardsdatascience.com/how-a-neural-network-learned-its-own-fraud-rules-a-neuro-symbolic-ai-experiment/) — differentiable rule induction during training

The key difference: those articles use symbolic rules during training. This
experiment uses them at inference time, as a real-time correction signal after
deployment. No retraining pipeline required.

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

Python 3.8 or higher.

---

## Citation

If you use this code or the approach described here, please cite:

```
@article{self_healing_nn_2026,
  title   = {Self-Healing Neural Networks in PyTorch:
             Fix Model Drift in Real Time Without Retraining},
  author  = {[Emmimal P Alexander]},
  website = {https://emitechlogic.com/}
  journal = {Towards Data Science},
  year    = {2026},
  url     = {https://towardsdatascience.com/[article-url]}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Disclosure

This repository is based on independent experiments using a fully synthetic
dataset generated in code to simulate credit card fraud detection behavior.
No real transaction data was used. The synthetic data design was inspired by
published feature characteristics of the ULB Credit Card Fraud dataset
(Dal Pozzolo et al., 2015), but no actual data from that dataset was loaded
or used.
