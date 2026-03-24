"""
Self-Healing Neural Networks — v2 (The Threshold Problem)
==========================================================
STATUS: BROKEN — documented for educational purposes

FIX FROM v1:
    Real labels used instead of torch.ones()

NEW BUG:
    FIDI threshold = 2.0
    Maximum Z-Score observed = 1.21
    Monitor never fires. Zero healing events.

RESULT: 0/15 batches healed. Model stays dormant.
        Accuracy: 58.9% (same as no healing)

LESSON: Calibrate your drift detector against your actual data.
        A threshold of 2.0 sounds conservative.
        If your Z-Score never reaches 2.0, it is just broken.
        See v3 for proper calibration (threshold=1.0, window=10).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

CFG = {
    "n_train"       : 5000,
    "n_test"        : 1000,
    "fraud_ratio"   : 0.15,
    "drift_strength": 1.8,
    "hidden_dim"    : 64,
    "epochs"        : 20,
    "batch_size"    : 100,
    "n_batches"     : 15,
    "fidi_window"   : 20,    # BUG: too slow
    "fidi_threshold": 2.0,   # BUG: max Z observed was only 1.21
    "conflict_min"  : 15,    # BUG: also too strict
    "heal_steps"    : 5,
    "heal_lr"       : 0.003,
    "alpha"         : 0.70,
    "lambda_lag"    : 0.80,
}


def generate_fraud_data(n_samples, drift=False):
    n_fraud  = int(n_samples * CFG["fraud_ratio"])
    n_normal = n_samples - n_fraud
    v14_n = (np.random.normal(-CFG["drift_strength"], 1.0, n_normal)
             if drift else np.random.normal(0.0, 1.0, n_normal))
    rest_n = np.random.randn(n_normal, 9)
    X_n = np.column_stack([rest_n[:, :7], v14_n, rest_n[:, 7:]])
    v14_f = np.random.normal(-2.5, 0.8, n_fraud)
    rest_f = np.random.randn(n_fraud, 9)
    X_f = np.column_stack([rest_f[:, :7], v14_f, rest_f[:, 7:]])
    X = np.vstack([X_n, X_f]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(np.float32)
    idx = np.random.permutation(len(y))
    return torch.tensor(X[idx]), torch.tensor(y[idx])


class ReflexiveLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim)
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.adapter(x)


class SelfHealingMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.reflexive   = ReflexiveLayer(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.reflex_optimizer = optim.Adam(
            self.reflexive.parameters(), lr=CFG["heal_lr"]
        )

    def forward(self, x):
        return self.output_head(self.reflexive(self.backbone(x))).squeeze()

    def freeze_for_healing(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.output_head.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


def train(model, X, y):
    opt = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(1, CFG["epochs"] + 1):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), CFG["batch_size"]):
            idx = perm[i:i + CFG["batch_size"]]
            opt.zero_grad()
            nn.BCELoss()(model(X[idx]), y[idx]).backward()
            opt.step()
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}/{CFG['epochs']}")
    model.eval()


def evaluate(model, X, y, tag=""):
    model.eval()
    with torch.no_grad():
        probs = model(X)
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean().item()
    print(f"  [{tag}] Accuracy: {acc:.4f}")
    return acc


def main():
    print("=" * 55)
    print("  SELF-HEALING v2 — THE THRESHOLD PROBLEM")
    print("  Expected: 0 healing events, dormant model")
    print("=" * 55)

    X_train, y_train = generate_fraud_data(CFG["n_train"])
    X_drift, y_drift = generate_fraud_data(CFG["n_test"], drift=True)

    model = SelfHealingMLP(10, CFG["hidden_dim"])
    print("\nTraining...")
    train(model, X_train, y_train)

    mu    = X_train[:, 7].mean().item()
    sigma = X_train[:, 7].std().item() + 1e-8
    print(f"\nFIDI calibrated | mu={mu:.3f}  sigma={sigma:.3f}")
    print(f"FIDI threshold   | {CFG['fidi_threshold']}  (max Z will be ~1.21)")
    print(f"This means: healing will never trigger.\n")

    fidi_window  = deque(maxlen=CFG["fidi_window"])
    heal_count   = 0

    print(f"  {'Batch':>5}  {'Z':>6}  {'Fired':>6}  {'Acc':>6}")
    print("  " + "-" * 35)

    for i in range(CFG["n_batches"]):
        idx = np.random.choice(len(X_drift), CFG["batch_size"])
        xb  = X_drift[idx]
        yb  = y_drift[idx]

        with torch.no_grad():
            probs = model(xb)
            acc = ((probs > 0.5).float() == yb).float().mean().item()

        fidi_window.append(float(xb[:, 7].mean()))
        z = abs((np.mean(list(fidi_window)) - mu) / sigma) if len(fidi_window) >= 3 else 0.0
        drifting = z > CFG["fidi_threshold"]  # never True because z stays ~1.21

        n_conf = int(((probs < 0.5) & (xb[:, 7] < -1.5)).sum())
        fired  = drifting or (n_conf >= CFG["conflict_min"])

        if fired:
            heal_count += 1
            sym_lb = (xb[:, 7] < -1.5).float()
            model.freeze_for_healing()
            model.train()
            for _ in range(CFG["heal_steps"]):
                model.reflex_optimizer.zero_grad()
                p = model(xb)
                loss = (CFG["alpha"] * nn.BCELoss()(p, yb) +
                        (1 - CFG["alpha"]) * CFG["lambda_lag"] *
                        nn.BCELoss()(p, sym_lb))
                loss.backward()
                model.reflex_optimizer.step()
            model.unfreeze_all()
            model.eval()

        print(f"  {i+1:>5}  {z:>6.2f}  "
              f"{'YES' if fired else 'no':>6}  {acc:>6.3f}")

    print(f"\n  Healing events: {heal_count}/{CFG['n_batches']}")
    print(f"  Note: FIDI threshold={CFG['fidi_threshold']}, max Z~1.21")
    print(f"  The monitor never crossed its own threshold.\n")
    evaluate(model, X_drift, y_drift, "Final (v2)")
    print("\n  See v3 for threshold=1.0 and window=10 — the fix.")


if __name__ == "__main__":
    main()
