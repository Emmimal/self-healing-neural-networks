"""
Self-Healing Neural Networks — v1 (The Label Bug)
==================================================
STATUS: BROKEN — documented for educational purposes

THE BUG:
    y_conflict = torch.ones(X_conflict.shape[0])
    This forces fraud predictions on ALL conflict samples
    regardless of real labels. Causes catastrophic collapse.

RESULT: Accuracy dropped to 17% after healing.
        Healing made things dramatically worse.

LESSON: Real labels are not optional.
        See v2 for the fix.
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
    "n_train"       : 2000,
    "n_test"        : 1000,
    "fraud_ratio"   : 0.15,
    "drift_strength": 3.5,
    "hidden_dim"    : 64,
    "epochs"        : 20,
    "batch_size"    : 100,
    "n_batches"     : 10,
    "fidi_threshold": 2.0,    # BUG: too high, rarely fires
    "conflict_min"  : 5,
    "heal_steps"    : 8,
    "heal_lr"       : 0.005,
    "lambda_lagrange": 2.0,   # BUG: too aggressive
}


def generate_fraud_data(n_samples, drift=False):
    n_fraud  = int(n_samples * CFG["fraud_ratio"])
    n_normal = n_samples - n_fraud
    v14_n = (np.random.normal(-CFG["drift_strength"], 1.2, n_normal)
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
    for _ in range(CFG["epochs"]):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), CFG["batch_size"]):
            idx = perm[i:i + CFG["batch_size"]]
            opt.zero_grad()
            nn.BCELoss()(model(X[idx]), y[idx]).backward()
            opt.step()
    model.eval()


def evaluate(model, X, y, tag=""):
    model.eval()
    with torch.no_grad():
        preds = (model(X) > 0.5).float()
        acc = (preds == y).float().mean().item()
    print(f"  [{tag}] Accuracy: {acc:.4f}")
    return acc


def main():
    print("=" * 50)
    print("  SELF-HEALING v1 — THE LABEL BUG")
    print("  Expected: catastrophic collapse")
    print("=" * 50)

    X_train, y_train = generate_fraud_data(CFG["n_train"])
    X_drift, y_drift = generate_fraud_data(CFG["n_test"], drift=True)

    model = SelfHealingMLP(10, CFG["hidden_dim"])
    train(model, X_train, y_train)

    print("\nPre-drift:")
    evaluate(model, X_train[:500], y_train[:500], "Pre-drift")

    print("\nStreaming drift batches...")
    fidi_window = deque(maxlen=20)
    mu = X_train[:, 7].mean().item()
    sigma = X_train[:, 7].std().item() + 1e-8

    for i in range(CFG["n_batches"]):
        idx = np.random.choice(len(X_drift), CFG["batch_size"])
        xb  = X_drift[idx]
        yb  = y_drift[idx]

        with torch.no_grad():
            probs = model(xb)
            pre_acc = ((probs > 0.5).float() == yb).float().mean().item()

        fidi_window.append(float(xb[:, 7].mean()))
        z = abs((np.mean(list(fidi_window)) - mu) / sigma)
        drifting = z > CFG["fidi_threshold"]

        # Symbolic conflict mask
        mask = (probs < 0.5) & (xb[:, 7] < -1.5)
        n_conf = int(mask.sum())

        should_heal = drifting or (n_conf > CFG["conflict_min"])
        post_acc = pre_acc

        if should_heal:
            # THE BUG IS HERE
            y_conflict = torch.ones(xb.shape[0])  # forces fraud on everything
            sym_labels = (xb[:, 7] < -1.5).float()

            model.freeze_for_healing()
            model.train()
            for _ in range(CFG["heal_steps"]):
                model.reflex_optimizer.zero_grad()
                p = model(xb)
                loss = (CFG["lambda_lagrange"] *
                        nn.BCELoss()(p, y_conflict))  # BUG: no real labels
                loss.backward()
                model.reflex_optimizer.step()
            model.unfreeze_all()
            model.eval()

            with torch.no_grad():
                post_acc = ((model(xb) > 0.5).float() == yb).float().mean().item()

        print(f"  Batch {i+1:2d} | Z={z:.2f} | "
              f"Before={pre_acc:.3f} After={post_acc:.3f} | "
              f"{'HEALED' if should_heal else 'dormant'}")

    print("\nFinal evaluation:")
    evaluate(model, X_drift, y_drift, "Post-drift (v1)")
    print("\n  This is expected to be terrible.")
    print("  See v3 for real labels. See v7 for the working version.")


if __name__ == "__main__":
    main()
