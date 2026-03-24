"""
============================================================
  Self-Healing Neural Networks v6.0  — THE PROOF RUN
  Fixing Model Drift Without Retraining (PyTorch)
============================================================
  WHAT v6 FIXES:
  ① Loss reverts to v3 ratio: 0.70*BCE + 0.30*constraint
  ② pos_weight REMOVED entirely from healing
  ③ Strict healing gate:
       drift_detected OR n_conflicts >= 8
       AND RealFraudFrac >= 0.10 (skip pure-normal conflict batches)
       AND n_conflicts >= 5
  ④ Dynamic constraint multiplier:
       FraudFrac < 0.15  → constraint_mult = 1.5 (symbolic pulls harder)
       FraudFrac >= 0.15 → constraint_mult = 0.8 (real labels lead)
  ⑤ Adaptive steps/lr by conflict size:
       n_conflicts < 15  → steps=3, lr=0.002
       n_conflicts >= 15 → steps=6, lr=0.005
  ⑥ Each plot saved as a SEPARATE PNG file
============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import copy
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# HYPERPARAMETERS
# ============================================================
CFG = {
    "n_train"            : 5000,
    "n_test"             : 1000,
    "fraud_ratio"        : 0.15,
    "drift_strength"     : 2.2,
    "hidden_dim"         : 64,
    "epochs"             : 20,
    "batch_size"         : 100,
    "n_batches"          : 20,
    "fidi_window"        : 10,
    "fidi_threshold"     : 1.0,
    "conflict_prob_thr"  : 0.30,
    # Strict healing gate
    "conflict_min"       : 5,      # absolute minimum
    "conflict_for_drift" : 8,      # needed when drift detected
    "min_fraud_frac"     : 0.10,   # skip batches with <10% fraud in conflicts
    # Loss (back to v3 ratio, no pos_weight)
    "semi_sup_alpha"     : 0.70,
    "lambda_lagrange"    : 0.80,
    # Dynamic params
    "steps_small"        : 3,
    "steps_large"        : 6,
    "lr_small"           : 0.002,
    "lr_large"           : 0.005,
    "conflict_size_thr"  : 15,     # boundary between small/large
    # Dynamic constraint multiplier
    "const_mult_low"     : 1.5,    # FraudFrac < 0.15
    "const_mult_high"    : 0.8,    # FraudFrac >= 0.15
}


# ============================================================
# 1. DATA
# ============================================================

def generate_fraud_data(n_samples, drift=False,
                        drift_strength=None,
                        fraud_ratio=CFG["fraud_ratio"]):
    if drift_strength is None:
        drift_strength = CFG["drift_strength"]

    n_fraud  = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    v14_normal = (np.random.normal(-drift_strength, 1.0, n_normal)
                  if drift else np.random.normal(0.0, 1.0, n_normal))

    rest_n = np.random.randn(n_normal, 9)
    X_n    = np.column_stack([rest_n[:, :7], v14_normal, rest_n[:, 7:]])

    v14_fraud = np.random.normal(-2.5, 0.8, n_fraud)
    rest_f    = np.random.randn(n_fraud, 9)
    X_f       = np.column_stack([rest_f[:, :7], v14_fraud, rest_f[:, 7:]])

    X   = np.vstack([X_n, X_f]).astype(np.float32)
    y   = np.concatenate([np.zeros(n_normal),
                           np.ones(n_fraud)]).astype(np.float32)
    idx = np.random.permutation(len(y))
    return torch.tensor(X[idx]), torch.tensor(y[idx])


# ============================================================
# 2. REFLEXIVE LAYER
# ============================================================

class ReflexiveLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(),
            nn.Linear(dim, dim)
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.adapter(x)


# ============================================================
# 3. SELF-HEALING MLP
# ============================================================

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
        # Optimizer rebuilt dynamically per heal call (adaptive lr)
        self._heal_lr = CFG["lr_small"]

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

    def get_reflex_optimizer(self, lr):
        return optim.Adam(self.reflexive.parameters(), lr=lr)


# ============================================================
# 4. BASELINE MLP
# ============================================================

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()


# ============================================================
# 5. FIDI MONITOR
# ============================================================

class FIDIMonitor:
    def __init__(self):
        self.window    = deque(maxlen=CFG["fidi_window"])
        self.threshold = CFG["fidi_threshold"]
        self.mu        = None
        self.sigma     = None
        self.history   = []

    def calibrate(self, X_clean):
        v14        = X_clean[:, 7].numpy()
        self.mu    = float(np.mean(v14))
        self.sigma = float(np.std(v14)) + 1e-8
        print(f"    FIDI calibrated | μ={self.mu:.3f}  "
              f"σ={self.sigma:.3f}  threshold={self.threshold}")

    def check(self, X_batch):
        self.window.append(float(X_batch[:, 7].mean()))
        z = 0.0
        if len(self.window) >= 3:
            z = abs((np.mean(list(self.window)) - self.mu) / self.sigma)
        self.history.append(z)
        return z > self.threshold, z


# ============================================================
# 6. SYMBOLIC RULE ENGINE
# ============================================================

class SymbolicRuleEngine:
    THRESHOLD = -1.5

    def predict(self, X):
        return (X[:, 7] < self.THRESHOLD).float()

    def type_a_mask(self, X, mlp_probs):
        return (mlp_probs < CFG["conflict_prob_thr"]) & \
               (X[:, 7] < self.THRESHOLD)

    def n_conflicts(self, X, mlp_probs):
        return int(self.type_a_mask(X, mlp_probs).sum())


# ============================================================
# 7. HEALING LOSS  (v3 ratio, no pos_weight, dynamic constraint)
# ============================================================

def healing_loss(probs, real_labels, symbolic_labels,
                 fraud_frac):
    """
    Loss = alpha * BCE(real) + (1-alpha) * lambda * mult * constraint
    mult is dynamic:
        fraud_frac < 0.15 → 1.5 (symbolic pulls harder, less real signal)
        fraud_frac >= 0.15 → 0.8 (real labels are trustworthy)
    No pos_weight — prevents fraud bias on 0%-fraud conflict batches.
    """
    alpha = CFG["semi_sup_alpha"]
    lam   = CFG["lambda_lagrange"]
    mult  = (CFG["const_mult_low"]
             if fraud_frac < 0.15
             else CFG["const_mult_high"])

    bce        = nn.BCELoss()
    real_loss  = bce(probs, real_labels)
    constraint = bce(probs, symbolic_labels)
    total      = alpha * real_loss + (1 - alpha) * lam * mult * constraint
    return total, real_loss.item(), constraint.item(), mult


# ============================================================
# 8. STRICT HEALING GATE
# ============================================================

def should_heal(drifting, n_conf, fraud_frac):
    """
    Gate logic:
      Pass if:
        (drift detected OR enough conflicts)
        AND fraud fraction >= min_fraud_frac
        AND absolute conflict count >= conflict_min
    This prevents healing on pure-normal conflict batches.
    """
    signal    = drifting or (n_conf >= CFG["conflict_for_drift"])
    has_fraud = fraud_frac >= CFG["min_fraud_frac"]
    enough    = n_conf >= CFG["conflict_min"]
    return signal and has_fraud and enough


# ============================================================
# 9. ADAPTIVE HEALING LOOP
# ============================================================

def heal(model, X_batch, y_batch, symbolic_engine, fraud_frac):
    """
    Steps and lr are adaptive based on conflict batch size.
    Only Type A conflicts used for healing.
    """
    with torch.no_grad():
        probs_now = model(X_batch)
    mask = symbolic_engine.type_a_mask(X_batch, probs_now)

    X_heal = X_batch[mask] if mask.sum() >= 2 else X_batch
    y_heal = y_batch[mask] if mask.sum() >= 2 else y_batch
    sym_lb = symbolic_engine.predict(X_heal)

    # Adaptive steps & lr
    n_heal  = int(mask.sum())
    steps   = (CFG["steps_small"]
               if n_heal < CFG["conflict_size_thr"]
               else CFG["steps_large"])
    lr      = (CFG["lr_small"]
               if n_heal < CFG["conflict_size_thr"]
               else CFG["lr_large"])

    optimizer = model.get_reflex_optimizer(lr)

    model.freeze_for_healing()
    model.train()

    traj_total, traj_real, traj_const, mults = [], [], [], []

    for _ in range(steps):
        optimizer.zero_grad()
        p                  = model(X_heal)
        loss, r, c, mult   = healing_loss(p, y_heal, sym_lb, fraud_frac)
        loss.backward()
        optimizer.step()
        traj_total.append(loss.item())
        traj_real.append(r)
        traj_const.append(c)
        mults.append(mult)

    model.unfreeze_all()
    model.eval()
    return traj_total, traj_real, traj_const, mults[0], steps, lr


# ============================================================
# 10. TRAINING
# ============================================================

def train(model, X, y, label):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    print(f"\n    Training {label}...")

    for epoch in range(1, CFG["epochs"] + 1):
        perm  = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), CFG["batch_size"]):
            idx = perm[i:i + CFG["batch_size"]]
            optimizer.zero_grad()
            loss = criterion(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 5 == 0:
            print(f"      Epoch [{epoch:2d}/{CFG['epochs']}]  "
                  f"Loss: {total:.4f}")
    model.eval()


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate(model, X, y, tag=""):
    model.eval()
    with torch.no_grad():
        probs = model(X)
        preds = (probs > 0.5).float()
        acc   = (preds == y).float().mean().item()

    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    print(f"\n  ── {tag} ──")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1        : {f1:.4f}")
    print(f"     TP={int(tp)}  TN={int(tn)}  FP={int(fp)}  FN={int(fn)}")
    return dict(acc=acc, prec=prec, rec=rec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


# ============================================================
# 12. STREAMING EXPERIMENT
# ============================================================

def stream_experiment(heal_model, base_model,
                      X_drift, y_drift, fidi, symbolic):
    BS      = CFG["batch_size"]
    n_batch = CFG["n_batches"]

    log = dict(
        z_scores=[], conflicts=[], baseline_acc=[],
        before_acc=[], after_acc=[], healed=[],
        bce_real=[], bce_const=[], fraud_frac=[],
        const_mult=[], steps_used=[], skipped_reason=[]
    )

    print(f"\n  Streaming {n_batch} batches (size={BS})...\n")
    print(f"  {'B':>3}  {'Z':>5}  {'Status':>10}  "
          f"{'C-A':>4}  {'Fraud%':>7}  {'Event':>12}  "
          f"{'Base':>6}  {'Before':>7}  {'After':>7}  {'Δ':>6}")
    print("  " + "─" * 88)

    for i in range(n_batch):
        idx = np.random.choice(len(X_drift), BS, replace=True)
        xb  = X_drift[idx]
        yb  = y_drift[idx]

        with torch.no_grad():
            base_acc  = ((base_model(xb) > 0.5).float() == yb
                         ).float().mean().item()
            probs_pre = heal_model(xb)
            pre_acc   = ((probs_pre > 0.5).float() == yb
                         ).float().mean().item()

        drifting, z = fidi.check(xb)
        status      = "DRIFT  🔴" if drifting else "stable 🟢"
        n_conf      = symbolic.n_conflicts(xb, probs_pre)

        # Fraud fraction in conflict samples
        mask  = symbolic.type_a_mask(xb, probs_pre)
        frac  = yb[mask].mean().item() if mask.sum() > 0 else 0.0

        # ── Strict gate ───────────────────────────────────
        heal_flag   = should_heal(drifting, n_conf, frac)
        event       = "dormant"
        skip_reason = ""
        post_acc    = pre_acc
        delta       = 0.0
        bce_r_avg   = 0.0
        bce_c_avg   = 0.0
        c_mult      = 0.0
        steps_used  = 0

        if not heal_flag:
            # Log why healing was skipped
            if frac < CFG["min_fraud_frac"] and n_conf >= CFG["conflict_min"]:
                skip_reason = "low-fraud skip"
                event       = "⛔ skipped"
            elif n_conf < CFG["conflict_min"]:
                skip_reason = "few conflicts"
                event       = "dormant"
        else:
            traj, tr, tc, c_mult, steps_used, lr_used = heal(
                heal_model, xb, yb, symbolic, frac
            )
            with torch.no_grad():
                post_acc = ((heal_model(xb) > 0.5).float() == yb
                            ).float().mean().item()
            event     = "⚡ HEALED"
            delta     = post_acc - pre_acc
            bce_r_avg = np.mean(tr)
            bce_c_avg = np.mean(tc)

        delta_str = f"{delta:+.3f}" if event == "⚡ HEALED" else "    —  "

        print(f"  {i+1:>3}  {z:>5.2f}  {status:>10}  "
              f"{n_conf:>4}  {frac*100:>6.1f}%  {event:>12}  "
              f"{base_acc:>6.3f}  {pre_acc:>7.3f}  "
              f"{post_acc:>7.3f}  {delta_str:>6}")

        if event == "⚡ HEALED":
            print(f"       └─ steps={steps_used}  "
                  f"const_mult={c_mult:.1f}  "
                  f"BCE_real={bce_r_avg:.4f}  "
                  f"BCE_constraint={bce_c_avg:.4f}  "
                  f"FraudFrac={frac*100:.1f}%")
        elif event == "⛔ skipped":
            print(f"       └─ Skipped: {skip_reason}  "
                  f"(conflicts={n_conf}, fraud={frac*100:.1f}%)")

        log["z_scores"].append(z)
        log["conflicts"].append(n_conf)
        log["baseline_acc"].append(base_acc)
        log["before_acc"].append(pre_acc)
        log["after_acc"].append(post_acc)
        log["healed"].append(event == "⚡ HEALED")
        log["bce_real"].append(bce_r_avg)
        log["bce_const"].append(bce_c_avg)
        log["fraud_frac"].append(frac)
        log["const_mult"].append(c_mult)
        log["steps_used"].append(steps_used)
        log["skipped_reason"].append(skip_reason)

    return log


# ============================================================
# 13. SEPARATE PLOT FUNCTIONS
# ============================================================

BG  = "#0d1117"
PAN = "#161b22"
G   = "#00e676"
R   = "#ff1744"
B   = "#40c4ff"
Y   = "#ffd740"
TXT = "#e6edf3"
GRY = "#30363d"


def ax_style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(PAN)
    ax.set_title(title, color=TXT, fontsize=12,
                 fontweight="bold", pad=10)
    ax.tick_params(colors=TXT, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=TXT)
    if ylabel:
        ax.set_ylabel(ylabel, color=TXT)
    for s in ax.spines.values():
        s.set_edgecolor(GRY)


def save_fig(fig, filename):
    fig.patch.set_facecolor(BG)
    fig.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    📊 Saved → {filename}")


# ── Plot 1: Batch Accuracy Comparison (hero) ───────────────
def plot_accuracy(log, clean_acc, fname="plot1_accuracy.png"):
    fig, ax = plt.subplots(figsize=(13, 5))
    batches = range(1, len(log["z_scores"]) + 1)

    ax.plot(batches, log["baseline_acc"],
            color=R, lw=2, label="Baseline — No Healing", alpha=0.85)
    ax.plot(batches, log["before_acc"],
            color=Y, lw=1.5, ls="--",
            label="Self-Healing — Pre-Heal", alpha=0.70)
    ax.plot(batches, log["after_acc"],
            color=G, lw=2.5,
            label="Self-Healing — Post-Heal", alpha=0.95)

    for i, h in enumerate(log["healed"]):
        if h:
            ax.axvline(x=i+1, color=B, alpha=0.25, lw=1.0, ls=":")

    ax.axhline(y=clean_acc, color=G, ls="--",
               alpha=0.2, lw=1, label=f"Clean Baseline ({clean_acc*100:.1f}%)")
    ax.set_ylim(0, 1.08)
    ax.legend(facecolor=PAN, labelcolor=TXT, fontsize=9)
    ax_style(ax, "Batch Accuracy: Baseline vs Self-Healing Under Drift",
             xlabel="Batch", ylabel="Accuracy")
    save_fig(fig, fname)


# ── Plot 2: FIDI Z-Score ───────────────────────────────────
def plot_fidi(log, fname="plot2_fidi_zscore.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    batches = range(1, len(log["z_scores"]) + 1)

    ax.plot(batches, log["z_scores"], color=B, lw=2, label="|Z-Score|")
    ax.fill_between(
        batches, log["z_scores"], CFG["fidi_threshold"],
        where=[z > CFG["fidi_threshold"] for z in log["z_scores"]],
        color=R, alpha=0.2, label="Drift Zone"
    )
    ax.axhline(y=CFG["fidi_threshold"], color=Y, lw=1.3, ls="--",
               label=f"Threshold ({CFG['fidi_threshold']})")
    ax.set_ylim(0)
    ax.legend(facecolor=PAN, labelcolor=TXT, fontsize=9)
    ax_style(ax, "FIDI Z-Score Monitor (V14 Distribution Shift)",
             xlabel="Batch", ylabel="|Z-Score|")
    save_fig(fig, fname)


# ── Plot 3: Conflict count per batch ──────────────────────
def plot_conflicts(log, fname="plot3_conflicts.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    batches = list(range(1, len(log["conflicts"]) + 1))
    colors  = [R if c >= CFG["conflict_min"] else G
               for c in log["conflicts"]]

    ax.bar(batches, log["conflicts"], color=colors, alpha=0.85, width=0.7)
    ax.axhline(y=CFG["conflict_min"], color=Y, lw=1.3, ls="--",
               label=f"Min threshold ({CFG['conflict_min']})")
    ax.axhline(y=CFG["conflict_for_drift"], color=R, lw=1.3, ls=":",
               label=f"Drift threshold ({CFG['conflict_for_drift']})")
    ax.legend(facecolor=PAN, labelcolor=TXT, fontsize=9)
    ax_style(ax, "Type-A Conflicts per Batch (MLP=Normal vs Rule=Fraud)",
             xlabel="Batch", ylabel="Conflict Count")
    save_fig(fig, fname)


# ── Plot 4: Fraud fraction in conflict batches ────────────
def plot_fraud_frac(log, fname="plot4_fraud_fraction.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    batches = list(range(1, len(log["fraud_frac"]) + 1))
    colors  = [G if f >= CFG["min_fraud_frac"] else R
               for f in log["fraud_frac"]]

    ax.bar(batches, [f * 100 for f in log["fraud_frac"]],
           color=colors, alpha=0.85, width=0.7)
    ax.axhline(y=CFG["min_fraud_frac"] * 100, color=Y,
               lw=1.3, ls="--",
               label=f"Min fraud gate ({CFG['min_fraud_frac']*100:.0f}%)")
    ax.legend(facecolor=PAN, labelcolor=TXT, fontsize=9)
    ax_style(ax, "Real Fraud Fraction in Conflict Samples (Healing Gate)",
             xlabel="Batch", ylabel="Fraud % in Conflicts")
    save_fig(fig, fname)


# ── Plot 5: Pre vs Post heal per healed batch ─────────────
def plot_pre_post(log, fname="plot5_pre_post_heal.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    healed_i = [i for i, h in enumerate(log["healed"]) if h]

    if not healed_i:
        ax.text(0.5, 0.5, "No healing events triggered",
                transform=ax.transAxes, ha="center",
                va="center", color=TXT, fontsize=13)
    else:
        pre_v  = [log["before_acc"][i] for i in healed_i]
        post_v = [log["after_acc"][i]  for i in healed_i]
        xh     = np.arange(len(healed_i))

        ax.bar(xh - 0.2, [v*100 for v in pre_v],
               width=0.38, color=R, alpha=0.85, label="Pre-Heal")
        ax.bar(xh + 0.2, [v*100 for v in post_v],
               width=0.38, color=G, alpha=0.85, label="Post-Heal")

        for xi, (pre, post) in enumerate(zip(pre_v, post_v)):
            d = (post - pre) * 100
            ax.text(xi, max(pre, post)*100 + 1.5,
                    f"{'+' if d >= 0 else ''}{d:.0f}%",
                    ha="center",
                    color=G if d >= 0 else R,
                    fontsize=8, fontweight="bold")

        step = max(1, len(healed_i) // 10)
        ax.set_xticks(xh[::step])
        ax.set_xticklabels(
            [f"B{healed_i[j]+1}" for j in range(0, len(healed_i), step)],
            fontsize=8, color=TXT
        )

    ax.set_ylim(0, 115)
    ax.legend(facecolor=PAN, labelcolor=TXT, fontsize=9)
    ax_style(ax, "Pre vs Post Heal Accuracy (Healed Batches Only)",
             xlabel="Batch", ylabel="Accuracy (%)")
    save_fig(fig, fname)


# ── Plot 6: Healing loss decomposition ────────────────────
def plot_heal_loss(log, fname="plot6_healing_loss.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    healed_i = [i for i, h in enumerate(log["healed"]) if h]

    if healed_i:
        xs = [i+1 for i in healed_i]
        ax.plot(xs, [log["bce_real"][i]  for i in healed_i],
                color=G, lw=2, marker="o", ms=5,
                label="BCE Real Labels")
        ax.plot(xs, [log["bce_const"][i] for i in healed_i],
                color=Y, lw=2, marker="s", ms=5,
                label="BCE Constraint (Symbolic)")
        # shade dynamic multiplier regions
        for i in healed_i:
            c = G if log["const_mult"][i] == CFG["const_mult_high"] else R
            ax.axvspan(i+0.6, i+1.4, color=c, alpha=0.06)

    ax.legend(facecolor=PAN, labelcolor=TXT, fontsize=9)
    ax_style(ax, "Healing Loss Decomposition (Real BCE vs Constraint BCE)",
             xlabel="Batch", ylabel="Avg Loss per Heal Event")
    save_fig(fig, fname)


# ── Plot 7: Full metrics summary (hero bar chart) ─────────
def plot_metrics_summary(clean_m, drift_m, healed_m,
                         fname="plot7_metrics_summary.png"):
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics  = ["Accuracy", "Precision", "Recall", "F1"]
    clean_v  = [clean_m["acc"],  clean_m["prec"],
                clean_m["rec"],  clean_m["f1"]]
    drift_v  = [drift_m["acc"],  drift_m["prec"],
                drift_m["rec"],  drift_m["f1"]]
    healed_v = [healed_m["acc"], healed_m["prec"],
                healed_m["rec"], healed_m["f1"]]

    xm = np.arange(len(metrics))
    b1 = ax.bar(xm - 0.25, clean_v,  width=0.23,
                color=G, alpha=0.85, label="Clean Baseline")
    b2 = ax.bar(xm,        drift_v,  width=0.23,
                color=R, alpha=0.85, label="Under Drift (No Healing)")
    b3 = ax.bar(xm + 0.25, healed_v, width=0.23,
                color=B, alpha=0.85, label="Self-Healed (v6)")

    # Value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + 0.012, f"{h:.2f}",
                    ha="center", color=TXT,
                    fontsize=8, fontweight="bold")

    ax.set_xticks(xm)
    ax.set_xticklabels(metrics, color=TXT, fontsize=11)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Score", color=TXT)
    ax.legend(facecolor=PAN, labelcolor=TXT, fontsize=9)
    ax_style(ax, "Full Metrics Comparison: Clean vs Drift vs Self-Healed")
    save_fig(fig, fname)


# ── Plot 8: Version progression (article narrative) ───────
def plot_version_history(clean_acc, drift_acc, healed_acc,
                         fname="plot8_version_history.png"):
    fig, ax = plt.subplots(figsize=(11, 6))

    versions = ["v1\n(Label bug)", "v2\n(No trigger)",
                "v3\n(+33.7pp)", "v4\n(Overshot)",
                "v5\n(Neutral)", "v6\n(Balanced)"]
    # Approximate accuracies from run history
    accs     = [0.17, 0.589, 0.783, 0.381, 0.430, healed_acc]
    colors   = [R, R, G, R, Y, B]

    bars = ax.bar(versions, [v*100 for v in accs],
                  color=colors, alpha=0.85, width=0.6)

    ax.axhline(y=clean_acc*100, color=G, lw=1.5, ls="--",
               alpha=0.5, label=f"Clean baseline ({clean_acc*100:.1f}%)")
    ax.axhline(y=drift_acc*100, color=R, lw=1.5, ls="--",
               alpha=0.5, label=f"No-healing drift ({drift_acc*100:.1f}%)")

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f"{val*100:.1f}%",
                ha="center", color=TXT,
                fontsize=10, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", color=TXT)
    ax.legend(facecolor=PAN, labelcolor=TXT, fontsize=9)
    ax_style(ax, "Iteration Journey: From Catastrophic Failure to Proof")
    save_fig(fig, fname)


# ============================================================
# 14. MAIN
# ============================================================

def main():
    print("=" * 62)
    print("  SELF-HEALING NEURAL NETWORKS  v6.0  — PROOF RUN")
    print("  Strict-Gated Lagrangian Healing + FIDI + Reflexive Loop")
    print("=" * 62)

    # ── Data ──────────────────────────────────────────────
    print("\n[1/6] Generating data...")
    X_train, y_train = generate_fraud_data(CFG["n_train"], drift=False)
    X_test,  y_test  = generate_fraud_data(CFG["n_test"],  drift=False)
    X_drift, y_drift = generate_fraud_data(CFG["n_test"],  drift=True)

    v14_c = X_test[:, 7].numpy()
    v14_d = X_drift[:, 7].numpy()
    print(f"    Train  : {len(X_train)}  |  Test: {len(X_test)}"
          f"  |  Fraud: {y_train.mean().item()*100:.1f}%")
    print(f"    V14 clean | mean={v14_c.mean():.3f}  "
          f"pct<-1.5={np.mean(v14_c<-1.5)*100:.1f}%")
    print(f"    V14 drift | mean={v14_d.mean():.3f}  "
          f"pct<-1.5={np.mean(v14_d<-1.5)*100:.1f}%")

    # ── Models ────────────────────────────────────────────
    print("\n[2/6] Initialising models...")
    heal_model = SelfHealingMLP(10, CFG["hidden_dim"])
    base_model = BaselineMLP(10,   CFG["hidden_dim"])
    symbolic   = SymbolicRuleEngine()
    fidi       = FIDIMonitor()

    print(f"    SelfHealingMLP : "
          f"{sum(p.numel() for p in heal_model.parameters()):,} params")
    print(f"    BaselineMLP    : "
          f"{sum(p.numel() for p in base_model.parameters()):,} params")
    print(f"    SymbolicRule   : V14 < {symbolic.THRESHOLD} → Fraud")
    print(f"    Heal loss      : {CFG['semi_sup_alpha']}×BCE(real) + "
          f"{1-CFG['semi_sup_alpha']}×{CFG['lambda_lagrange']}×[dynamic]×constraint")
    print(f"    Healing gate   : drift OR n_conf>={CFG['conflict_for_drift']}"
          f"  AND fraud>={CFG['min_fraud_frac']*100:.0f}%"
          f"  AND n_conf>={CFG['conflict_min']}")

    # ── Train ─────────────────────────────────────────────
    print("\n[3/6] Training on clean data...")
    train(heal_model, X_train, y_train, "SelfHealingMLP")
    train(base_model, X_train, y_train, "BaselineMLP")

    # ── FIDI ──────────────────────────────────────────────
    print("\n[4/6] Calibrating FIDI...")
    fidi.calibrate(X_train)

    # ── Clean eval ────────────────────────────────────────
    print("\n[5/6] Pre-Drift Evaluation:")
    clean_h = evaluate(heal_model, X_test, y_test,
                       "SelfHealing — Clean")
    _       = evaluate(base_model,  X_test, y_test,
                       "Baseline    — Clean")

    # ── Stream ────────────────────────────────────────────
    heal_stream = copy.deepcopy(heal_model)
    base_stream = copy.deepcopy(base_model)

    print("\n[6/6] Streaming drift experiment...")
    log = stream_experiment(heal_stream, base_stream,
                            X_drift, y_drift, fidi, symbolic)

    # ── Final eval ────────────────────────────────────────
    print("\n── Final Evaluation on Full Drift Test Set ──")
    drift_m  = evaluate(base_stream, X_drift, y_drift,
                        "Baseline    — Under Drift")
    healed_m = evaluate(heal_stream, X_drift, y_drift,
                        "SelfHealing — After Adaptation")

    # ── Summary ───────────────────────────────────────────
    drop     = (clean_h["acc"]  - drift_m["acc"])  * 100
    recovery = (healed_m["acc"] - drift_m["acc"])  * 100
    retained = (healed_m["acc"] / clean_h["acc"])  * 100
    f1_gain  =  healed_m["f1"]  - drift_m["f1"]
    n_healed = sum(log["healed"])
    n_skipped= sum(1 for r in log["skipped_reason"] if r)

    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY  v6.0")
    print("=" * 62)
    print(f"\n  {'Stage':<35} {'Acc':>7}  {'Prec':>7}  "
          f"{'Rec':>7}  {'F1':>7}")
    print(f"  {'─' * 58}")
    print(f"  {'Clean Baseline':<35} "
          f"{clean_h['acc']*100:>6.1f}%  "
          f"{clean_h['prec']:>7.4f}  "
          f"{clean_h['rec']:>7.4f}  "
          f"{clean_h['f1']:>7.4f}")
    print(f"  {'Under Drift — No Healing':<35} "
          f"{drift_m['acc']*100:>6.1f}%  "
          f"{drift_m['prec']:>7.4f}  "
          f"{drift_m['rec']:>7.4f}  "
          f"{drift_m['f1']:>7.4f}")
    print(f"  {'Under Drift — Self-Healed':<35} "
          f"{healed_m['acc']*100:>6.1f}%  "
          f"{healed_m['prec']:>7.4f}  "
          f"{healed_m['rec']:>7.4f}  "
          f"{healed_m['f1']:>7.4f}")
    print(f"\n  {'─' * 58}")
    print(f"\n  Accuracy drop from drift        : -{drop:.2f}%")
    print(f"  Accuracy recovery               : {recovery:+.2f}%")
    print(f"  Baseline retention              :  {retained:.1f}%")
    print(f"  F1 gain from healing            : {f1_gain:+.4f}")
    print(f"  Healing events triggered        :  {n_healed}/{CFG['n_batches']}")
    print(f"  Low-fraud batches skipped       :  {n_skipped}")
    print(f"\n  ✅ Backbone weights  : NEVER retrained")
    print(f"  ✅ ReflexiveLayer    : adaptive steps (3–6) per event")
    print(f"  ✅ No pos_weight     : removed (was root cause of v4/v5)")
    print(f"  ✅ Strict gate       : skips pure-normal conflict batches")
    print(f"  ✅ No API. No external data. Pure PyTorch.")
    print("=" * 62)

    # ── Version history ───────────────────────────────────
    print("""
  VERSION HISTORY (article narrative):
  ──────────────────────────────────────────────────────────
  v1  → Catastrophic collapse   (torch.ones() label bug)
  v2  → No healing triggered    (thresholds too strict)
  v3  → +33.7pp acc recovery    (recall collapsed to 14%)
  v4  → pos_weight=6× overshot  (accuracy crashed to 38%)
  v5  → Neutral outcome         (mild pos_weight, no gate)
  v6  → Strict gate + no pw     ← THIS RUN
  ──────────────────────────────────────────────────────────
    """)

    # ── Plots (8 separate files) ──────────────────────────
    print("Saving 8 separate plots...\n")
    plot_accuracy(log, clean_h["acc"],     "plot1_accuracy.png")
    plot_fidi(log,                          "plot2_fidi_zscore.png")
    plot_conflicts(log,                     "plot3_conflicts.png")
    plot_fraud_frac(log,                    "plot4_fraud_fraction.png")
    plot_pre_post(log,                      "plot5_pre_post_heal.png")
    plot_heal_loss(log,                     "plot6_healing_loss.png")
    plot_metrics_summary(clean_h,
                         drift_m, healed_m,"plot7_metrics_summary.png")
    plot_version_history(clean_h["acc"],
                         drift_m["acc"],
                         healed_m["acc"],  "plot8_version_history.png")

    print("""
  MAP PLOTS TO ARTICLE SECTIONS:
  ──────────────────────────────────────────────────────────
  plot1_accuracy.png       →  "Self-Healing in Action" (HERO)
  plot2_fidi_zscore.png    →  "Detecting Drift in Real Time"
  plot3_conflicts.png      →  "The Conscience Layer"
  plot4_fraud_fraction.png →  "Why the Gate Matters"
  plot5_pre_post_heal.png  →  "Batch-Level Recovery Evidence"
  plot6_healing_loss.png   →  "Lagrangian Constraint Deep-Dive"
  plot7_metrics_summary.png→  "Results" (HERO bar chart)
  plot8_version_history.png→  "The Journey to the Proof"
  ──────────────────────────────────────────────────────────
    """)


if __name__ == "__main__":
    main()
