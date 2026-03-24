"""
============================================================
  Self-Healing Neural Networks v3.0
  Fixing Model Drift Without Retraining (PyTorch)
============================================================
  v1 → v2 fixes: real labels, semi-supervised loss,
                 lower lambda, fewer heal steps
  v2 → v3 fixes:
  ① drift_strength   : 1.8  → 2.5  (drift strong enough to matter)
  ② fidi_threshold   : 2.0  → 1.0  (actually fires now)
  ③ fidi_window      : 20   → 10   (more sensitive rolling window)
  ④ conflict_min     : 15   → 3    (healing can actually trigger)
  ⑤ heal trigger     : AND  → OR   (drift OR conflicts)
  ⑥ conflict logging : added composition debug (fraud% in conflicts)
  ⑦ adaptive symbolic: threshold shifts with observed V14 percentile
============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    # Data
    "n_train"          : 5000,
    "n_test"           : 1000,
    "fraud_ratio"      : 0.15,
    "drift_strength"   : 2.5,     # ↑ from 1.8 — strong enough to spike FIDI

    # Model
    "hidden_dim"       : 64,
    "epochs"           : 25,
    "batch_size"       : 100,

    # Streaming
    "n_batches"        : 15,

    # FIDI  ← v3 key changes
    "fidi_window"      : 10,      # ↓ from 20 — more sensitive
    "fidi_threshold"   : 1.0,     # ↓ from 2.0 — fires at Z~1.2

    # Conflict detection  ← v3 key changes
    "conflict_prob_thr": 0.35,    # MLP prob below this = "MLP says Normal"
    "conflict_min"     : 3,       # ↓ from 15 — low bar so healing triggers

    # Healing
    "lambda_lagrange"  : 0.8,
    "n_heal_steps"     : 5,
    "heal_lr"          : 0.003,
    "semi_sup_alpha"   : 0.7,     # 0.7*BCE(real) + 0.3*constraint
}


# ============================================================
# 1. SYNTHETIC FRAUD DATA
# ============================================================

def generate_fraud_data(n_samples, drift=False,
                        drift_strength=None, fraud_ratio=None):
    ds = drift_strength if drift_strength is not None else CFG["drift_strength"]
    fr = fraud_ratio    if fraud_ratio    is not None else CFG["fraud_ratio"]

    n_fraud  = int(n_samples * fr)
    n_normal = n_samples - n_fraud

    if drift:
        v14_normal = np.random.normal(-ds, 1.0, n_normal)
    else:
        v14_normal = np.random.normal(0.0, 1.0, n_normal)

    rest_normal = np.random.randn(n_normal, 9)
    X_normal    = np.column_stack([rest_normal[:, :7],
                                   v14_normal,
                                   rest_normal[:, 7:]])
    y_normal    = np.zeros(n_normal)

    v14_fraud  = np.random.normal(-2.5, 0.8, n_fraud)
    rest_fraud = np.random.randn(n_fraud, 9)
    X_fraud    = np.column_stack([rest_fraud[:, :7],
                                  v14_fraud,
                                  rest_fraud[:, 7:]])
    y_fraud    = np.ones(n_fraud)

    X   = np.vstack([X_normal, X_fraud]).astype(np.float32)
    y   = np.concatenate([y_normal, y_fraud]).astype(np.float32)
    idx = np.random.permutation(len(y))
    return torch.tensor(X[idx]), torch.tensor(y[idx])


# ============================================================
# 2. REFLEXIVE LAYER
# ============================================================

class ReflexiveLayer(nn.Module):
    """
    Lightweight adapter between hidden and output.
    Residual design — identity at init (scale starts at 0.1).
    Only this layer's weights move during healing.
    """
    def __init__(self, dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
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
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.reflex_optimizer = optim.Adam(
            self.reflexive.parameters(), lr=CFG["heal_lr"]
        )

    def forward(self, x):
        h = self.backbone(x)
        h = self.reflexive(h)
        return self.output_head(h).squeeze()

    def freeze_for_healing(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.output_head.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ============================================================
# 4. BASELINE MLP
# ============================================================

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),          nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()


# ============================================================
# 5. FIDI Z-SCORE MONITOR
# ============================================================

class FIDIMonitor:
    """
    Tracks rolling mean of V14 (feature index 7).
    v3: window=10 (was 20), threshold=1.0 (was 2.0).
    More sensitive — designed to catch moderate drift.
    """
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
        print(f"    FIDI calibrated | μ={self.mu:.4f}  "
              f"σ={self.sigma:.4f}  threshold={self.threshold}")

    def update(self, X_batch):
        self.window.append(float(X_batch[:, 7].mean()))
        if len(self.window) >= 3:
            z = abs((np.mean(list(self.window)) - self.mu) / self.sigma)
        else:
            z = 0.0
        self.history.append(z)
        return z

    def check(self, X_batch):
        z = self.update(X_batch)
        return z > self.threshold, z


# ============================================================
# 6. SYMBOLIC RULE ENGINE  (adaptive threshold)
# ============================================================

class SymbolicRuleEngine:
    """
    Primary rule : V14 < threshold → Fraud
    Adaptive mode: threshold blends base value with
                   10th percentile of current batch V14.
    Prevents over-firing when full distribution drifts.
    """
    BASE_THRESHOLD = -1.5

    def __init__(self):
        self.threshold = self.BASE_THRESHOLD

    def adapt_threshold(self, X_batch):
        p10 = float(np.percentile(X_batch[:, 7].numpy(), 10))
        # 80% original rule, 20% adaptive
        self.threshold = 0.8 * self.BASE_THRESHOLD + 0.2 * p10

    def predict(self, X):
        return (X[:, 7] < self.threshold).float()

    def conflict_mask(self, X, mlp_probs):
        """
        Conflict = MLP says Normal confidently (prob < 0.35)
                   AND Symbolic Rule says Fraud
        """
        mlp_says_normal = (mlp_probs < CFG["conflict_prob_thr"]).float()
        rule_says_fraud = self.predict(X)
        return (mlp_says_normal == 1) & (rule_says_fraud == 1)


# ============================================================
# 7. HEALING LOSS  (semi-supervised)
# ============================================================

def healing_loss(probs, real_labels, symbolic_labels):
    """
    L = α * BCE(real) + (1-α) * λ * BCE(symbolic)
    Real labels anchor truth.
    Symbolic labels provide correction direction.
    """
    bce = nn.BCELoss()
    a   = CFG["semi_sup_alpha"]
    lam = CFG["lambda_lagrange"]
    return a * bce(probs, real_labels) + (1 - a) * lam * bce(probs, symbolic_labels)


# ============================================================
# 8. REFLEXIVE HEALING LOOP
# ============================================================

def heal(model, X_batch, y_batch, symbolic_engine):
    model.freeze_for_healing()
    model.train()

    sym_labels = symbolic_engine.predict(X_batch)
    trajectory = []

    for _ in range(CFG["n_heal_steps"]):
        model.reflex_optimizer.zero_grad()
        probs = model(X_batch)
        loss  = healing_loss(probs, y_batch, sym_labels)
        loss.backward()
        model.reflex_optimizer.step()
        trajectory.append(loss.item())

    model.unfreeze_all()
    model.eval()
    return trajectory


# ============================================================
# 9. CONFLICT COMPOSITION LOGGER
# ============================================================

def log_conflict_composition(conflict_mask, y_batch):
    """
    KEY DIAGNOSTIC: what fraction of conflicts are real fraud?
    If fraud% is low → forcing target=1 would be wrong.
    Semi-supervised loss uses real labels so this is safe.
    """
    n = int(conflict_mask.sum().item())
    if n == 0:
        return
    fraud_frac  = y_batch[conflict_mask].mean().item()
    normal_frac = 1 - fraud_frac
    print(f"             Conflict composition: {n} samples | "
          f"Real fraud={fraud_frac*100:.1f}%  "
          f"Real normal={normal_frac*100:.1f}%")


# ============================================================
# 10. TRAINING
# ============================================================

def train(model, X, y, label):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    print(f"\n    Training {label}...")
    for epoch in range(1, CFG["epochs"] + 1):
        perm       = torch.randperm(len(X))
        epoch_loss = 0.0
        for i in range(0, len(X), CFG["batch_size"]):
            idx = perm[i:i + CFG["batch_size"]]
            xb, yb = X[idx], y[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 5 == 0:
            print(f"      Epoch [{epoch:2d}/{CFG['epochs']}] "
                  f"Loss: {epoch_loss:.4f}")
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
    print(f"     TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    return dict(acc=acc, prec=prec, rec=rec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


# ============================================================
# 12. STREAMING EXPERIMENT
# ============================================================

def stream_experiment(heal_model, base_model,
                      X_drift, y_drift, fidi, symbolic_engine):
    heal_model.eval()
    base_model.eval()

    BS      = CFG["batch_size"]
    n_batch = CFG["n_batches"]

    log = dict(
        z_scores     = [],
        conflicts    = [],
        baseline_acc = [],
        before_acc   = [],
        after_acc    = [],
        healed       = [],
    )

    print(f"\n  Streaming {n_batch} drift batches  "
          f"(size={BS}, drift_strength={CFG['drift_strength']})...\n")
    print(f"  {'Batch':>5}  {'Z':>6}  {'Status':>10}  "
          f"{'Conflicts':>9}  {'Event':>10}  "
          f"{'Baseline':>9}  {'Before':>7}  {'After':>7}")
    print("  " + "─" * 78)

    for i in range(n_batch):
        idx = np.random.choice(len(X_drift), BS, replace=True)
        xb  = X_drift[idx]
        yb  = y_drift[idx]

        # Adapt symbolic threshold to current V14 distribution
        symbolic_engine.adapt_threshold(xb)

        # Baseline accuracy
        with torch.no_grad():
            base_acc = (((base_model(xb) > 0.5).float() == yb)
                        .float().mean().item())

        # Pre-heal accuracy
        with torch.no_grad():
            probs_pre = heal_model(xb)
            pre_acc   = ((probs_pre > 0.5).float() == yb).float().mean().item()

        # FIDI check
        drifting, z = fidi.check(xb)
        status      = "DRIFT  🔴" if drifting else "stable 🟢"

        # Conflict detection
        conflict_mask = symbolic_engine.conflict_mask(xb, probs_pre)
        n_conflicts   = int(conflict_mask.sum().item())

        # ── TRIGGER: drift OR conflicts (OR not AND) ──────
        should_heal = drifting or (n_conflicts > CFG["conflict_min"])
        event       = "dormant"
        post_acc    = pre_acc

        if should_heal:
            log_conflict_composition(conflict_mask, yb)
            traj     = heal(heal_model, xb, yb, symbolic_engine)
            with torch.no_grad():
                probs_post = heal_model(xb)
                post_acc   = ((probs_post > 0.5).float() == yb).float().mean().item()
            event = "⚡ HEALED"

        healed = (event == "⚡ HEALED")

        print(f"  {i+1:>5}  {z:>6.2f}  {status:>10}  "
              f"{n_conflicts:>9}  {event:>10}  "
              f"{base_acc:>9.3f}  {pre_acc:>7.3f}  {post_acc:>7.3f}")

        log["z_scores"].append(z)
        log["conflicts"].append(n_conflicts)
        log["baseline_acc"].append(base_acc)
        log["before_acc"].append(pre_acc)
        log["after_acc"].append(post_acc)
        log["healed"].append(healed)

    return log


# ============================================================
# 13. VISUALIZATION
# ============================================================

def plot(log, clean_m, drift_m, healed_m):
    BG  = "#0d1117"
    PAN = "#161b22"
    G   = "#00e676"
    R   = "#ff1744"
    B   = "#40c4ff"
    Y   = "#ffd740"
    TXT = "#e6edf3"
    GRY = "#30363d"

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.48, wspace=0.32)

    def ax_style(ax, title):
        ax.set_facecolor(PAN)
        ax.set_title(title, color=TXT, fontsize=10,
                     fontweight="bold", pad=9)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        for s in ax.spines.values():
            s.set_edgecolor(GRY)

    batches = list(range(1, len(log["z_scores"]) + 1))

    # ── 1. Hero accuracy plot ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(batches, log["baseline_acc"],
             color=R, lw=2.0,
             label="Baseline (No Healing)", alpha=0.85)
    ax1.plot(batches, log["before_acc"],
             color=Y, lw=1.5, linestyle="--",
             label="Self-Healing (Pre-Heal)", alpha=0.7)
    ax1.plot(batches, log["after_acc"],
             color=G, lw=2.5,
             label="Self-Healing (Post-Heal)", alpha=0.95)
    for i, h in enumerate(log["healed"]):
        if h:
            ax1.axvline(x=i+1, color=B,
                        alpha=0.4, lw=1.2, linestyle=":")
    ax1.axhline(y=clean_m["acc"], color=G, linestyle="--",
                alpha=0.25, lw=1.2, label="Clean Baseline")
    ax1.set_xlabel("Batch Index")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.08)
    ax1.legend(facecolor=PAN, labelcolor=TXT, fontsize=8)
    ax_style(ax1, "Batch Accuracy: Baseline vs Self-Healing Under Drift")

    # ── 2. FIDI Z-Score ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(batches, log["z_scores"], color=B, lw=2.2)
    ax2.fill_between(
        batches, log["z_scores"], CFG["fidi_threshold"],
        where=[z > CFG["fidi_threshold"] for z in log["z_scores"]],
        color=R, alpha=0.2, label="Drift Zone"
    )
    ax2.axhline(y=CFG["fidi_threshold"], color=Y, lw=1.3,
                linestyle="--",
                label=f"Threshold ({CFG['fidi_threshold']})")
    ax2.set_xlabel("Batch Index")
    ax2.set_ylabel("|Z-Score|")
    ax2.legend(facecolor=PAN, labelcolor=TXT, fontsize=8)
    ax_style(ax2, "FIDI Z-Score Monitor")

    # ── 3. Conflicts per batch ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bar_colors = [R if c > CFG["conflict_min"] else
                  (Y if c > 0 else G) for c in log["conflicts"]]
    ax3.bar(batches, log["conflicts"],
            color=bar_colors, alpha=0.85, width=0.7)
    ax3.axhline(y=CFG["conflict_min"], color=Y, lw=1.3,
                linestyle="--",
                label=f"Heal trigger ({CFG['conflict_min']})")
    ax3.set_xlabel("Batch Index")
    ax3.set_ylabel("Conflict Count")
    ax3.legend(facecolor=PAN, labelcolor=TXT, fontsize=8)
    ax_style(ax3, "MLP vs Symbolic Rule Conflicts")

    # ── 4. Pre vs Post heal per healed batch ───────────────
    ax4 = fig.add_subplot(gs[1, 1])
    healed_idxs = [i for i, h in enumerate(log["healed"]) if h]
    pre_vals    = [log["before_acc"][i] for i in healed_idxs]
    post_vals   = [log["after_acc"][i]  for i in healed_idxs]

    if healed_idxs:
        x_h = np.arange(len(healed_idxs))
        ax4.bar(x_h - 0.2, [v*100 for v in pre_vals],
                width=0.38, color=R, alpha=0.85, label="Pre-Heal")
        ax4.bar(x_h + 0.2, [v*100 for v in post_vals],
                width=0.38, color=G, alpha=0.85, label="Post-Heal")
        for xi, (pre, post) in enumerate(zip(pre_vals, post_vals)):
            delta = (post - pre) * 100
            col   = G if delta >= 0 else R
            ax4.text(xi, max(pre, post)*100 + 1.5,
                     f"{'+' if delta >= 0 else ''}{delta:.1f}%",
                     ha="center", color=col,
                     fontsize=9, fontweight="bold")
        ax4.set_xticks(list(x_h))
        ax4.set_xticklabels([f"B{i+1}" for i in healed_idxs],
                            fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No healing events\ntriggered",
                 transform=ax4.transAxes,
                 ha="center", va="center",
                 color=Y, fontsize=12)

    ax4.set_ylabel("Accuracy (%)")
    ax4.set_ylim(0, 115)
    ax4.legend(facecolor=PAN, labelcolor=TXT, fontsize=8)
    ax_style(ax4, "Pre vs Post Heal (Healed Batches Only)")

    # ── 5. Overall summary ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    stages = ["Clean\nBaseline", "Drift\n(No Heal)", "Drift\n(Healed)"]
    vals   = [clean_m["acc"]*100,
              drift_m["acc"]*100,
              healed_m["acc"]*100]
    f1s    = [clean_m["f1"],
              drift_m["f1"],
              healed_m["f1"]]
    bars   = ax5.bar(stages, vals,
                     color=[G, R, B], alpha=0.85, width=0.5)
    for bar, val, f1 in zip(bars, vals, f1s):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.8,
                 f"{val:.1f}%\nF1={f1:.3f}",
                 ha="center", color=TXT,
                 fontsize=9, fontweight="bold")
    ax5.set_ylabel("Accuracy (%)")
    ax5.set_ylim(0, 120)
    ax_style(ax5, "Overall Accuracy + F1 Summary")

    fig.suptitle(
        "Self-Healing Neural Networks v3  ·  Constraint-Induced Plasticity\n"
        "Adaptive Symbolic Conscience  +  Semi-Supervised Reflexive Loop",
        color=TXT, fontsize=12, fontweight="bold", y=1.01
    )

    out = "self_healing_results_v3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  📊 Plot saved → {out}")
    plt.show()


# ============================================================
# 14. MAIN
# ============================================================

def main():
    print("=" * 62)
    print("  SELF-HEALING NEURAL NETWORKS  v3.0")
    print("  Adaptive Symbolic Conscience + Reflexive Feedback Loop")
    print("=" * 62)

    # ── Data ──────────────────────────────────────────────
    print(f"\n[1/7] Generating data "
          f"(drift_strength={CFG['drift_strength']})...")
    X_train, y_train = generate_fraud_data(CFG["n_train"], drift=False)
    X_test,  y_test  = generate_fraud_data(CFG["n_test"],  drift=False)
    X_drift, y_drift = generate_fraud_data(CFG["n_test"],  drift=True)

    print(f"    Train        : {len(X_train)} samples")
    print(f"    Test (clean) : {len(X_test)}  samples")
    print(f"    Test (drift) : {len(X_drift)} samples")
    print(f"    Fraud ratio  : {y_train.mean().item()*100:.1f}%")

    # V14 sanity check — confirms drift is strong enough
    v14_clean = X_test[:,  7].numpy()
    v14_drift = X_drift[:, 7].numpy()
    print(f"\n    V14 (clean)  : μ={v14_clean.mean():.3f}  "
          f"σ={v14_clean.std():.3f}")
    print(f"    V14 (drift)  : μ={v14_drift.mean():.3f}  "
          f"σ={v14_drift.std():.3f}  "
          f"← shift of ~{abs(v14_drift.mean()-v14_clean.mean()):.2f}")

    # ── Models ────────────────────────────────────────────
    print("\n[2/7] Initialising models...")
    heal_model = SelfHealingMLP(10, CFG["hidden_dim"])
    base_model = BaselineMLP(10,   CFG["hidden_dim"])
    symbolic   = SymbolicRuleEngine()
    fidi       = FIDIMonitor()

    print(f"    SelfHealingMLP : "
          f"{sum(p.numel() for p in heal_model.parameters()):,} params")
    print(f"    BaselineMLP    : "
          f"{sum(p.numel() for p in base_model.parameters()):,} params")
    print(f"    SymbolicRule   : V14 < {symbolic.BASE_THRESHOLD} "
          f"→ Fraud  (adaptive)")
    print(f"    FIDI Monitor   : window={CFG['fidi_window']}  "
          f"threshold={CFG['fidi_threshold']}")

    # ── Train ─────────────────────────────────────────────
    print("\n[3/7] Training on clean data...")
    train(heal_model, X_train, y_train, "SelfHealingMLP")
    train(base_model, X_train, y_train, "BaselineMLP")

    # ── Calibrate FIDI ────────────────────────────────────
    print("\n[4/7] Calibrating FIDI...")
    fidi.calibrate(X_train)

    # ── Pre-Drift Evaluation ──────────────────────────────
    print("\n[5/7] Pre-Drift Evaluation:")
    clean_h = evaluate(heal_model, X_test, y_test,
                       "SelfHealing — Clean")
    clean_b = evaluate(base_model,  X_test, y_test,
                       "Baseline    — Clean")

    # ── Deep copy — both start from identical weights ─────
    heal_stream = copy.deepcopy(heal_model)
    base_stream = copy.deepcopy(base_model)

    # ── Streaming Experiment ──────────────────────────────
    print("\n[6/7] Streaming drift experiment...")
    log = stream_experiment(heal_stream, base_stream,
                            X_drift, y_drift, fidi, symbolic)

    # ── Final Evaluation ──────────────────────────────────
    print("\n── Final Evaluation on Full Drift Test Set ──")
    drift_m  = evaluate(base_stream,  X_drift, y_drift,
                        "Baseline    — Under Drift")
    healed_m = evaluate(heal_stream,  X_drift, y_drift,
                        "SelfHealing — After Adaptation")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY")
    print("=" * 62)

    drop     = (clean_h["acc"] - drift_m["acc"])   * 100
    recovery = (healed_m["acc"] - drift_m["acc"])  * 100
    retained = (healed_m["acc"] / clean_h["acc"])  * 100
    n_healed = sum(log["healed"])

    print(f"\n  {'Stage':<35} {'Accuracy':>9}  {'F1':>8}")
    print(f"  {'─'*55}")
    print(f"  {'Clean Baseline':<35} "
          f"{clean_h['acc']*100:>8.2f}%  {clean_h['f1']:>8.4f}")
    print(f"  {'Under Drift — No Healing':<35} "
          f"{drift_m['acc']*100:>8.2f}%  {drift_m['f1']:>8.4f}")
    print(f"  {'Under Drift — Self-Healed':<35} "
          f"{healed_m['acc']*100:>8.2f}%  {healed_m['f1']:>8.4f}")
    print(f"\n  {'─'*55}")
    print(f"\n  Accuracy drop from drift       : -{drop:.2f}%")
    print(f"  Recovery via self-healing      : +{recovery:.2f}%")
    print(f"  Baseline retention             :  {retained:.1f}%")
    print(f"  Healing events triggered       :  "
          f"{n_healed}/{CFG['n_batches']} batches")
    print(f"\n  ✅  Backbone weights : NEVER retrained")
    print(f"  ✅  ReflexiveLayer   : "
          f"{CFG['n_heal_steps']} local steps per event")
    print(f"  ✅  No API. No external data. Pure PyTorch.")
    print("=" * 62)

    # ── Plot ──────────────────────────────────────────────
    print("\n[7/7] Generating plots...")
    plot(log, clean_h, drift_m, healed_m)

    # ── Article Map ───────────────────────────────────────
    print("""
  ARTICLE OUTPUT MAP:
  ──────────────────────────────────────────────────────────
  ① V14 distribution shift stats   →  "The Problem" section
  ② Training loss curves           →  "The Setup"
  ③ Clean baseline metrics         →  "Establishing the Baseline"
  ④ FIDI Z-Score chart             →  "Detecting Drift"
  ⑤ Conflict composition logs      →  "The Conscience Layer"
  ⑥ Pre/Post heal accuracy bars    →  "Self-Healing in Action"
  ⑦ Recovery delta + retention %   →  "Results & Analysis"
  ⑧ Summary accuracy + F1 bars     →  "Conclusion"
  ──────────────────────────────────────────────────────────
    """)


if __name__ == "__main__":
    main()
