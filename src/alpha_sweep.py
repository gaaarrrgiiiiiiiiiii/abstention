"""
alpha_sweep.py
==============
Ablation study: sweep the DAC abstention penalty α over a range of values
and plot the risk-coverage Pareto frontier.

Each value of α produces a different trained model with a different
abstention rate. Sweeping reveals the optimal operating point for a
given operational budget (how many transactions can be manually reviewed).

Usage:
    cd src
    python alpha_sweep.py

Outputs:
    results/alpha_pareto.csv          — per-alpha metrics
    results/plot_pareto_frontier.png  — Pareto frontier figure
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from dataset import load_data, FraudDataset, resolve_path
from abstention_model import AbstentionModel
from metrics import coverage, selective_risk
from sklearn.metrics import f1_score, precision_score, recall_score
from seed import set_seed

# ── alpha sweep configuration ─────────────────────────────────────────────────
ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIXED_SEED   = 42
MAX_EPOCHS   = 60
PATIENCE     = 10
LR           = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 256
CLASS_WEIGHT_CAP = 50.0

# ── DAC loss ──────────────────────────────────────────────────────────────────

class DACLoss(nn.Module):
    """
    Deep Abstaining Classifier loss with class weighting.
    L_DAC = w[y] * [-log(p_y + p_abstain) + alpha * p_abstain]
    """

    def __init__(self, alpha: float, class_weights: torch.Tensor):
        super().__init__()
        self.alpha = alpha
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        p_abstain = probs[:, 2]

        # p_true: probability of the correct class
        p_true = probs[torch.arange(len(targets)), targets]

        # Per-sample weights
        w = self.class_weights[targets]

        loss = w * (-torch.log(p_true + p_abstain + 1e-8) + self.alpha * p_abstain)
        return loss.mean()


# ── training for one alpha value ──────────────────────────────────────────────

def train_for_alpha(alpha: float, X_train, y_train, X_val, y_val,
                    input_dim: int, device: torch.device,
                    baseline_weights: dict | None = None) -> dict:
    """Train an abstention model for a given alpha and return evaluation metrics."""
    set_seed(FIXED_SEED)

    n_legit = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    fraud_w  = min(n_legit / n_fraud, CLASS_WEIGHT_CAP)
    class_weights = torch.tensor([1.0, fraud_w], dtype=torch.float32).to(device)

    model = AbstentionModel(input_dim=input_dim, dropout=0.3).to(device)

    # Warm-start from baseline weights (if available)
    if baseline_weights is not None:
        model_dict = model.state_dict()
        compatible = {
            k: v for k, v in baseline_weights.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(compatible)
        model.load_state_dict(model_dict)

    criterion = DACLoss(alpha=alpha, class_weights=class_weights)
    optimiser = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    train_loader = DataLoader(
        FraudDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t  = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimiser.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Evaluate on validation set (using it as pseudo-test for sweep speed)
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        logits   = model(X_test_t)
        probs_np = torch.softmax(logits, dim=1).cpu().numpy()
        preds_np = torch.argmax(logits, dim=1).cpu().numpy()

    cov      = coverage(preds_np)
    sel_risk = selective_risk(y_val, preds_np)
    n_abst   = int((preds_np == 2).sum())

    mask = preds_np != 2
    if mask.sum() > 0:
        f1   = f1_score(y_val[mask], preds_np[mask], pos_label=1,
                        average="binary", zero_division=0.0)
        prec = precision_score(y_val[mask], preds_np[mask], pos_label=1,
                               average="binary", zero_division=0.0)
        rec  = recall_score(y_val[mask], preds_np[mask], pos_label=1,
                            average="binary", zero_division=0.0)
    else:
        f1 = prec = rec = 0.0

    return {
        "alpha":          alpha,
        "coverage":       float(cov),
        "selective_risk": float(sel_risk),
        "f1":             float(f1),
        "precision":      float(prec),
        "recall":         float(rec),
        "abstained":      n_abst,
        "abstain_pct":    float(n_abst / len(y_val)),
        "val_loss":       float(best_val_loss),
    }


# ── Pareto frontier plot ──────────────────────────────────────────────────────

def plot_pareto_frontier(df: pd.DataFrame, out_path: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Coverage vs Selective Risk (Pareto)
    cmap = plt.cm.RdYlGn
    colors = [cmap(1 - i / (len(df) - 1)) for i in range(len(df))]

    for i, row in df.iterrows():
        axes[0].scatter(
            row["coverage"] * 100, row["selective_risk"] * 100,
            color=colors[i], s=120, zorder=3, edgecolor="black", linewidth=0.8
        )
        axes[0].annotate(
            f"α={row['alpha']:.1f}",
            (row["coverage"] * 100, row["selective_risk"] * 100),
            xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold"
        )
    axes[0].plot(df["coverage"] * 100, df["selective_risk"] * 100,
                 "--", color="gray", linewidth=1.2, alpha=0.7, zorder=2)
    axes[0].set_xlabel("Coverage (%)")
    axes[0].set_ylabel("Selective Risk (%)")
    axes[0].set_title("Risk-Coverage Pareto Frontier\n(by Abstention Penalty α)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Right: F1 vs Coverage
    axes[1].plot(df["coverage"] * 100, df["f1"], marker="o", color="#2196F3",
                 linewidth=2, markersize=7, markeredgecolor="black", markeredgewidth=0.8)
    for i, row in df.iterrows():
        axes[1].annotate(
            f"α={row['alpha']:.1f}",
            (row["coverage"] * 100, row["f1"]),
            xytext=(5, -12), textcoords="offset points", fontsize=9
        )
    axes[1].set_xlabel("Coverage (%)")
    axes[1].set_ylabel("F1 Score (Fraud Class)")
    axes[1].set_title("F1 Score vs Coverage\n(by Abstention Penalty α)")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("DAC Abstention Penalty Ablation Study", fontweight="bold",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Pareto frontier plot: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_alpha_sweep():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Alpha Sweep — Device: {DEVICE}")
    print(f"Alpha values: {ALPHA_VALUES}")
    print("=" * 70)

    # Load data once
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_data()
    input_dim = X_train.shape[1]

    # Load baseline weights for warm-starting each alpha model
    baseline_path = resolve_path("baseline_model.pth")
    baseline_weights = None
    if os.path.exists(baseline_path):
        baseline_weights = torch.load(baseline_path, map_location=DEVICE, weights_only=True)
        print(f"Loaded baseline weights for warm-start from {baseline_path}")

    os.makedirs(resolve_path("results"), exist_ok=True)
    results = []
    total_start = time.time()

    for alpha in ALPHA_VALUES:
        print(f"\n--- Training with α = {alpha:.1f} ---")
        t0 = time.time()
        metrics = train_for_alpha(
            alpha=alpha,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            input_dim=input_dim,
            device=DEVICE,
            baseline_weights=baseline_weights,
        )
        elapsed = time.time() - t0
        metrics["train_time_sec"] = round(elapsed, 1)
        results.append(metrics)
        print(
            f"  α={alpha:.1f}  Coverage={metrics['coverage']:.4f}  "
            f"Risk={metrics['selective_risk']:.4f}  F1={metrics['f1']:.3f}  "
            f"Abstained={metrics['abstained']} ({metrics['abstain_pct']:.2%})  "
            f"Time={elapsed:.0f}s"
        )

    df = pd.DataFrame(results)
    csv_path = resolve_path("results/alpha_pareto.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nPareto data saved: {csv_path}")

    plot_path = resolve_path("results/plot_pareto_frontier.png")
    plot_pareto_frontier(df, plot_path)

    total = time.time() - total_start
    print(f"\nAlpha sweep completed in {total / 60:.1f} minutes.")
    print(df[["alpha", "coverage", "selective_risk", "f1", "abstained"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run_alpha_sweep()
