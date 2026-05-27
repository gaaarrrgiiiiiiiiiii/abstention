"""
selective_net_baseline.py
=========================
Implements SelectiveNet (Geifman & El-Yaniv, ICML 2019) as a proper
comparison baseline for the Deep Abstaining Classifier (DAC).

SelectiveNet learns a joint (classification head, selection head) architecture
and optimizes both coverage and selective risk end-to-end, with a Lagrangian
coverage constraint.

Usage:
    cd src
    python selective_net_baseline.py

Outputs:
    results/selective_net_metrics.csv   — evaluation metrics
    selective_net_model.pth             — trained model weights
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from dataset import load_data, FraudDataset, resolve_path
from metrics import coverage, selective_risk, expected_calibration_error
from seed import set_seed


# ── SelectiveNet architecture ─────────────────────────────────────────────────

class SelectiveNet(nn.Module):
    """
    SelectiveNet: shared representation + classification head + selection head.

    The selection head g(x) outputs a scalar in [0,1] representing the model's
    confidence in its own prediction. At inference, samples with g(x) < threshold
    are abstained on.

    Architecture mirrors the DAC model for a fair comparison:
        Shared: Linear(D→128) → BN → ReLU → Dropout → Linear(128→64) → BN → ReLU
        Classification head: Linear(64→2)
        Selection head:       Linear(64→1) → Sigmoid
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, 2)
        self.selector   = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.classifier(h)
        g      = self.selector(h).squeeze(1)   # (B,)  ∈ [0,1]
        return logits, g


# ── SelectiveNet loss ─────────────────────────────────────────────────────────

class SelectiveNetLoss(nn.Module):
    """
    Joint loss for SelectiveNet.

    L = selective_risk_loss + lambda_coverage * (coverage_constraint)^2
        + alpha_aux * auxiliary_loss

    Where:
        selective_risk_loss = (1/n) Σ L_CE(f(x), y) * g(x) / coverage_estimate
        coverage_estimate   = (1/n) Σ g(x)
        coverage_constraint = max(0, c_target - coverage_estimate)  [hinge]
        auxiliary_loss      = standard CE on all samples (prevents degenerate g=0)
    """

    def __init__(self, c_target: float = 0.99, lambda_coverage: float = 32.0,
                 alpha_aux: float = 0.5, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.c_target       = c_target
        self.lambda_coverage = lambda_coverage
        self.alpha_aux      = alpha_aux
        self.ce             = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def forward(self, logits: torch.Tensor, g: torch.Tensor,
                targets: torch.Tensor):
        per_sample_ce = self.ce(logits, targets)           # (B,)

        # Coverage estimate
        cov_est = g.mean()

        # Selective empirical risk: Σ CE * g / coverage_estimate
        selective_emp_risk = (per_sample_ce * g).sum() / (g.sum() + 1e-8)

        # Coverage constraint (Lagrangian penalty)
        cov_penalty = (max(0.0, self.c_target - cov_est.item())) ** 2
        cov_loss    = self.lambda_coverage * cov_penalty

        # Auxiliary loss (keeps classification head on track)
        aux_loss = per_sample_ce.mean()

        total = selective_emp_risk + cov_loss + self.alpha_aux * aux_loss
        return total, selective_emp_risk.item(), cov_est.item()


# ── training ──────────────────────────────────────────────────────────────────

def train_selective_net(X_train, y_train, X_val, y_val, input_dim: int,
                        device: torch.device, seed: int = 42,
                        c_target: float = 0.99,
                        max_epochs: int = 60, patience: int = 10):

    set_seed(seed)

    n_legit = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    fraud_w  = min(n_legit / n_fraud, 50.0)
    class_weights = torch.tensor([1.0, fraud_w], dtype=torch.float32).to(device)

    model     = SelectiveNet(input_dim=input_dim, dropout=0.3).to(device)
    criterion = SelectiveNetLoss(c_target=c_target, class_weights=class_weights)
    optimiser = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    train_loader = DataLoader(
        FraudDataset(X_train, y_train), batch_size=256, shuffle=True
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t  = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    training_log = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits, g = model(xb)
            loss, _, _ = criterion(logits, g, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            epoch_loss += loss.item() * len(xb)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, val_g = model(X_val_t)
            val_loss, sel_risk_v, cov_v = criterion(val_logits, val_g, y_val_t)

        scheduler.step(val_loss)
        training_log.append({
            "epoch": epoch + 1, "val_loss": val_loss.item(),
            "val_coverage": cov_v, "val_sel_risk": sel_risk_v
        })

        if val_loss.item() < best_val_loss - 1e-5:
            best_val_loss = val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), resolve_path("selective_net_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}.")
                break

    # Reload best weights
    model.load_state_dict(
        torch.load(resolve_path("selective_net_model.pth"), map_location=device, weights_only=True)
    )
    return model, pd.DataFrame(training_log)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_selective_net(model: SelectiveNet, X_test: np.ndarray,
                           y_test: np.ndarray, device: torch.device,
                           abstain_threshold: float = 0.5):
    """
    Evaluate the trained SelectiveNet.
    Samples where g(x) < abstain_threshold are abstained on (class=2 proxy).
    """
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits, g = model(X_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        g_np   = g.cpu().numpy()
        preds  = torch.argmax(logits, dim=1).cpu().numpy()

    # Apply selection: abstain where g < threshold
    final_preds = preds.copy()
    final_preds[g_np < abstain_threshold] = 2   # proxy for abstain

    cov      = coverage(final_preds)
    sel_risk_val = selective_risk(y_test, final_preds)
    n_abst   = int((final_preds == 2).sum())

    mask = final_preds != 2
    if mask.sum() > 0:
        acc  = accuracy_score(y_test[mask], final_preds[mask])
        f1   = f1_score(y_test[mask], final_preds[mask], pos_label=1, average="binary", zero_division=0.0)
        prec = precision_score(y_test[mask], final_preds[mask], pos_label=1, average="binary", zero_division=0.0)
        rec  = recall_score(y_test[mask], final_preds[mask], pos_label=1, average="binary", zero_division=0.0)
    else:
        acc = f1 = prec = rec = 0.0

    return {
        "Model Name":        f"SelectiveNet (c={abstain_threshold:.2f})",
        "Accuracy":          acc,
        "Coverage":          float(cov),
        "Selective Risk":    float(sel_risk_val),
        "F1 Score":          float(f1),
        "Precision (Fraud)": float(prec),
        "Recall (Fraud)":    float(rec),
        "Abstained Total":   n_abst,
        "Abstain Rate":      float(n_abst / len(y_test)),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run_selective_net_baseline():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SelectiveNet Baseline — Device: {DEVICE}")

    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_data()
    input_dim = X_train.shape[1]
    print(f"Input dim: {input_dim}, Train: {len(X_train)}, Test: {len(X_test)}")

    print("\nTraining SelectiveNet (c_target=0.99)...")
    t0 = time.time()
    model, log_df = train_selective_net(
        X_train, y_train, X_val, y_val, input_dim=input_dim,
        device=DEVICE, c_target=0.99
    )
    print(f"Training complete in {(time.time()-t0)/60:.1f} min.")

    # Evaluate at multiple thresholds
    thresholds = [0.3, 0.5, 0.7]
    results = []
    for thresh in thresholds:
        metrics = evaluate_selective_net(model, X_test, y_test, DEVICE,
                                         abstain_threshold=thresh)
        results.append(metrics)
        print(
            f"  Threshold={thresh:.1f}  Coverage={metrics['Coverage']:.4f}  "
            f"F1={metrics['F1 Score']:.3f}  Abstained={metrics['Abstained Total']}"
        )

    df = pd.DataFrame(results)
    out_path = resolve_path("results/selective_net_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    log_df.to_csv(resolve_path("results/selective_net_training_log.csv"), index=False)
    return df


if __name__ == "__main__":
    run_selective_net_baseline()
