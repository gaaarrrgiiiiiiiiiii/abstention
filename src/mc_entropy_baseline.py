"""
mc_entropy_baseline.py
======================
MC Dropout predictive-entropy abstention baseline.

Uses 20 stochastic forward passes with dropout enabled to estimate
predictive uncertainty, then abstracts on the samples with the highest
entropy — matched to the same coverage as the DAC model.

This creates a clean "post-hoc entropy thresholding vs. end-to-end learned
abstention" comparison that reviewers of selective classification papers
consistently expect.

Usage:
    cd src
    python mc_entropy_baseline.py

Outputs:
    results/mc_entropy_metrics.csv — metrics at multiple coverage targets
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from dataset import load_data, resolve_path
from abstention_model import AbstentionModel
from mc_dropout import mc_dropout_inference
from metrics import coverage, selective_risk


def mc_entropy_abstention(model: AbstentionModel, X_test: np.ndarray,
                          y_test: np.ndarray, device: torch.device,
                          target_coverage: float = 0.9962,
                          num_passes: int = 20) -> dict:
    """
    Abstain on samples with the highest predictive entropy until
    the target coverage is reached.

    Args:
        target_coverage: fraction of samples NOT abstained on.
    """
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    mean_probs, _ = mc_dropout_inference(model, X_t, num_passes=num_passes)

    # Predictive entropy: H[y|x] = -Σ p_c * log(p_c)
    p_clipped = np.clip(mean_probs[:, :2], 1e-8, 1.0)   # only 2 real classes
    entropy    = -np.sum(p_clipped * np.log(p_clipped), axis=1)

    base_preds = np.argmax(mean_probs[:, :2], axis=1)

    N = len(X_test)
    n_to_abstain = int(round(N * (1.0 - target_coverage)))

    final_preds = base_preds.copy()
    if n_to_abstain > 0:
        # Abstain on the n_to_abstain samples with the highest entropy
        abstain_idx = np.argpartition(entropy, -n_to_abstain)[-n_to_abstain:]
        final_preds[abstain_idx] = 2

    cov      = coverage(final_preds)
    sel_risk_v = selective_risk(y_test, final_preds)
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
        "Model Name":           f"MC-Entropy (cov={target_coverage:.4f})",
        "Target Coverage":      target_coverage,
        "Actual Coverage":      float(cov),
        "Selective Risk":       float(sel_risk_v),
        "Accuracy":             float(acc),
        "F1 Score":             float(f1),
        "Precision (Fraud)":    float(prec),
        "Recall (Fraud)":       float(rec),
        "Abstained Total":      n_abst,
        "MC Passes":            num_passes,
    }


def run_mc_entropy_baseline():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MC Dropout Entropy Baseline — Device: {DEVICE}")

    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_data()
    input_dim = X_test.shape[1]

    model_path = resolve_path("abstention_model.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Run the main pipeline first.")
        return

    model = AbstentionModel(input_dim=input_dim, dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print(f"Loaded model from {model_path}")

    # Evaluate at coverage targets matching the DAC model's operating points
    coverage_targets = [0.9962, 0.9959, 0.9994, 0.999]
    results = []
    for cov_target in coverage_targets:
        metrics = mc_entropy_abstention(
            model, X_test, y_test, DEVICE,
            target_coverage=cov_target, num_passes=20
        )
        results.append(metrics)
        print(
            f"  Target cov={cov_target:.4f}  Actual cov={metrics['Actual Coverage']:.4f}  "
            f"F1={metrics['F1 Score']:.3f}  Abstained={metrics['Abstained Total']}"
        )

    df = pd.DataFrame(results)
    out_path = resolve_path("results/mc_entropy_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")
    return df


if __name__ == "__main__":
    run_mc_entropy_baseline()
