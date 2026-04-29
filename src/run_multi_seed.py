"""
Multi-Seed Statistical Validation Runner.

Runs the complete training pipeline (baseline → abstention → 4 experiments)
across multiple random seeds and computes statistical summaries.

This script validates that the reported findings (e.g., abstention collapse)
are reproducible and statistically significant, not artifacts of random
initialization.

Usage:
    cd src
    python run_multi_seed.py
"""

import os
import sys
import time
import csv
import numpy as np
import pandas as pd
import torch
from scipy import stats

from dataset import load_data, FraudDataset, resolve_path
from baseline_model import BaselineModel
from abstention_model import AbstentionModel
from train_baseline import train_baseline
from train_abstention import train_abstention
from train_experiments import run_experiment, EXPERIMENTS
from evaluation import evaluate_model, EVAL_MODELS
from seed import set_seed
from torch.utils.data import DataLoader


# ============================================================
# Configuration
# ============================================================
SEEDS = [42, 123, 256]
METRICS_KEYS = ["Accuracy", "Coverage", "Selective Risk", "ECE", "F1 Score",
                "Precision (Fraud)", "Recall (Fraud)",
                "Abstained Total", "Abstained Fraud"]
MODEL_NAMES = list(EVAL_MODELS.keys())


def evaluate_all_models(device):
    """Evaluate all 6 models and return metrics dict."""
    _, _, X_test_np, _, _, y_test_np, _ = load_data("data/creditcard.csv")
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test = y_test_np

    results = {}
    for name, config in EVAL_MODELS.items():
        if not os.path.exists(resolve_path(config["file"])):
            print(f"  Skipping {name} (not found: {config['file']})")
            continue

        if config["architecture"] == "baseline":
            model = BaselineModel(input_dim=30, dropout=0.0).to(device)
        else:
            model = AbstentionModel(input_dim=30, dropout=0.0).to(device)

        model.load_state_dict(torch.load(resolve_path(config["file"]), map_location=device, weights_only=True))
        metrics = evaluate_model(model, X_test, y_test, device, config["has_abstain"])
        results[name] = metrics

    return results


def run_single_seed(seed, device):
    """Run the full pipeline for a single seed and return evaluation metrics."""
    print(f"\n{'#' * 90}")
    print(f"# MULTI-SEED RUN: Seed = {seed}")
    print(f"{'#' * 90}")

    seed_start = time.time()

    # Phase 1: Baseline
    print(f"\n--- Phase 1: Baseline (seed={seed}) ---")
    train_baseline(seed=seed)

    # Phase 2: Abstention
    print(f"\n--- Phase 2: Abstention (seed={seed}) ---")
    train_abstention(seed=seed)

    # Phase 3: Experiments 1-4
    print(f"\n--- Phase 3: Experiments (seed={seed}) ---")
    X_train, X_val, _, y_train, y_val, _, _ = load_data("data/creditcard.csv")
    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)

    n_legit = sum(y_train == 0)
    n_fraud = sum(y_train == 1)
    fraud_weight = min(n_legit / n_fraud, 50.0)
    class_weights = torch.tensor([1.0, fraud_weight], dtype=torch.float32).to(device)

    for exp_id, config in EXPERIMENTS.items():
        batch_size = 64 if config["grad_accum"] else 256
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        run_experiment(
            exp_id=exp_id,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            use_grad_accum=config["grad_accum"],
            use_mixed_precision=config["mixed_precision"],
            class_weights=class_weights,
            seed=seed,
        )

    # Phase 4: Evaluate all models
    print(f"\n--- Phase 4: Evaluation (seed={seed}) ---")
    results = evaluate_all_models(device)

    elapsed = time.time() - seed_start
    print(f"\n  Seed {seed} completed in {elapsed / 60:.1f} minutes")

    return results


def compute_statistics(all_results):
    """Compute mean ± std and paired t-tests across seeds."""

    # Collect per-model, per-metric arrays
    summary_rows = []

    for model_name in MODEL_NAMES:
        row = {"Model Name": model_name}
        for metric in METRICS_KEYS:
            values = []
            for seed_results in all_results:
                if model_name in seed_results:
                    values.append(seed_results[model_name].get(metric, 0.0))

            if values:
                arr = np.array(values)
                row[f"{metric} Mean"] = np.mean(arr)
                row[f"{metric} Std"] = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
                row[f"{metric} Values"] = str(values)
            else:
                row[f"{metric} Mean"] = np.nan
                row[f"{metric} Std"] = np.nan
                row[f"{metric} Values"] = "[]"

        summary_rows.append(row)

    return summary_rows


def run_statistical_tests(all_results):
    """Run paired t-tests for key comparisons."""

    comparisons = [
        ("Exp 1 (Standard)", "Baseline", "DAC vs Baseline"),
        ("Exp 2 (Grad Accum)", "Exp 1 (Standard)", "Grad Accum causes collapse?"),
        ("Exp 3 (Mixed Prec)", "Exp 1 (Standard)", "Mixed Prec causes collapse?"),
        ("Exp 4 (Combined)", "Exp 1 (Standard)", "Combined recovers performance?"),
        ("Exp 4 (Combined)", "Exp 2 (Grad Accum)", "Combined vs Grad Accum alone"),
    ]

    test_metric = "F1 Score"
    test_rows = []

    for model_a, model_b, description in comparisons:
        values_a = []
        values_b = []

        for seed_results in all_results:
            if model_a in seed_results and model_b in seed_results:
                values_a.append(seed_results[model_a].get(test_metric, 0.0))
                values_b.append(seed_results[model_b].get(test_metric, 0.0))

        if len(values_a) >= 2:
            arr_a = np.array(values_a)
            arr_b = np.array(values_b)
            t_stat, p_value = stats.ttest_rel(arr_a, arr_b)
            significant = "Yes" if p_value < 0.05 else "No"
        else:
            t_stat = np.nan
            p_value = np.nan
            significant = "Insufficient data"

        test_rows.append({
            "Comparison": description,
            "Model A": model_a,
            "Model B": model_b,
            "Metric": test_metric,
            "Mean A": np.mean(values_a) if values_a else np.nan,
            "Mean B": np.mean(values_b) if values_b else np.nan,
            "t-statistic": t_stat,
            "p-value": p_value,
            "Significant (p<0.05)": significant,
            "N Seeds": len(values_a),
        })

    return test_rows


def main():
    """Run multi-seed validation pipeline."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(SEEDS)}")

    os.makedirs(resolve_path("results"), exist_ok=True)
    total_start = time.time()

    all_results = []

    for seed in SEEDS:
        try:
            seed_results = run_single_seed(seed, DEVICE)
            all_results.append(seed_results)
        except Exception as e:
            print(f"ERROR: Seed {seed} failed with: {e}")
            continue

    if not all_results:
        print("ERROR: No seeds completed successfully.")
        sys.exit(1)

    # ---- Compute summary statistics ----
    print("\n" + "=" * 90)
    print("MULTI-SEED SUMMARY STATISTICS")
    print("=" * 90)

    summary_rows = compute_statistics(all_results)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(resolve_path("results/multi_seed_summary.csv"), index=False)

    # Print formatted summary
    print(f"\n{'Model':<22} | {'F1 Mean':>8} | {'F1 Std':>7} | {'Acc Mean':>8} | {'Cov Mean':>8} | {'ECE Mean':>8}")
    print("-" * 75)
    for row in summary_rows:
        print(f"{row['Model Name']:<22} | "
              f"{row.get('F1 Score Mean', 0):.4f}  | "
              f"{row.get('F1 Score Std', 0):.4f} | "
              f"{row.get('Accuracy Mean', 0):.4f}  | "
              f"{row.get('Coverage Mean', 0):.4f}  | "
              f"{row.get('ECE Mean', 0):.4f}")

    # ---- Statistical tests ----
    print("\n" + "=" * 90)
    print("PAIRED T-TESTS (on F1 Score)")
    print("=" * 90)

    test_rows = run_statistical_tests(all_results)
    df_tests = pd.DataFrame(test_rows)
    df_tests.to_csv(resolve_path("results/statistical_tests.csv"), index=False)

    for row in test_rows:
        p_str = f"{row['p-value']:.4f}" if not np.isnan(row['p-value']) else "N/A"
        print(f"  {row['Comparison']:<40} | "
              f"t={row['t-statistic']:>7.3f} | "
              f"p={p_str:<6} | "
              f"Sig: {row['Significant (p<0.05)']}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 90}")
    print(f"MULTI-SEED VALIDATION COMPLETED IN {total_time / 60:.1f} MINUTES")
    print(f"Seeds used: {SEEDS}")
    print(f"Summary saved to: results/multi_seed_summary.csv")
    print(f"Tests saved to: results/statistical_tests.csv")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
