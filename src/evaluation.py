import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from dataset import load_data, resolve_path
from baseline_model import BaselineModel
from abstention_model import AbstentionModel
from metrics import classification_metrics, coverage, selective_risk, expected_calibration_error

# ============================================================
# Model Loading Configuration
# ============================================================
EVAL_MODELS = {
    "Baseline": {
        "file": "baseline_model.pth", 
        "architecture": "baseline",
        "has_abstain": False
    },
    "Abstention (Phase 2)": {
        "file": "abstention_model.pth", 
        "architecture": "abstention",
        "has_abstain": True
    },
    "Exp 1 (Standard)": {
        "file": "results/experiment_1_model.pth", 
        "architecture": "abstention",
        "has_abstain": True
    },
    "Exp 2 (Grad Accum)": {
        "file": "results/experiment_2_model.pth", 
        "architecture": "abstention",
        "has_abstain": True
    },
    "Exp 3 (Mixed Prec)": {
        "file": "results/experiment_3_model.pth", 
        "architecture": "abstention",
        "has_abstain": True
    },
    "Exp 4 (Combined)": {
        "file": "results/experiment_4_model.pth", 
        "architecture": "abstention",
        "has_abstain": True
    }
}


def evaluate_model(model, X_test, y_test, device, has_abstain):
    """Run evaluation for a single model and compute all metrics including confusion matrix."""
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
    # ECE requires extracting the confidence of the *predicted* class
    # For abstention models, if it predicts abstain, we don't calculate ECE on those.
    # So we take the max probability of class 0 and 1.
    if has_abstain:
        # Probabilities of the actual classes (ignoring abstain logit for max)
        class_probs = probs_np[:, :2]
        pred_class_probs = np.max(class_probs, axis=1)
        pred_class_binary = np.argmax(class_probs, axis=1) # 0 or 1
        
        # Only compute ECE on samples where the model actually made a prediction (not abstained)
        non_abstained = (preds_np != 2)
        if sum(non_abstained) > 0:
            ece_true = y_test[non_abstained] == pred_class_binary[non_abstained]
            ece_prob = pred_class_probs[non_abstained]
            ece = expected_calibration_error(ece_true.astype(float), ece_prob)
        else:
            ece = 0.0
            
    else:
        # Standard baseline model
        pred_probs = np.max(probs_np, axis=1)
        ece_true = (y_test == preds_np).astype(float)
        ece = expected_calibration_error(ece_true, pred_probs)
        
    # Calculate main metrics
    cov = coverage(preds_np)
    sel_risk = selective_risk(y_test, preds_np)
    
    # Calculate general classification metrics (on non-abstained purely for F1 reference)
    non_abstain_mask = (preds_np != 2)
    if sum(non_abstain_mask) > 0:
        filtered_preds = preds_np[non_abstain_mask]
        filtered_true = y_test[non_abstain_mask]
        acc = accuracy_score(filtered_true, filtered_preds)
        f1 = f1_score(filtered_true, filtered_preds, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
        prec = precision_score(filtered_true, filtered_preds, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
        rec = recall_score(filtered_true, filtered_preds, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)

        # Confusion matrix on non-abstained predictions (2x2: legit vs fraud)
        cm = confusion_matrix(filtered_true, filtered_preds, labels=[0, 1])
    else:
        acc = 0.0
        f1 = 0.0
        prec = 0.0
        rec = 0.0
        cm = np.zeros((2, 2), dtype=int)

    # Abstention breakdown: how many of each true class were abstained on
    n_abstained_total = sum(preds_np == 2)
    if has_abstain and n_abstained_total > 0:
        abstained_mask = (preds_np == 2)
        n_abstained_legit = sum(y_test[abstained_mask] == 0)
        n_abstained_fraud = sum(y_test[abstained_mask] == 1)
    else:
        n_abstained_legit = 0
        n_abstained_fraud = 0
        
    return {
        "Accuracy": acc,
        "Coverage": cov,
        "Selective Risk": sel_risk,
        "ECE": ece,
        "F1 Score": f1,
        "Precision (Fraud)": prec,
        "Recall (Fraud)": rec,
        "Abstained Total": int(n_abstained_total),
        "Abstained Legit": int(n_abstained_legit),
        "Abstained Fraud": int(n_abstained_fraud),
        "CM_TN": int(cm[0, 0]),
        "CM_FP": int(cm[0, 1]),
        "CM_FN": int(cm[1, 0]),
        "CM_TP": int(cm[1, 1]),
    }


def run_comprehensive_evaluation():
    """Evaluate all trained models on the test set."""
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {DEVICE}")
    print("=" * 80)
    
    # ----------------------------
    # Load Test Data
    # ----------------------------
    _, _, X_test_np, _, _, y_test_np, _ = load_data("data/creditcard.csv")
    
    # Convert to pure tensors (fixes the `.values` bug from old code)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(DEVICE)
    y_test = y_test_np  # keep as numpy for sklearn metrics
    
    print(f"Test Set Size: {len(y_test)} | Fraud Samples: {sum(y_test)}")
    print("=" * 80)
    
    # ----------------------------
    # Evaluate All Models
    # ----------------------------
    results = []
    
    for name, config in EVAL_MODELS.items():
        resolved_path = resolve_path(config["file"])
        if not os.path.exists(resolved_path):
            print(f"Skipping {name} (File not found: {resolved_path})")
            continue
            
        print(f"Evaluating: {name}...")
        
        # Initialize architecture
        if config["architecture"] == "baseline":
            model = BaselineModel(input_dim=30, dropout=0.0).to(DEVICE)
        else:
            model = AbstentionModel(input_dim=30, dropout=0.0).to(DEVICE)
            
        # Load weights
        model.load_state_dict(torch.load(resolve_path(config["file"]), map_location=DEVICE, weights_only=True))
        
        # Get metrics
        metrics = evaluate_model(model, X_test, y_test, DEVICE, config["has_abstain"])
        metrics["Model Name"] = name
        results.append(metrics)
        
    # ----------------------------
    # Output Results
    # ----------------------------
    if not results:
        print("No trained models found to evaluate.")
        return
        
    df_results = pd.DataFrame(results)
    
    # Save main results (backward compatible)
    main_cols = ["Model Name", "Accuracy", "Coverage", "Selective Risk", "ECE", "F1 Score"]
    df_main = df_results[main_cols]
    
    os.makedirs(resolve_path("results"), exist_ok=True)
    df_main.to_csv(resolve_path("results/final_results.csv"), index=False)
    
    # Save extended results with confusion matrices and per-class metrics
    df_results.to_csv(resolve_path("results/final_results_extended.csv"), index=False)
    
    # Display nicely formatted output
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)
    
    # Print main metrics table
    format_str = "{:<22} | {:>8.4f} | {:>8.4f} | {:>14.4f} | {:>6.4f} | {:>8.4f}"
    print("{:<22} | {:>8} | {:>8} | {:>14} | {:>6} | {:>8}".format(
        "Model Name", "Accuracy", "Coverage", "Selective Risk", "ECE", "F1 Score"
    ))
    print("-" * 80)
    
    for _, row in df_main.iterrows():
        print(format_str.format(
            row["Model Name"],
            row["Accuracy"],
            row["Coverage"],
            row["Selective Risk"],
            row["ECE"],
            row["F1 Score"]
        ))
        
    # Print confusion matrices
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES (on non-abstained predictions)")
    print("=" * 80)
    
    for _, row in df_results.iterrows():
        name = row["Model Name"]
        print(f"\n  {name}:")
        print(f"    Predicted ->   Legit    Fraud")
        print(f"    True Legit:  {int(row['CM_TN']):>8}  {int(row['CM_FP']):>8}")
        print(f"    True Fraud:  {int(row['CM_FN']):>8}  {int(row['CM_TP']):>8}")
        print(f"    Precision(Fraud): {row['Precision (Fraud)']:.4f} | Recall(Fraud): {row['Recall (Fraud)']:.4f}")
        if row["Abstained Total"] > 0:
            print(f"    Abstained: {int(row['Abstained Total'])} total "
                  f"({int(row['Abstained Legit'])} legit, {int(row['Abstained Fraud'])} fraud)")

    print("\n" + "=" * 80)
    print("Results saved to results/final_results.csv")
    print("Extended results saved to results/final_results_extended.csv")

if __name__ == "__main__":
    run_comprehensive_evaluation()