import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score

from dataset import load_data
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
    """Run evaluation for a single model and compute all metrics."""
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
    else:
        acc = 0.0
        f1 = 0.0
        
    return {
        "Accuracy": acc,
        "Coverage": cov,
        "Selective Risk": sel_risk,
        "ECE": ece,
        "F1 Score": f1
    }


def run_comprehensive_evaluation():
    """Evaluate all trained models on the test set."""
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {DEVICE}")
    print("=" * 80)
    
    # ----------------------------
    # Load Test Data
    # ----------------------------
    _, _, X_test_np, _, _, y_test_np = load_data("data/creditcard.csv")
    
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
        if not os.path.exists(config["file"]):
            print(f"Skipping {name} (File not found: {config['file']})")
            continue
            
        print(f"Evaluating: {name}...")
        
        # Initialize architecture
        if config["architecture"] == "baseline":
            model = BaselineModel(input_dim=30, dropout=0.0).to(DEVICE)
        else:
            model = AbstentionModel(input_dim=30, dropout=0.0).to(DEVICE)
            
        # Load weights
        model.load_state_dict(torch.load(config["file"], map_location=DEVICE, weights_only=True))
        
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
    
    # Reorder columns
    df_results = df_results[["Model Name", "Accuracy", "Coverage", "Selective Risk", "ECE", "F1 Score"]]
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/final_results.csv", index=False)
    
    # Display nicely formatting
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)
    
    # Print formatted table
    format_str = "{:<22} | {:>8.4f} | {:>8.4f} | {:>14.4f} | {:>6.4f} | {:>8.4f}"
    print("{:<22} | {:>8} | {:>8} | {:>14} | {:>6} | {:>8}".format(
        "Model Name", "Accuracy", "Coverage", "Selective Risk", "ECE", "F1 Score"
    ))
    print("-" * 80)
    
    for _, row in df_results.iterrows():
        print(format_str.format(
            row["Model Name"],
            row["Accuracy"],
            row["Coverage"],
            row["Selective Risk"],
            row["ECE"],
            row["F1 Score"]
        ))
        
    print("=" * 80)
    print("Results saved to results/final_results.csv")

if __name__ == "__main__":
    run_comprehensive_evaluation()