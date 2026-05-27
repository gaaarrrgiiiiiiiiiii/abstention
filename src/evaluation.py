import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

from dataset import load_data, resolve_path
from baseline_model import BaselineModel
from abstention_model import AbstentionModel
from metrics import classification_metrics, coverage, selective_risk, expected_calibration_error
from conformal import apply_conformal_abstention
from mc_dropout import mc_dropout_inference

import sys
# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.decision_agent import DecisionAgent
from agents.reward import calculate_reward

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


def evaluate_model(model, X_test, y_test, X_val, y_val, device, has_abstain):
    """Run evaluation for a single model and compute all metrics including confusion matrix and conformal guarantees."""
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
        
    metrics = {
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

    # ── AUROC / AUPR / Brier Score (on non-abstained samples only) ────────────
    # These are threshold-free metrics essential for imbalanced fraud detection.
    if has_abstain:
        fraud_probs = probs_np[:, 1] / (probs_np[:, 0] + probs_np[:, 1] + 1e-8)
    else:
        fraud_probs = probs_np[:, 1]

    mask_eval = non_abstain_mask  # already computed above
    if mask_eval.sum() > 0 and len(np.unique(y_test[mask_eval])) > 1:
        try:
            auroc = roc_auc_score(y_test[mask_eval], fraud_probs[mask_eval])
        except Exception:
            auroc = 0.0
        try:
            aupr  = average_precision_score(y_test[mask_eval], fraud_probs[mask_eval])
        except Exception:
            aupr  = 0.0
        # Brier score: mean squared error between predicted fraud prob and binary label
        brier = float(np.mean((fraud_probs[mask_eval] - y_test[mask_eval].astype(np.float32)) ** 2))
    else:
        auroc = aupr = brier = 0.0

    metrics["AUROC"]       = float(auroc)
    metrics["AUPR"]        = float(aupr)
    metrics["Brier Score"] = float(brier)
    
    # ----------------------------
    # Conformal Prediction Eval
    # ----------------------------
    # Calibrate on validation set, then evaluate coverage guarantee on test set.
    # For abstention models, we use class_probs (only classes 0 & 1) so conformal
    # prediction operates in the standard 2-class space.
    try:
        with torch.no_grad():
            val_outputs = model(X_val)
            val_probs = torch.softmax(val_outputs, dim=1).cpu().numpy()
        
        if has_abstain:
            # Use only class 0 and 1 probabilities for conformal prediction
            val_class_probs = val_probs[:, :2]
            val_class_probs = val_class_probs / val_class_probs.sum(axis=1, keepdims=True)  # renormalize
            test_class_probs = probs_np[:, :2]
            test_class_probs = test_class_probs / test_class_probs.sum(axis=1, keepdims=True)
            pred_sets, cp = apply_conformal_abstention(val_class_probs, y_val, test_class_probs, alpha=0.1)
        else:
            pred_sets, cp = apply_conformal_abstention(val_probs, y_val, probs_np, alpha=0.1)
            
        conf_coverage, conf_avg_set_size = cp.evaluate_coverage(pred_sets, y_test)
        metrics["Conformal Coverage"] = conf_coverage
        metrics["Conformal Avg Set Size"] = conf_avg_set_size
    except Exception as e:
        print(f"  Warning: Conformal prediction failed: {e}")
        metrics["Conformal Coverage"] = 0.0
        metrics["Conformal Avg Set Size"] = 0.0
    
    return metrics


def evaluate_msp_baseline(model, X_test, y_test, device, target_coverage=0.9962):
    """
    Evaluates a Maximum Softmax Probability (MSP) baseline by thresholding
    confidence on the standard baseline model to match a target coverage.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
    # Standard 2-class softmax probabilities
    pred_probs = np.max(probs, axis=1)
    base_preds = np.argmax(probs, axis=1)
    
    # Sort probabilities to find threshold matching target coverage
    num_samples = len(X_test)
    num_to_abstain = int(round(num_samples * (1.0 - target_coverage)))
    
    if num_to_abstain > 0:
        # Find the threshold: the (1 - target_coverage) quantile of pred_probs
        threshold = np.partition(pred_probs, num_to_abstain)[num_to_abstain]
    else:
        threshold = 0.0
        
    preds_np = np.copy(base_preds)
    preds_np[pred_probs < threshold] = 2 # Abstain
    
    cov = coverage(preds_np)
    sel_risk = selective_risk(y_test, preds_np)
    
    non_abstain_mask = (preds_np != 2)
    if sum(non_abstain_mask) > 0:
        filtered_preds = preds_np[non_abstain_mask]
        filtered_true = y_test[non_abstain_mask]
        acc = accuracy_score(filtered_true, filtered_preds)
        f1 = f1_score(filtered_true, filtered_preds, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
        prec = precision_score(filtered_true, filtered_preds, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
        rec = recall_score(filtered_true, filtered_preds, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
        cm = confusion_matrix(filtered_true, filtered_preds, labels=[0, 1])
    else:
        acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0
        cm = np.zeros((2, 2), dtype=int)
        
    n_abstained_total = sum(preds_np == 2)
    n_abstained_legit = sum(y_test[preds_np == 2] == 0) if n_abstained_total > 0 else 0
    n_abstained_fraud = sum(y_test[preds_np == 2] == 1) if n_abstained_total > 0 else 0
    
    # ECE on non-abstained
    if sum(non_abstain_mask) > 0:
        ece_true = y_test[non_abstain_mask] == base_preds[non_abstain_mask]
        ece_prob = pred_probs[non_abstain_mask]
        ece = expected_calibration_error(ece_true.astype(float), ece_prob)
    else:
        ece = 0.0
        
    metrics = {
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
        "Conformal Coverage": 0.0,
        "Conformal Avg Set Size": 0.0,
        "Model Name": f"MSP Baseline (Cov={cov*100:.2f}%)"
    }
    return metrics


def train_decision_agent(model, X_train_val, y_train_val, device, epochs=5, lr=1e-3):
    """
    Trains the RL Decision Agent policy on a training/validation subset.
    """
    print("Training RL Decision Agent policy...")
    input_dim = X_train_val.shape[1]
    agent = DecisionAgent(feature_dim=input_dim, uncertainty_dim=3, lr=lr, device=device)
    
    # Run MC Dropout once on the training subset to get base probs and uncertainties
    mean_probs, epistemic_unc = mc_dropout_inference(model, X_train_val, num_passes=10)
    max_probs = np.max(mean_probs, axis=1)
    aleatoric_unc = 1.0 - max_probs
    base_preds = np.argmax(mean_probs, axis=1)
    
    num_samples = len(X_train_val)
    
    for epoch in range(epochs):
        agent.policy_net.train()
        epoch_rewards = 0

        for i in range(num_samples):
            feat = X_train_val[i:i+1]

            # Heuristic verbalized uncertainty: margin from decision boundary
            # Replaces the old hardcoded 0.5 placeholder.
            base_prob_i = float(max_probs[i])
            epi_i       = float(epistemic_unc[i])
            margin      = abs(base_prob_i - 0.5)
            margin_score = 1.0 - float(np.exp(-6 * margin))
            epi_penalty  = float(np.clip(epi_i * 4, 0, 1))
            verb_unc_i   = float(np.clip(1.0 - (margin_score - 0.5 * epi_penalty), 0.05, 0.95))

            unc = torch.tensor(
                [[aleatoric_unc[i], epistemic_unc[i], verb_unc_i]],
                dtype=torch.float32
            ).to(device)
            
            action = agent.select_action(feat, unc)
            
            true_label = y_train_val[i]
            pred_class = base_preds[i]
            
            if action == 1:
                # Abstain
                reward = calculate_reward(true_label, predicted_label=2)
            else:
                # Predict
                reward = calculate_reward(true_label, predicted_label=pred_class)
                
            agent.store_reward(reward)
            epoch_rewards += reward
            
        loss = agent.update_policy()
        print(f"  Epoch {epoch+1}/{epochs} | Policy Loss: {loss:.4f} | Total Reward: {epoch_rewards:.2f}")
        
    return agent


def evaluate_decision_agent(agent, model, X_test, y_test, device):
    """
    Evaluates the trained RL Decision Agent policy on the test set.
    """
    agent.policy_net.eval()
    
    mean_probs, epistemic_unc = mc_dropout_inference(model, X_test, num_passes=10)
    max_probs = np.max(mean_probs, axis=1)
    aleatoric_unc = 1.0 - max_probs
    base_preds = np.argmax(mean_probs, axis=1)
    
    num_samples = len(X_test)
    preds_np = np.zeros(num_samples, dtype=int)
    
    with torch.no_grad():
        for i in range(num_samples):
            feat = X_test[i:i+1]

            # Same heuristic as training
            base_prob_i  = float(max_probs[i])
            epi_i        = float(epistemic_unc[i])
            margin       = abs(base_prob_i - 0.5)
            margin_score = 1.0 - float(np.exp(-6 * margin))
            epi_penalty  = float(np.clip(epi_i * 4, 0, 1))
            verb_unc_i   = float(np.clip(1.0 - (margin_score - 0.5 * epi_penalty), 0.05, 0.95))

            unc = torch.tensor(
                [[aleatoric_unc[i], epistemic_unc[i], verb_unc_i]],
                dtype=torch.float32
            ).to(device)
            
            abstain_prob = agent.policy_net(feat, unc).item()
            action = 1 if abstain_prob > 0.5 else 0
            
            if action == 1:
                preds_np[i] = 2 # Abstain
            else:
                preds_np[i] = base_preds[i] # Predict base class
                
    cov = coverage(preds_np)
    sel_risk = selective_risk(y_test, preds_np)
    
    non_abstain_mask = (preds_np != 2)
    if sum(non_abstain_mask) > 0:
        filtered_preds = preds_np[non_abstain_mask]
        filtered_true = y_test[non_abstain_mask]
        acc = accuracy_score(filtered_true, filtered_preds)
        f1 = f1_score(filtered_true, filtered_preds, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
        prec = precision_score(filtered_true, filtered_preds, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
        rec = recall_score(filtered_true, filtered_preds, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
        cm = confusion_matrix(filtered_true, filtered_preds, labels=[0, 1])
    else:
        acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0
        cm = np.zeros((2, 2), dtype=int)
        
    n_abstained_total = sum(preds_np == 2)
    n_abstained_legit = sum(y_test[preds_np == 2] == 0) if n_abstained_total > 0 else 0
    n_abstained_fraud = sum(y_test[preds_np == 2] == 1) if n_abstained_total > 0 else 0
    
    class_probs = mean_probs[:, :2]
    pred_class_probs = np.max(class_probs, axis=1)
    pred_class_binary = np.argmax(class_probs, axis=1)
    if sum(non_abstain_mask) > 0:
        ece_true = y_test[non_abstain_mask] == pred_class_binary[non_abstain_mask]
        ece_prob = pred_class_probs[non_abstain_mask]
        ece = expected_calibration_error(ece_true.astype(float), ece_prob)
    else:
        ece = 0.0
        
    metrics = {
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
        "Conformal Coverage": 0.0,
        "Conformal Avg Set Size": 0.0,
        "Model Name": "Decision Agent (RL)"
    }
    return metrics


def run_comprehensive_evaluation():
    """Evaluate all trained models on the test set."""
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {DEVICE}")
    print("=" * 80)
    
    # ----------------------------
    # Load Test Data
    # ----------------------------
    _, X_val_np, X_test_np, _, y_val_np, y_test_np, _ = load_data()
    
    # Convert to pure tensors (fixes the `.values` bug from old code)
    X_val = torch.tensor(X_val_np, dtype=torch.float32).to(DEVICE)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(DEVICE)
    y_test = y_test_np  # keep as numpy for sklearn metrics
    y_val = y_val_np
    
    input_dim = X_test.shape[1]
    
    print(f"Test Set Size: {len(y_test)} | Fraud Samples: {sum(y_test)} | Input Dim: {input_dim}")
    print("=" * 80)
    
    # ----------------------------
    # Evaluate All Models
    # ----------------------------
    results = []
    
    baseline_model_loaded = None
    abstention_coverage_target = 0.9962
    
    for name, config in EVAL_MODELS.items():
        resolved_path = resolve_path(config["file"])
        if not os.path.exists(resolved_path):
            print(f"Skipping {name} (File not found: {resolved_path})")
            continue
            
        print(f"Evaluating: {name}...")
        
        # Initialize architecture
        if config["architecture"] == "baseline":
            model = BaselineModel(input_dim=input_dim, dropout=0.0).to(DEVICE)
        else:
            model = AbstentionModel(input_dim=input_dim, dropout=0.0).to(DEVICE)
            
        # Load weights
        model.load_state_dict(torch.load(resolve_path(config["file"]), map_location=DEVICE, weights_only=True))
        
        # Get metrics
        metrics = evaluate_model(model, X_test, y_test, X_val, y_val, DEVICE, config["has_abstain"])
        metrics["Model Name"] = name
        results.append(metrics)
        
        # Keep references for dynamic baseline evaluation
        if name == "Baseline":
            baseline_model_loaded = model
        if name == "Abstention (Phase 2)":
            abstention_coverage_target = metrics["Coverage"]

    # ----------------------------
    # Evaluate MSP Baseline and RL Decision Agent
    # ----------------------------
    if baseline_model_loaded is not None:
        print(f"Evaluating MSP Baseline (matching coverage {abstention_coverage_target*100:.2f}%)...")
        msp_metrics = evaluate_msp_baseline(baseline_model_loaded, X_test, y_test, DEVICE, target_coverage=abstention_coverage_target)
        results.append(msp_metrics)
        
        # Train and evaluate RL Decision Agent
        train_val_size = min(10000, len(X_val))
        X_val_sub = X_val[:train_val_size]
        y_val_sub = y_val[:train_val_size]
        
        # We need the base model (Abstention (Phase 2)) to generate uncertainties for RL Gating
        abstention_model_loaded = None
        for name, config in EVAL_MODELS.items():
            if name == "Abstention (Phase 2)" and os.path.exists(resolve_path(config["file"])):
                abstention_model_loaded = AbstentionModel(input_dim=input_dim, dropout=0.0).to(DEVICE)
                abstention_model_loaded.load_state_dict(torch.load(resolve_path(config["file"]), map_location=DEVICE, weights_only=True))
                break
                
        if abstention_model_loaded is not None:
            rl_agent = train_decision_agent(abstention_model_loaded, X_val_sub, y_val_sub, DEVICE, epochs=5)
            rl_metrics = evaluate_decision_agent(rl_agent, abstention_model_loaded, X_test, y_test, DEVICE)
            results.append(rl_metrics)

    # ----------------------------
    # Output Results
    # ----------------------------
    if not results:
        print("No trained models found to evaluate.")
        return
        
    df_results = pd.DataFrame(results)
    
    # Save main results (backward compatible)
    main_cols = ["Model Name", "Accuracy", "Coverage", "Selective Risk", "ECE", "F1 Score", "Conformal Coverage", "Conformal Avg Set Size"]
    df_main = df_results[main_cols]
    
    os.makedirs(resolve_path("results"), exist_ok=True)
    df_main.to_csv(resolve_path("results/final_results.csv"), index=False)
    
    # Save extended results with confusion matrices and per-class metrics
    df_results.to_csv(resolve_path("results/final_results_extended.csv"), index=False)
    
    # Display nicely formatted output
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)

    hdr_fmt = "{:<22} | {:>8} | {:>8} | {:>14} | {:>6} | {:>8} | {:>7} | {:>7} | {:>7}"
    row_fmt = "{:<22} | {:>8.4f} | {:>8.4f} | {:>14.4f} | {:>6.4f} | {:>8.4f} | {:>7.4f} | {:>7.4f} | {:>7.4f}"
    print(hdr_fmt.format(
        "Model Name", "Accuracy", "Coverage", "Selective Risk",
        "ECE", "F1 Score", "AUROC", "AUPR", "Brier"
    ))
    print("-" * 110)

    for _, row in df_results.iterrows():
        print(row_fmt.format(
            str(row["Model Name"])[:22],
            row.get("Accuracy", 0.0),
            row.get("Coverage", 0.0),
            row.get("Selective Risk", 0.0),
            row.get("ECE", 0.0),
            row.get("F1 Score", 0.0),
            row.get("AUROC", 0.0),
            row.get("AUPR", 0.0),
            row.get("Brier Score", 0.0),
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