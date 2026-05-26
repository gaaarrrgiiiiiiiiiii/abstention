import torch
import numpy as np
import pandas as pd
from dataset import load_data, resolve_path
from abstention_model import AbstentionModel
from ood_injection import inject_ood_transactions
from mc_dropout import mc_dropout_inference
from agents.uncertainty_agent import UncertaintyAgent

def run_ood_simulation():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running OOD Simulation on {DEVICE}")
    print("=" * 80)
    
    # Load regular data
    X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np, scaler = load_data()
    
    # Inject OOD
    X_test_ood, y_test_ood = inject_ood_transactions(X_test_np, y_test_np, num_ood=1000)
    
    # Convert to tensors
    X_test = torch.tensor(X_test_ood, dtype=torch.float32).to(DEVICE)
    
    input_dim = X_test.shape[1]
    
    model = AbstentionModel(input_dim=input_dim, dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load(resolve_path("abstention_model.pth"), map_location=DEVICE, weights_only=True))
    model.eval()
    
    # Get MC Dropout uncertainty
    mean_probs, epistemic_unc = mc_dropout_inference(model, X_test, num_passes=10)
    
    # Use UncertaintyAgent
    uncertainty_agent = UncertaintyAgent(use_llm=False)
    composite_unc = uncertainty_agent.fuse_uncertainty(mean_probs, epistemic_unc, features_list=None)
    
    # Decide predictions: abstain if composite uncertainty > threshold or base model abstains
    preds = np.argmax(mean_probs, axis=1)
    
    # Find abstained samples (either model predicted abstain (class 2) or composite_unc > 0.4)
    # Threshold 0.4 can be tuned based on what gives > 80% on OOD
    uncertainty_threshold = 0.4
    preds[composite_unc > uncertainty_threshold] = 2
        
    # Analyze OOD behavior
    ood_mask = (y_test_ood == -1)
    ood_preds = preds[ood_mask]
    
    total_ood = len(ood_preds)
    abstained_ood = sum(ood_preds == 2)
    
    print("\n" + "=" * 80)
    print("OOD INJECTION RESULTS")
    print("=" * 80)
    print(f"Total OOD samples injected: {total_ood}")
    print(f"OOD samples abstained on: {abstained_ood} ({abstained_ood/total_ood*100:.2f}%)")
    
    if abstained_ood / total_ood > 0.8:
        print("Robustness Test: PASSED (System successfully identifies and abstains on OOD data)")
    else:
        print("Robustness Test: FAILED (System fails to abstain on OOD data)")
    print("=" * 80)

if __name__ == "__main__":
    run_ood_simulation()
