import os
import sys
import torch
import numpy as np
import pandas as pd

# Resolve project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.dataset import load_data, resolve_path
from src.abstention_model import AbstentionModel
from src.mc_dropout import mc_dropout_inference
from src.simulation_loop import simulate_48h_loop, mock_human_analyst

from agents.perception_agent import PerceptionAgent
from agents.uncertainty_agent import UncertaintyAgent
from agents.decision_agent import DecisionAgent
from agents.reflection_agent import ReflectionAgent
from agents.memory_store import MemoryStore

def run_agent_pipeline():
    print("=" * 80)
    print("PHASE 6: MULTI-AGENT GOVERNANCE EXECUTION")
    print("=" * 80)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print("Loading test data...")
    # By default, load_data returns scaled features. 
    _, _, X_test, _, _, y_test, _ = load_data()
    
    # We take a small subset for simulation to keep execution fast
    num_samples = min(500, len(X_test))
    X_test_sub = X_test[:num_samples]
    y_test_sub = y_test[:num_samples]
    
    features_df = pd.DataFrame(X_test_sub, columns=[f"Feature_{i}" for i in range(X_test_sub.shape[1])])
    
    # 2. Load Base Model
    input_dim = X_test_sub.shape[1]
    model = AbstentionModel(input_dim=input_dim).to(DEVICE)
    model_path = resolve_path("abstention_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded trained AbstentionModel from {model_path}.")
    else:
        print("WARNING: trained AbstentionModel not found. Using untrained weights.")

    # 3. Base Inference and Epistemic Uncertainty
    print("Running base model inference and MC Dropout...")
    X_tensor = torch.tensor(X_test_sub, dtype=torch.float32).to(DEVICE)
    mean_probs, epistemic_unc = mc_dropout_inference(model, X_tensor, num_passes=10)
    
    # Aleatoric uncertainty is 1 - max probability
    max_probs = np.max(mean_probs, axis=1)
    aleatoric_unc = 1.0 - max_probs
    base_preds = np.argmax(mean_probs, axis=1)

    # 4. Initialize Agents
    print("Initializing Multi-Agent System...")
    # use_llm=False to avoid TinyLlama overhead unless explicitly needed
    uncertainty_agent = UncertaintyAgent(use_llm=False)
    reflection_agent = ReflectionAgent(use_llm=False)
    
    # uncertainty_dim is 3 (Aleatoric, Epistemic, LLM-verbalized)
    decision_agent = DecisionAgent(feature_dim=input_dim, uncertainty_dim=3, device=DEVICE)
    memory_store = MemoryStore(review_capacity=20)

    # 5. Execute Agentic Uncertainty Fusion and Decision Making
    print("Fusing uncertainty and making decisions via RL Policy...")
    abstained_indices = []
    composite_uncertainties = []
    final_actions = []

    # Features tensor for the Decision Agent
    features_tensor = torch.tensor(X_test_sub, dtype=torch.float32).to(DEVICE)

    for i in range(num_samples):
        # The agent expects a dict of uncertainties
        unc_dict = {
            'aleatoric': float(aleatoric_unc[i]),
            'epistemic': float(epistemic_unc[i]),
            'verbalized': 0.0 # Heuristic/Template fallback outputs 0.0 for LLM slot
        }
        
        # Fuse uncertainty for memory store and reflection
        fused_unc = uncertainty_agent.fuse_uncertainty(
            base_probs=mean_probs[i:i+1],
            epistemic_unc=epistemic_unc[i:i+1],
            features_list=None
        )[0]
        composite_uncertainties.append(fused_unc)
        
        # The DecisionAgent's GatingNetwork expects the raw uncertainty dimensions
        unc_tensor = torch.tensor([[unc_dict['aleatoric'], unc_dict['epistemic'], unc_dict['verbalized']]], dtype=torch.float32).to(DEVICE)
        feat_tensor = features_tensor[i:i+1]
        
        # Policy action: 0 = Predict, 1 = Abstain
        action = decision_agent.select_action(feat_tensor, unc_tensor)
        final_actions.append(action)
        
        if action == 1:
            abstained_indices.append(i)

    print(f"Decision Agent abstained on {len(abstained_indices)} out of {num_samples} transactions.")

    # 6. Store in Memory
    if len(abstained_indices) > 0:
        abstain_df = features_df.iloc[abstained_indices].copy()
        
        # For memory store, we also pass uncertainties and base predictions
        unc_list = [composite_uncertainties[i] for i in abstained_indices]
        pred_list = [base_preds[i] for i in abstained_indices]
        
        memory_store.add_transactions(abstain_df, unc_list, pred_list)
        
        # 7. Simulate 48-Hour Loop
        simulate_48h_loop(memory_store, decision_agent, reflection_agent, mock_human_analyst)
    else:
        print("No transactions were abstained. Skipping simulation loop.")
        
    print("Multi-Agent Governance Pipeline Execution Complete!")

if __name__ == "__main__":
    run_agent_pipeline()
