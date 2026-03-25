import torch
import numpy as np
from dataset import load_data
from abstention_model import AbstentionModel
from evaluation import EVAL_MODELS
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, X_test_np, _, _, y_test_np = load_data("data/creditcard.csv")

X_test = torch.tensor(X_test_np, dtype=torch.float32).to(DEVICE)
y_test = y_test_np

for name in ["Exp 1 (Standard)", "Exp 2 (Grad Accum)", "Exp 3 (Mixed Prec)"]:
    print(f"\n--- {name} ---")
    config = EVAL_MODELS[name]
    model = AbstentionModel(input_dim=30, dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(config["file"], map_location=DEVICE, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1).cpu().numpy()
        
    non_abstain_mask = (preds != 2)
    filtered_preds = preds[non_abstain_mask]
    filtered_true = y_test[non_abstain_mask]
    
    print(f"Total abstained: {sum(preds == 2)}")
    print(f"Filtered true positives (fraud): {sum(filtered_true == 1)}")
    print(f"Filtered false positives (predicted fraud but legit): {sum((filtered_preds == 1) & (filtered_true == 0))}")
    print(f"Filtered false negatives (predicted legit but fraud): {sum((filtered_preds == 0) & (filtered_true == 1))}")
    
    # Let's see confusion matrix
    cm = confusion_matrix(filtered_true, filtered_preds, labels=[0, 1])
    print(f"Confusion Matrix on non-abstained:\n{cm}")
    
