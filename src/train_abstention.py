import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os

from dataset import load_data, FraudDataset, resolve_path
from abstention_model import AbstentionModel
from baseline_model import BaselineModel
from seed import set_seed


# ============================================================
# DAC Loss (Deep Abstaining Classifier)
# ============================================================
def dac_loss(outputs, targets, alpha=0.3, class_weights=None):
    """
    Deep Abstaining Classifier loss with class weighting.

    The model outputs 3 logits: [class_0, class_1, abstain].
    - For non-abstained samples: standard cross-entropy on classes 0,1
    - For abstained samples: penalty of alpha

    alpha controls the abstention cost:
    - Lower alpha = model abstains more freely
    - Higher alpha = model is punished more for abstaining
    
    class_weights: tensor of shape [2] giving weights for class 0 and class 1.
                   Higher weight on class 1 forces model to learn fraud predictions
                   instead of routing everything to abstain.
    """
    probs = torch.softmax(outputs, dim=1)

    # Probability assigned to abstain class
    p_abstain = probs[:, 2]

    # Probability assigned to true class (for classes 0 and 1)
    p_true = probs[range(len(targets)), targets]

    # DAC loss: -log(p_true + p_abstain) + alpha * p_abstain
    loss = -torch.log(p_true + p_abstain + 1e-8) + alpha * p_abstain

    # Apply per-sample class weighting so fraud samples get stronger gradients
    if class_weights is not None:
        sample_weights = class_weights[targets]
        loss = loss * sample_weights

    return loss.mean()


# ============================================================
# Evaluation with Abstention Metrics
# ============================================================
def evaluate_abstention(model, loader, device, alpha=0.3, class_weights=None):
    """Evaluate abstention model: returns loss, accuracy, coverage, selective_risk, F1."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            loss = dac_loss(outputs, y, alpha, class_weights=class_weights)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)

    # Coverage: fraction of non-abstained predictions
    non_abstain_mask = [p != 2 for p in all_preds]
    coverage = sum(non_abstain_mask) / len(all_preds)

    # Selective accuracy and risk on non-abstained samples only
    if sum(non_abstain_mask) > 0:
        filtered_preds = [p for p, m in zip(all_preds, non_abstain_mask) if m]
        filtered_labels = [l for l, m in zip(all_labels, non_abstain_mask) if m]
        sel_acc = accuracy_score(filtered_labels, filtered_preds)
        sel_risk = 1.0 - sel_acc
        sel_f1 = f1_score(filtered_labels, filtered_preds, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
    else:
        sel_acc = 0.0
        sel_risk = 1.0
        sel_f1 = 0.0

    return avg_loss, sel_acc, coverage, sel_risk, sel_f1


# ============================================================
# Main Training Function
# ============================================================
def train_abstention(seed=42):
    """Train abstention model with DAC loss, initialized from baseline weights."""

    # ----------------------------
    # Reproducibility
    # ----------------------------
    set_seed(seed)

    # ----------------------------
    # Configuration
    # ----------------------------
    BATCH_SIZE = 256
    EPOCHS = 30
    LR = 0.0001
    PATIENCE = 5
    DROPOUT = 0.3
    ALPHA = 0.3           # abstention penalty
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")
    print(f"Abstention penalty (alpha): {ALPHA}")
    print("=" * 90)

    # ----------------------------
    # Load Data
    # ----------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_data()

    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print("=" * 90)

    # Compute class weights to handle imbalance (boost fraud class)
    n_legit = sum(y_train == 0)
    n_fraud = sum(y_train == 1)
    fraud_weight = min(n_legit / n_fraud, 50.0)  # capped at 50 for DAC loss stability
    class_weights = torch.tensor([1.0, fraud_weight], dtype=torch.float32).to(DEVICE)
    print(f"Class weights for DAC loss: [1.0, {fraud_weight:.1f}]")

    # ----------------------------
    # Model: Initialize from baseline (transfer learning)
    # ----------------------------
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")
    model = AbstentionModel(input_dim=input_dim, dropout=DROPOUT).to(DEVICE)

    # Load pretrained baseline weights for shared layers
    if os.path.exists(resolve_path("baseline_model.pth")):
        baseline_state = torch.load(resolve_path("baseline_model.pth"), map_location=DEVICE, weights_only=True)
        # Transfer weights from baseline layers (skip the final output layer)
        model_state = model.state_dict()
        transferred = 0
        for key in baseline_state:
            # Match all layers except the final linear layer (net.8.weight, net.8.bias)
            if key in model_state and baseline_state[key].shape == model_state[key].shape:
                model_state[key] = baseline_state[key]
                transferred += 1
        model.load_state_dict(model_state)
        print(f"Transferred {transferred} parameter tensors from baseline model")
    else:
        print("WARNING: No baseline_model.pth found, training from scratch")

    # ----------------------------
    # Optimizer + Scheduler
    # ----------------------------
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Initialize Mixed Precision GradScaler
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == 'cuda'))

    # ----------------------------
    # Metrics CSV
    # ----------------------------
    os.makedirs(resolve_path("results"), exist_ok=True)
    csv_path = resolve_path("results/abstention_training_metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "epoch", "train_loss", "val_loss",
        "sel_accuracy", "coverage", "sel_risk", "sel_f1",
        "learning_rate"
    ])

    # ----------------------------
    # Training Loop
    # ----------------------------
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Sel Acc':>8} | {'Coverage':>8} | {'Sel Risk':>8} | {'Sel F1':>8} | {'LR':>10}")
    print("-" * 95)

    for epoch in range(EPOCHS):

        # ---- Training ----
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass
            with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                outputs = model(x)
                loss = dac_loss(outputs, y, ALPHA, class_weights)
                
            # Mixed Precision Backward Pass
            scaler.scale(loss).backward()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step and update scaler
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation ----
        val_loss, sel_acc, coverage, sel_risk, sel_f1 = evaluate_abstention(
            model, val_loader, DEVICE, ALPHA, class_weights
        )

        # ---- LR Scheduling ----
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        # ---- Log ----
        writer.writerow([
            epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{sel_acc:.6f}", f"{coverage:.6f}", f"{sel_risk:.6f}", f"{sel_f1:.6f}",
            f"{current_lr:.8f}"
        ])

        print(f"{epoch+1:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
              f"{sel_acc:>8.4f} | {coverage:>8.4f} | {sel_risk:>8.4f} | {sel_f1:>8.4f} | {current_lr:>10.6f}")

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), resolve_path("abstention_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    csv_file.close()

    print("=" * 90)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Metrics saved to: {csv_path}")
    print(f"Best model saved to: {resolve_path('abstention_model.pth')}")
    print("=" * 90)

    return best_val_loss


if __name__ == "__main__":
    train_abstention()