import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os
import joblib

from dataset import load_data, FraudDataset, resolve_path
from baseline_model import BaselineModel
from seed import set_seed


def evaluate_metrics(model, loader, criterion, device):
    """Evaluate model on a DataLoader and return loss + classification metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, acc, prec, rec, f1


def train_baseline(seed=42):
    """Train the baseline model with early stopping, LR scheduling, and per-epoch metrics."""

    # ----------------------------
    # Reproducibility
    # ----------------------------
    set_seed(seed)

    # ----------------------------
    # Configuration
    # ----------------------------
    BATCH_SIZE = 256
    EPOCHS = 60
    LR = 0.0001
    PATIENCE = 10          # early stopping patience
    DROPOUT = 0.3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")
    print("=" * 80)

    # ----------------------------
    # Load Data
    # ----------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data()

    # Persist the fitted scaler for API serving (avoids re-reading CSV)
    scaler_path = resolve_path("scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Fraud in train: {sum(y_train)} | Fraud in val: {sum(y_val)}")
    print("=" * 80)

    # ----------------------------
    # Model
    # ----------------------------
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")
    model = BaselineModel(input_dim=input_dim, dropout=DROPOUT).to(DEVICE)

    # ----------------------------
    # Class Weights (handle imbalance)
    # ----------------------------
    n_legit = sum(y_train == 0)
    n_fraud = sum(y_train == 1)
    weight_fraud = n_legit / n_fraud  # ~577
    # Cap at 100 — extreme weights (500+) cause noisy val loss on imbalanced data
    weights = torch.tensor([1.0, min(weight_fraud, 100.0)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Class weights: [1.0, {weights[1].item():.1f}]")

    # ----------------------------
    # Optimizer + Scheduler
    # ----------------------------
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ----------------------------
    # Metrics CSV
    # ----------------------------
    os.makedirs(resolve_path("results"), exist_ok=True)
    csv_path = resolve_path("results/baseline_training_metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "epoch", "train_loss", "val_loss",
        "val_accuracy", "val_precision", "val_recall", "val_f1",
        "learning_rate"
    ])

    # ----------------------------
    # Training Loop with Early Stopping
    # ----------------------------
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8} | {'LR':>10}")
    print("-" * 90)

    for epoch in range(EPOCHS):

        # ---- Training ----
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation Metrics ----
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_metrics(
            model, val_loader, criterion, DEVICE
        )

        # ---- LR Scheduling ----
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        # ---- Log Metrics ----
        writer.writerow([
            epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{val_acc:.6f}", f"{val_prec:.6f}", f"{val_rec:.6f}", f"{val_f1:.6f}",
            f"{current_lr:.8f}"
        ])

        print(f"{epoch+1:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
              f"{val_acc:>8.4f} | {val_prec:>9.4f} | {val_rec:>8.4f} | {val_f1:>8.4f} | {current_lr:>10.6f}")

        # ---- Early Stopping + Best Model Checkpoint ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), resolve_path("baseline_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    csv_file.close()

    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Metrics saved to: {csv_path}")
    print(f"Best model saved to: {resolve_path('baseline_model.pth')}")
    print("=" * 80)

    return best_val_loss


if __name__ == "__main__":
    train_baseline()