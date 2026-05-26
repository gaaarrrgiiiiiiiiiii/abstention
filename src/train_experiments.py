import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import csv
import os
import copy
import time
import psutil

from dataset import load_data, FraudDataset, resolve_path
from abstention_model import AbstentionModel
from train_abstention import dac_loss, evaluate_abstention
from seed import set_seed


# ============================================================
# Experiment Configurations
# ============================================================
EXPERIMENTS = {
    1: {"name": "Standard Training",      "grad_accum": False, "mixed_precision": False},
    2: {"name": "Gradient Accumulation",  "grad_accum": True,  "mixed_precision": False},
    3: {"name": "Mixed Precision",        "grad_accum": False, "mixed_precision": True},
    4: {"name": "Combined (GA + MP)",     "grad_accum": True,  "mixed_precision": True},
}


def run_experiment(
    exp_id,
    train_loader,
    val_loader,
    device,
    input_dim,
    use_grad_accum=False,
    use_mixed_precision=False,
    epochs=10,
    lr=0.0001,
    alpha=0.3,
    patience=3,
    accum_steps=4,
    class_weights=None,
    seed=42,
):

    config = EXPERIMENTS[exp_id]

    # Set seed for each experiment for reproducibility
    set_seed(seed)

    print(f"\n{'='*90}")
    print(f"  EXPERIMENT {exp_id}: {config['name']}")
    print(f"  Gradient Accumulation: {use_grad_accum} | Mixed Precision: {use_mixed_precision} | Seed: {seed}")
    print(f"{'='*90}")

    # ----------------------------
    # Model
    # ----------------------------
    model = AbstentionModel(input_dim=input_dim, dropout=0.3).to(device)

    # Transfer baseline weights if available
    if os.path.exists(resolve_path("baseline_model.pth")):
        from baseline_model import BaselineModel
        baseline_state = torch.load(resolve_path("baseline_model.pth"), map_location=device, weights_only=True)
        model_state = model.state_dict()

        for key in baseline_state:
            if key in model_state and baseline_state[key].shape == model_state[key].shape:
                model_state[key] = baseline_state[key]

        model.load_state_dict(model_state)

    # ----------------------------
    # Optimizer + Scheduler
    # ----------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    scaler = GradScaler(device=device.type) if use_mixed_precision else None

    # ----------------------------
    # CSV logging
    # ----------------------------
    os.makedirs(resolve_path("results"), exist_ok=True)

    csv_path = resolve_path(f"results/experiment_{exp_id}_metrics.csv")

    csv_file = open(csv_path, "w", newline="")

    writer = csv.writer(csv_file)

    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "sel_accuracy",
        "coverage",
        "sel_risk",
        "sel_f1",
        "learning_rate",
        "epoch_time_sec",
        "process_memory_mb",
        "throughput_samples_per_sec"
    ])

    # ----------------------------
    # Training loop
    # ----------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Sel Acc':>8} | {'Coverage':>8} | {'Sel Risk':>8} | {'Sel F1':>8} | "
          f"{'Time':>6} | {'Mem(MB)':>8}")
    print("-" * 100)

    for epoch in range(epochs):

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        epoch_start_time = time.time()

        # ----------------------------
        # Training
        # ----------------------------
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):

            x, y = x.to(device), y.to(device)

            if use_mixed_precision:

                with autocast(device_type=device.type):

                    outputs = model(x)

                    loss = dac_loss(outputs, y, alpha, class_weights)

                    if use_grad_accum:
                        loss = loss / accum_steps

                scaler.scale(loss).backward()

                if use_grad_accum:

                    if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):

                        scaler.unscale_(optimizer)

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        scaler.step(optimizer)

                        scaler.update()

                        optimizer.zero_grad()

                else:

                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    scaler.step(optimizer)

                    scaler.update()

                    optimizer.zero_grad()

            else:

                outputs = model(x)

                loss = dac_loss(outputs, y, alpha, class_weights)

                if use_grad_accum:
                    loss = loss / accum_steps

                loss.backward()

                if use_grad_accum:

                    if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        optimizer.step()

                        optimizer.zero_grad()

                else:

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()

                    optimizer.zero_grad()

            train_loss += loss.item() * (accum_steps if use_grad_accum else 1)

        train_loss /= len(train_loader)

        # ----------------------------
        # Validation
        # ----------------------------
        val_loss, sel_acc, coverage, sel_risk, sel_f1 = evaluate_abstention(
            model,
            val_loader,
            device,
            alpha,
            class_weights
        )

        # ----------------------------
        # Hardware Metrics
        # ----------------------------
        epoch_time = time.time() - epoch_start_time

        if device.type == "cuda":
            gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            gpu_memory = psutil.Process().memory_info().rss / (1024 ** 2)

        num_samples = len(train_loader.dataset)

        throughput = num_samples / epoch_time

        # ----------------------------
        # LR Scheduler
        # ----------------------------
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(val_loss)

        # ----------------------------
        # Log CSV
        # ----------------------------
        writer.writerow([
            epoch + 1,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{sel_acc:.6f}",
            f"{coverage:.6f}",
            f"{sel_risk:.6f}",
            f"{sel_f1:.6f}",
            f"{current_lr:.8f}",
            f"{epoch_time:.3f}",
            f"{gpu_memory:.2f}",
            f"{throughput:.2f}"
        ])

        print(f"{epoch+1:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
              f"{sel_acc:>8.4f} | {coverage:>8.4f} | {sel_risk:>8.4f} | {sel_f1:>8.4f} | "
              f"{epoch_time:>6.2f}s | {gpu_memory:>6.1f}MB")

        # ----------------------------
        # Early stopping
        # ----------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            best_state = copy.deepcopy(model.state_dict())

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(f"  Early stopping at epoch {epoch+1}")

                break

    csv_file.close()

    model_path = resolve_path(f"results/experiment_{exp_id}_model.pth")

    torch.save(best_state, model_path)

    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Saved to: {csv_path}, {model_path}")

    return best_val_loss


# ============================================================
# Run All Experiments
# ============================================================
def run_all_experiments():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {DEVICE}")

    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_data()

    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)

    # Compute class weights to handle imbalance
    n_legit = sum(y_train == 0)
    n_fraud = sum(y_train == 1)
    fraud_weight = min(n_legit / n_fraud, 50.0)
    class_weights = torch.tensor([1.0, fraud_weight], dtype=torch.float32).to(DEVICE)
    print(f"Class weights: [1.0, {fraud_weight:.1f}]")

    results = {}

    for exp_id, config in EXPERIMENTS.items():

        batch_size = 64 if config["grad_accum"] else 256

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        input_dim = X_train.shape[1]

        best_loss = run_experiment(
            exp_id=exp_id,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            input_dim=input_dim,
            use_grad_accum=config["grad_accum"],
            use_mixed_precision=config["mixed_precision"],
            class_weights=class_weights,
        )

        results[exp_id] = best_loss

    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"{'Exp':>4} | {'Configuration':<30} | {'Best Val Loss':>12}")
    print("-" * 55)

    for exp_id, loss in results.items():
        print(f"{exp_id:>4} | {EXPERIMENTS[exp_id]['name']:<30} | {loss:>12.6f}")

    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()