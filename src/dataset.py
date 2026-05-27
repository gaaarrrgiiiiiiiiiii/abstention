"""
dataset.py — IEEE-CIS Fraud Detection data loader with caching,
Perception Agent enrichment, and support for both stratified and
temporal data splits.

Temporal split is now the default for publication-quality evaluation.
A randomised stratified split is available via split_mode='stratified'
for backward compatibility and ablation studies.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import sys
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Resolve project root (parent of src/) regardless of CWD
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from agents.perception_agent import PerceptionAgent


def resolve_path(relative_path):
    """Resolve a project-relative path to an absolute path."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


class FraudDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Temporal split helper
# ─────────────────────────────────────────────────────────────────────────────

def _temporal_split(X: np.ndarray, y: np.ndarray, dt_values: np.ndarray,
                    train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Chronological split: train on the earliest `train_frac` of transactions,
    validate on the next `val_frac`, and test on the remainder.

    This prevents lookahead bias and simulates a realistic deployment scenario
    where the model is trained on past data and tested on future data.

    Args:
        X           : Feature matrix (N, D)
        y           : Labels (N,)
        dt_values   : TransactionDT values (N,) — seconds since reference epoch
        train_frac  : Fraction of transactions for training
        val_frac    : Fraction for validation

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
        and a dict with the DT boundary values for logging.
    """
    sort_idx = np.argsort(dt_values)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    dt_sorted = dt_values[sort_idx]

    n = len(X_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train = X_sorted[:train_end]
    y_train = y_sorted[:train_end]

    X_val = X_sorted[train_end:val_end]
    y_val = y_sorted[train_end:val_end]

    X_test = X_sorted[val_end:]
    y_test = y_sorted[val_end:]

    boundaries = {
        "train_dt_min": int(dt_sorted[0]),
        "train_dt_max": int(dt_sorted[train_end - 1]),
        "val_dt_min":   int(dt_sorted[train_end]),
        "val_dt_max":   int(dt_sorted[val_end - 1]),
        "test_dt_min":  int(dt_sorted[val_end]),
        "test_dt_max":  int(dt_sorted[-1]),
        "n_fraud_train": int(y_train.sum()),
        "n_fraud_val":   int(y_val.sum()),
        "n_fraud_test":  int(y_test.sum()),
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, boundaries


# ─────────────────────────────────────────────────────────────────────────────
# Main data loading function
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    path: str = "data/train_transaction.csv",
    identity_path: str = "data/train_identity.csv",
    split_mode: str = "temporal",          # "temporal" | "stratified"
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
    force_reload: bool = False,
):
    """
    Loads and preprocesses the IEEE-CIS Fraud Detection dataset.

    Parameters
    ----------
    split_mode : str
        "temporal"   — chronological split on TransactionDT (default, recommended
                        for publication; prevents lookahead bias).
        "stratified" — random stratified split (legacy; used for ablations).
    train_frac  : float  — fraction of data for training (temporal mode only).
    val_frac    : float  — fraction of data for validation (temporal mode only).
    random_state: int    — seed for stratified split (ignored in temporal mode).
    force_reload: bool   — bypass cache and re-preprocess from raw CSV.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray  — scaled feature arrays
    y_train, y_val, y_test : np.ndarray  — integer label arrays (0/1)
    scaler                  : StandardScaler-compatible object
    """
    path = resolve_path(path)
    identity_path = resolve_path(identity_path)

    # ── cache (mode-aware) ───────────────────────────────────────────────────
    cache_suffix = split_mode
    cache_file = resolve_path(f"data/preprocessed_cache_{cache_suffix}.npz")
    scaler_file = resolve_path(f"data/scaler_{cache_suffix}.pkl")

    if not force_reload and os.path.exists(cache_file) and os.path.exists(scaler_file):
        print(f"Loading cached preprocessed dataset ({split_mode} split)...")
        data_cache = np.load(cache_file)
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        return (
            data_cache["X_train"], data_cache["X_val"], data_cache["X_test"],
            data_cache["y_train"], data_cache["y_val"], data_cache["y_test"],
            scaler,
        )

    # ── load raw CSV in memory-efficient chunks ───────────────────────────────
    import gc

    print(f"Loading {path} in chunks...")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=50_000):
        chunk[chunk.select_dtypes(include=[np.float64]).columns] = (
            chunk.select_dtypes(include=[np.float64]).astype(np.float32)
        )
        chunk[chunk.select_dtypes(include=[np.int64]).columns] = (
            chunk.select_dtypes(include=[np.int64]).astype(np.int32)
        )
        chunks.append(chunk)
    data = pd.concat(chunks, axis=0)
    del chunks
    gc.collect()

    # ── merge identity file ───────────────────────────────────────────────────
    if os.path.exists(identity_path):
        print(f"Loading identity data from {identity_path}...")
        id_chunks = []
        for chunk in pd.read_csv(identity_path, chunksize=50_000):
            chunk[chunk.select_dtypes(include=[np.float64]).columns] = (
                chunk.select_dtypes(include=[np.float64]).astype(np.float32)
            )
            id_chunks.append(chunk)
        identity = pd.concat(id_chunks, axis=0)
        del id_chunks
        gc.collect()

        data = pd.merge(data, identity, on="TransactionID", how="left")
        del identity
        gc.collect()

        # Re-downcast after left join (NaNs cause upcasting)
        data[data.select_dtypes(include=[np.float64]).columns] = (
            data.select_dtypes(include=[np.float64]).astype(np.float32)
        )
    else:
        print("Identity file not found. Proceeding with transaction data only.")

    # ── Perception Agent enrichment ───────────────────────────────────────────
    perception_agent = PerceptionAgent()
    data = perception_agent.enrich_features(data)

    # ── target + feature extraction ───────────────────────────────────────────
    y = data["isFraud"].values.astype(np.int64)

    # Preserve TransactionDT for temporal split before dropping
    dt_values = data["TransactionDT"].values.copy() if "TransactionDT" in data.columns else None

    cols_to_drop = ["isFraud", "TransactionID", "TransactionDT"]
    X_df = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    del data
    gc.collect()

    print("Preprocessing categorical features and handling NaNs...")
    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        X_df[col] = X_df[col].astype(str).fillna("unknown")
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col])

    X_df = X_df.astype(np.float32)
    X_df.fillna(-999.0, inplace=True)
    X = X_df.values
    del X_df
    gc.collect()

    # ── split ─────────────────────────────────────────────────────────────────
    if split_mode == "temporal":
        if dt_values is None:
            print("WARNING: TransactionDT not found; falling back to stratified split.")
            split_mode = "stratified"
        else:
            print(f"Applying temporal split (train={train_frac:.0%}, val={val_frac:.0%}, "
                  f"test={(1-train_frac-val_frac):.0%})...")
            X_train, X_val, X_test, y_train, y_val, y_test, boundaries = _temporal_split(
                X, y, dt_values, train_frac=train_frac, val_frac=val_frac
            )
            print("  Temporal boundaries:")
            print(f"    Train DT : {boundaries['train_dt_min']} → {boundaries['train_dt_max']} "
                  f"({boundaries['n_fraud_train']} fraud)")
            print(f"    Val   DT : {boundaries['val_dt_min']} → {boundaries['val_dt_max']} "
                  f"({boundaries['n_fraud_val']} fraud)")
            print(f"    Test  DT : {boundaries['test_dt_min']} → {boundaries['test_dt_max']} "
                  f"({boundaries['n_fraud_test']} fraud)")

    if split_mode == "stratified":
        print(f"Applying stratified split (70 / 15 / 15, random_state={random_state})...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
        )

    # ── feature scaling (column-wise, float32 to avoid OOM) ──────────────────
    print("Scaling features...")
    num_features = X_train.shape[1]
    mean = np.zeros(num_features, dtype=np.float32)
    std = np.zeros(num_features, dtype=np.float32)

    for i in range(num_features):
        col = X_train[:, i]
        mean[i] = col.mean()
        s = col.std()
        std[i] = s if s != 0.0 else 1.0

    for i in range(num_features):
        X_train[:, i] = (X_train[:, i] - mean[i]) / std[i]
        X_val[:, i]   = (X_val[:, i]   - mean[i]) / std[i]
        X_test[:, i]  = (X_test[:, i]  - mean[i]) / std[i]

    # Build a StandardScaler-compatible object for API compatibility
    scaler = StandardScaler()
    scaler.mean_            = mean.astype(np.float64)
    scaler.var_             = (std ** 2).astype(np.float64)
    scaler.scale_           = std.astype(np.float64)
    scaler.n_samples_seen_  = len(X_train)

    # ── cache ─────────────────────────────────────────────────────────────────
    os.makedirs(resolve_path("data"), exist_ok=True)
    print("Saving cache...")
    np.savez_compressed(
        cache_file,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Dataset loaded. Train={len(X_train)}, Val={len(X_val)}, "
          f"Test={len(X_test)}, Features={num_features}")
    print(f"  Fraud in train: {y_train.sum()} ({y_train.mean():.4%})")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


if __name__ == "__main__":
    # Default: temporal split
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(split_mode="temporal")
    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))
    print("Fraud samples in train:", sum(y_train))
    print("Features:", X_train.shape[1])