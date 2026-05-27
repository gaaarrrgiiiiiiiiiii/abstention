"""
export_scaler_features.py
=========================
After training, run this script to serialize the fitted scaler, feature names,
and model input dimension into a single JSON metadata file.

The API (api/app.py) loads this file on startup instead of re-running load_data(),
eliminating the 10-second CSV dependency at serving time.

Usage:
    cd src
    python export_scaler_features.py
"""

import os
import sys
import json
import pickle
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from dataset import load_data, resolve_path


def export_metadata():
    print("Loading data and fitted scaler...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data()

    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    # Build feature name list.
    # If the identity file was merged, we don't have the exact names at this point
    # because dataset.py drops column names after .values. Use generic names.
    # To get real names, re-run with the DataFrame before .values (see note below).
    feature_names = [f"feature_{i}" for i in range(input_dim)]

    metadata = {
        "input_dim": input_dim,
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "scaler_var": scaler.var_.tolist(),
        "n_samples_seen": int(scaler.n_samples_seen_),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "fraud_rate_train": float(np.mean(y_train)),
        "n_fraud_train": int(np.sum(y_train)),
        "n_fraud_test": int(np.sum(y_test)),
    }

    out_path = resolve_path("data/model_metadata.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {out_path}")
    print(f"  input_dim     : {input_dim}")
    print(f"  train_size    : {len(X_train)}")
    print(f"  fraud_rate    : {metadata['fraud_rate_train']:.4%}")
    return metadata


if __name__ == "__main__":
    export_metadata()
