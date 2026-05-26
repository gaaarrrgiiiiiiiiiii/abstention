import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Resolve project root (parent of src/) regardless of CWD
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(PROJECT_ROOT)
from agents.perception_agent import PerceptionAgent


def resolve_path(relative_path):
    """Resolve a project-relative path to an absolute path."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


class FraudDataset(Dataset):

    def __init__(self, X, y):
        # convert to tensor
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(path="data/train_transaction.csv", identity_path="data/train_identity.csv"):
    """
    Loads and preprocesses the IEEE-CIS Fraud Detection dataset.
    If the identity file is found, it will be merged.
    """
    path = resolve_path(path)
    identity_path = resolve_path(identity_path)

    # Cache paths
    cache_file = resolve_path("data/preprocessed_cache.npz")
    scaler_file = resolve_path("data/scaler.pkl")
    import pickle
    
    if os.path.exists(cache_file) and os.path.exists(scaler_file):
        print("Loading cached preprocessed dataset...")
        data_cache = np.load(cache_file)
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        return (data_cache['X_train'], data_cache['X_val'], data_cache['X_test'],
                data_cache['y_train'], data_cache['y_val'], data_cache['y_test'], scaler)

    # Load data in chunks to prevent memory error
    chunks = []
    for chunk in pd.read_csv(path, chunksize=50000):
        float_cols = chunk.select_dtypes(include=[np.float64]).columns
        chunk[float_cols] = chunk[float_cols].astype(np.float32)
        int_cols = chunk.select_dtypes(include=[np.int64]).columns
        chunk[int_cols] = chunk[int_cols].astype(np.int32)
        chunks.append(chunk)
    data = pd.concat(chunks, axis=0)
    del chunks
    import gc
    gc.collect()

    if os.path.exists(identity_path):
        print(f"Loading identity data from {identity_path}...")
        identity_chunks = []
        for chunk in pd.read_csv(identity_path, chunksize=50000):
            float_cols = chunk.select_dtypes(include=[np.float64]).columns
            chunk[float_cols] = chunk[float_cols].astype(np.float32)
            int_cols = chunk.select_dtypes(include=[np.int64]).columns
            chunk[int_cols] = chunk[int_cols].astype(np.int32)
            identity_chunks.append(chunk)
        identity = pd.concat(identity_chunks, axis=0)
        del identity_chunks
        gc.collect()
        
        merged_data = pd.merge(data, identity, on='TransactionID', how='left')
        del data, identity
        gc.collect()
        
        data = merged_data
        
        # Downcast again because merge might upcast to float64 due to left-join NaNs
        float_cols = data.select_dtypes(include=[np.float64]).columns
        data[float_cols] = data[float_cols].astype(np.float32)
        int_cols = data.select_dtypes(include=[np.int64]).columns
        data[int_cols] = data[int_cols].astype(np.int32)
    else:
        print("Identity file not found. Proceeding with transaction data only.")

    # Apply Perception Agent for Behavioral Enrichment
    perception_agent = PerceptionAgent()
    data = perception_agent.enrich_features(data)

    # Target variable
    y = data['isFraud'].values
    
    # Drop irrelevant or unscalable columns
    cols_to_drop = ['isFraud', 'TransactionID', 'TransactionDT']
    X_df = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    print("Preprocessing categorical features and handling NaNs...")
    # Encode categorical columns
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X_df[col] = X_df[col].astype(str).fillna('unknown')
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col])

    # Convert everything to float32 first to consolidate types and free memory
    X_df = X_df.astype(np.float32)
    # Fill NaNs in-place on float32 DataFrame
    X_df.fillna(-999.0, inplace=True)

    X = X_df.values
    import gc
    del X_df
    gc.collect()

    # ---------------------------
    # Train / Validation / Test split
    # For IEEE-CIS, a time-based split is usually preferred, but for 
    # consistency with our previous random split, we continue with stratify.
    # ---------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    # ---------------------------
    # Feature Scaling (column-wise float32 to prevent OOM)
    # ---------------------------
    num_features = X_train.shape[1]
    mean = np.zeros(num_features, dtype=np.float32)
    std = np.zeros(num_features, dtype=np.float32)
    
    # Compute column-wise mean and std
    for i in range(num_features):
        col = X_train[:, i]
        mean[i] = col.mean()
        s = col.std()
        std[i] = s if s != 0.0 else 1.0
        
    # Scale column-wise in-place
    for i in range(num_features):
        m = mean[i]
        s = std[i]
        X_train[:, i] = (X_train[:, i] - m) / s
        X_val[:, i] = (X_val[:, i] - m) / s
        X_test[:, i] = (X_test[:, i] - m) / s
    
    # Populate StandardScaler attributes to keep it compatible
    scaler = StandardScaler()
    scaler.mean_ = mean.astype(np.float64)
    scaler.var_ = (std ** 2).astype(np.float64)
    scaler.scale_ = std.astype(np.float64)
    scaler.n_samples_seen_ = len(X_train)

    print("Saving cache...")
    np.savez_compressed(cache_file, 
             X_train=X_train, X_val=X_val, X_test=X_test,
             y_train=y_train, y_val=y_val, y_test=y_test)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data()

    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))

    print("Fraud samples in train:", sum(y_train))