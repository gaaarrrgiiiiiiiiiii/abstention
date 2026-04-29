import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

# Resolve project root (parent of src/) regardless of CWD
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def load_data(path):

    # Resolve path relative to project root if not absolute
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)

    data = pd.read_csv(path)

    X = data.drop("Class", axis=1).values
    y = data["Class"].values

    # ---------------------------
    # Train / Validation / Test split
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
    # Feature Scaling
    # Fit only on training data
    # ---------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data("data/creditcard.csv")

    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))

    print("Fraud samples in train:", sum(y_train))