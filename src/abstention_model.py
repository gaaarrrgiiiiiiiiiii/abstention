import torch.nn as nn


class AbstentionModel(nn.Module):
    """MLP with 3 output classes: Legitimate (0), Fraud (1), Abstain (2)."""

    def __init__(self, input_dim, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 3)  # 0=legit, 1=fraud, 2=abstain
        )

    def forward(self, x):
        return self.net(x)