import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    """
    The Gating Network learns to combine the multi-source uncertainty signals
    and the original transaction features to make a final abstention decision.
    
    Instead of a hard threshold, this provides a learned, differentiable path
    to decide whether to abstain.
    """
    
    def __init__(self, feature_dim, uncertainty_dim=3, hidden_dim=32):
        super(GatingNetwork, self).__init__()
        
        # We concatenate original features with the uncertainty features
        # (Aleatoric, Epistemic, Verbalized)
        input_dim = feature_dim + uncertainty_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            # Output is a single logit representing the propensity to abstain
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, features, uncertainties):
        """
        Args:
            features (torch.Tensor): Original transaction features (batch_size, feature_dim).
            uncertainties (torch.Tensor): Uncertainty scores (batch_size, uncertainty_dim).
            
        Returns:
            abstain_prob (torch.Tensor): Probability of abstaining (batch_size, 1).
        """
        # Concatenate features and uncertainties
        x = torch.cat([features, uncertainties], dim=1)
        
        logits = self.net(x)
        abstain_prob = torch.sigmoid(logits)
        
        return abstain_prob
