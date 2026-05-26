import torch
import numpy as np

def enable_dropout(model):
    """
    Enables dropout layers during inference for Monte Carlo Dropout.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_inference(model, X, num_passes=10):
    """
    Performs Monte Carlo Dropout inference.
    
    Args:
        model: The trained neural network model.
        X (torch.Tensor): Input data tensor.
        num_passes (int): Number of stochastic forward passes.
        
    Returns:
        mean_probs (np.ndarray): Mean probabilities across passes.
        uncertainty (np.ndarray): Predictive variance (epistemic uncertainty).
    """
    model.eval()
    enable_dropout(model)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs.unsqueeze(0))
            
    # Stack predictions: shape (num_passes, batch_size, num_classes)
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    
    # Calculate mean and variance
    mean_probs = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0)
    
    # Total epistemic uncertainty (sum of variances across classes)
    uncertainty = np.sum(variance, axis=1)
    
    return mean_probs, uncertainty
