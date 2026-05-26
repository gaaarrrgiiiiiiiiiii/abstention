import torch
import numpy as np

def ensemble_inference(models, X):
    """
    Performs inference using a Deep Ensemble.
    
    Args:
        models (list): List of trained PyTorch models.
        X (torch.Tensor): Input data tensor.
        
    Returns:
        mean_probs (np.ndarray): Mean probabilities across ensemble members.
        disagreement (np.ndarray): Variance across ensemble predictions (model disagreement).
    """
    predictions = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs.unsqueeze(0))
            
    # Stack predictions: shape (num_models, batch_size, num_classes)
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    
    # Calculate mean and variance
    mean_probs = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0)
    
    # Total disagreement (sum of variances across classes)
    disagreement = np.sum(variance, axis=1)
    
    return mean_probs, disagreement
