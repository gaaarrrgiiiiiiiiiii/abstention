import torch
import numpy as np

class ConformalPredictor:
    """
    Implements Split Conformal Prediction for classification.
    Provides mathematically guaranteed marginal coverage.
    """
    
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha (float): Target error rate. 1 - alpha is the target coverage.
                           e.g. alpha=0.1 guarantees 90% coverage.
        """
        self.alpha = alpha
        self.q_hat = None

    def calibrate(self, val_smx, val_labels):
        """
        Calibrates the conformal threshold using a validation set.
        
        Args:
            val_smx (torch.Tensor or np.ndarray): Softmax probabilities of the validation set (N, num_classes).
            val_labels (torch.Tensor or np.ndarray): Ground truth labels (N,).
        """
        if isinstance(val_smx, torch.Tensor):
            val_smx = val_smx.detach().cpu().numpy()
        if isinstance(val_labels, torch.Tensor):
            val_labels = val_labels.detach().cpu().numpy()

        N = len(val_labels)
        
        # Get the softmax probability of the true class
        # Using Adaptive Prediction Sets (APS) / THR approach: 
        # Non-conformity score is 1 - p(true_class)
        true_class_probs = val_smx[np.arange(N), val_labels]
        scores = 1.0 - true_class_probs
        
        # We want the (1-alpha)*(1+1/N) quantile of the non-conformity scores
        q_level = np.ceil((N + 1) * (1 - self.alpha)) / N
        if q_level > 1.0:
            q_level = 1.0
            
        self.q_hat = np.quantile(scores, q_level)
        print(f"Conformal Calibration: target_alpha={self.alpha}, N={N}, q_hat={self.q_hat:.4f}")

    def predict_sets(self, test_smx):
        """
        Generates prediction sets for test data based on the calibrated threshold.
        
        Args:
            test_smx (torch.Tensor or np.ndarray): Softmax probabilities of test set (M, num_classes).
            
        Returns:
            prediction_sets (list of lists): For each sample, a list of classes included in the set.
        """
        if self.q_hat is None:
            raise ValueError("Conformal Predictor is not calibrated. Call `calibrate()` first.")
            
        if isinstance(test_smx, torch.Tensor):
            test_smx = test_smx.detach().cpu().numpy()

        prediction_sets = []
        for i in range(len(test_smx)):
            # A class k is included if 1 - p(k) <= q_hat  =>  p(k) >= 1 - q_hat
            included_classes = np.where(test_smx[i] >= (1.0 - self.q_hat))[0].tolist()
            prediction_sets.append(included_classes)
            
        return prediction_sets

    def evaluate_coverage(self, prediction_sets, test_labels):
        """
        Evaluates the empirical marginal coverage of the prediction sets.
        
        Args:
            prediction_sets (list of lists): The output from predict_sets.
            test_labels (torch.Tensor or np.ndarray): Ground truth labels.
            
        Returns:
            coverage (float): The fraction of times the true label is in the prediction set.
            avg_set_size (float): The average size of the prediction sets.
        """
        if isinstance(test_labels, torch.Tensor):
            test_labels = test_labels.detach().cpu().numpy()
            
        covered = 0
        set_sizes = []
        for i, true_label in enumerate(test_labels):
            if true_label in prediction_sets[i]:
                covered += 1
            set_sizes.append(len(prediction_sets[i]))
            
        coverage = covered / len(test_labels)
        avg_set_size = np.mean(set_sizes)
        
        return coverage, avg_set_size

def apply_conformal_abstention(val_smx, val_labels, test_smx, alpha=0.1):
    """
    Wrapper for easy usage in evaluation pipelines.
    Returns prediction sets and the confidence threshold.
    """
    cp = ConformalPredictor(alpha=alpha)
    cp.calibrate(val_smx, val_labels)
    prediction_sets = cp.predict_sets(test_smx)
    return prediction_sets, cp
