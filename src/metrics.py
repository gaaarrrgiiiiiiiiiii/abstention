import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classification_metrics(y_true, y_pred):
    """Calculate standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
    recall = recall_score(y_true, y_pred, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
    f1 = f1_score(y_true, y_pred, labels=[0, 1], pos_label=1, average='binary', zero_division=0.0)
    return acc, precision, recall, f1


def coverage(y_pred):
    """
    Calculate coverage: the fraction of predictions where the model did not abstain.
    y_pred: list/array of predicted classes (0=legit, 1=fraud, 2=abstain)
    """
    if len(y_pred) == 0:
        return 0.0
    
    pred_count = sum(p != 2 for p in y_pred)
    return pred_count / len(y_pred)


def selective_risk(y_true, y_pred):
    """
    Calculate selective risk: the error rate on the non-abstained predictions.
    y_true: true labels
    y_pred: predicted classes (0, 1, or 2 for abstain)
    """
    non_abstained = [p != 2 for p in y_pred]
    
    if sum(non_abstained) == 0:
        return 1.0  # 100% risk if no predictions were made
        
    filtered_true = [t for t, valid in zip(y_true, non_abstained) if valid]
    filtered_pred = [p for p, valid in zip(y_pred, non_abstained) if valid]
    
    # Risk is 1 - accuracy
    return 1.0 - accuracy_score(filtered_true, filtered_pred)


def expected_calibration_error(y_true, probs, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    y_true: true labels
    probs: predicted probabilities for the predicted class
    n_bins: number of bins to divide the [0, 1] probability range
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        
        # Find entries falling into the current bin
        in_bin = np.logical_and(probs > bin_lower, probs <= bin_upper)
        prob_in_bin = in_bin.mean()
        
        if prob_in_bin > 0:
            
            # Accuracy of bin
            accuracy_in_bin = y_true[in_bin].mean()
            
            # Average confidence of bin
            avg_confidence_in_bin = probs[in_bin].mean()
            
            # Add absolute difference weighted by bin size
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
            
    return ece


def expected_harm(y_true, y_pred, alpha=0.5):
    """
    Calculate expected harm based on cost matrix.
    0 harm for correct prediction
    1 harm for incorrect prediction
    alpha harm for abstaining
    """
    harm = 0
    for t, p in zip(y_true, y_pred):
        if p == 2:
            harm += alpha
        elif p != t:
            harm += 1
            
    return harm / len(y_true)