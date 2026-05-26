import numpy as np
import pandas as pd

def inject_ood_transactions(X_test, y_test, num_ood=1000):
    """
    Synthesizes Out-of-Distribution (OOD) transactions by sampling far outside 
    the normal feature distributions and injects them into the test set.
    This helps validate if the Uncertainty Agent and DAC can successfully 
    identify and abstain on completely unseen anomalies.
    
    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        num_ood (int): Number of OOD samples to inject.
        
    Returns:
        X_test_ood, y_test_ood
    """
    print(f"Injecting {num_ood} Out-of-Distribution (OOD) transactions for robustness testing...")
    
    # Generate OOD features by sampling from a uniform distribution 
    # far outside the standard normal ranges of PCA components (e.g., -50 to 50)
    ood_features = np.random.uniform(low=-50, high=50, size=(num_ood, X_test.shape[1]))
    
    # If the dataset has specific column names, map them
    if isinstance(X_test, pd.DataFrame):
        ood_df = pd.DataFrame(ood_features, columns=X_test.columns)
        X_test_ood = pd.concat([X_test, ood_df], ignore_index=True)
    else:
        # Assuming numpy arrays
        X_test_ood = np.vstack((X_test, ood_features))
        
    # We assign y_test = -1 (or 2) to distinctly track OOD samples in evaluation
    if isinstance(y_test, pd.Series):
        y_ood = pd.Series([-1] * num_ood)
        y_test_ood = pd.concat([y_test, y_ood], ignore_index=True)
    else:
        y_ood = np.array([-1] * num_ood)
        y_test_ood = np.concatenate((y_test, y_ood))
        
    return X_test_ood, y_test_ood
