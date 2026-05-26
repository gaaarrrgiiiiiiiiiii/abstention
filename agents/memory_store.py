import pandas as pd
import numpy as np

class MemoryStore:
    """
    The Uncertainty-Aware Memory Store.
    Stores transactions that were abstained on, pending human review.
    Implements Thompson Sampling to decide which transactions to prioritize for review.
    """
    
    def __init__(self, review_capacity=100):
        self.memory = pd.DataFrame()
        self.review_capacity = review_capacity # How many we can review per batch
        
    def add_transactions(self, features_df, uncertainties, predictions):
        """
        Adds abstained transactions to memory.
        """
        df = features_df.copy()
        df['Uncertainty'] = uncertainties
        df['Prediction'] = predictions
        df['Review_Status'] = 'Pending'
        
        # In a real system, you'd append to a database.
        self.memory = pd.concat([self.memory, df], ignore_index=True)
        print(f"Memory Store: Added {len(df)} transactions. Total pending: {len(self.memory[self.memory['Review_Status'] == 'Pending'])}")
        
    def get_batch_for_review(self):
        """
        Uses Thompson Sampling inspired logic to select transactions.
        We want to prioritize high-uncertainty transactions.
        """
        pending = self.memory[self.memory['Review_Status'] == 'Pending'].copy()
        if len(pending) == 0:
            return pd.DataFrame()
            
        # We sample based on uncertainty (higher uncertainty = higher chance of being picked)
        probs = pending['Uncertainty'] / pending['Uncertainty'].sum()
        
        n_select = min(self.review_capacity, len(pending))
        
        selected_indices = np.random.choice(pending.index, size=n_select, replace=False, p=probs)
        
        return self.memory.loc[selected_indices]
        
    def update_reviewed_batch(self, indices, human_labels):
        """
        Updates the memory with the ground truth provided by human reviewers.
        """
        self.memory.loc[indices, 'Review_Status'] = 'Reviewed'
        self.memory.loc[indices, 'Human_Label'] = human_labels
        
        # Return the reviewed data for retraining
        return self.memory.loc[indices]
