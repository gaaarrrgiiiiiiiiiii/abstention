import pandas as pd
import numpy as np

class PerceptionAgent:
    """
    The Perception Agent calculates rolling behavioral profiles for entities (like users or devices).
    For IEEE-CIS, we use a combination of card details and address as a proxy for the 'User'.
    """
    
    def __init__(self, time_window_days=7):
        self.time_window_days = time_window_days
        self.history = pd.DataFrame()
        
    def _create_user_proxy(self, df):
        """
        Creates a UserID proxy since IEEE-CIS doesn't have explicit user IDs.
        Common approach: combine card1-card6, addr1.
        """
        return df['card1'].astype(str) + '-' + \
               df['card2'].astype(str) + '-' + \
               df['card3'].astype(str) + '-' + \
               df['card4'].astype(str) + '-' + \
               df['card5'].astype(str) + '-' + \
               df['card6'].astype(str) + '-' + \
               df['addr1'].astype(str)

    def enrich_features(self, df):
        """
        Enriches the input DataFrame with rolling behavioral statistics.
        Expects IEEE-CIS transaction format.
        """
        print("Perception Agent: Enriching features with behavioral profiles (leakage-free & memory-efficient)...")
        
        # Ensure TransactionDT is present for rolling windows
        if 'TransactionDT' not in df.columns:
            print("Warning: TransactionDT not found. Cannot compute rolling windows properly.")
            return df
            
        # Extract only columns needed for computation to avoid copying 390+ columns in memory
        calc_df = pd.DataFrame({
            'TransactionDT': df['TransactionDT'],
            'TransactionAmt': df['TransactionAmt']
        })
        
        # Create UserID proxy on calc_df
        calc_df['UserID'] = self._create_user_proxy(df)
        
        # Store original index and sort chronologically
        orig_index = calc_df.index
        calc_df = calc_df.sort_values('TransactionDT')

        # Compute Transaction count per user (causal)
        calc_df['User_Txn_Count'] = calc_df.groupby('UserID').cumcount() + 1
        
        # Compute average TransactionAmt per user (causal)
        cum_amt = calc_df.groupby('UserID')['TransactionAmt'].cumsum()
        calc_df['User_Avg_Amt'] = cum_amt / calc_df['User_Txn_Count']
        
        # Ratio of current transaction amount to user's average
        calc_df['Amt_to_Avg_Ratio'] = calc_df['TransactionAmt'] / (calc_df['User_Avg_Amt'] + 1e-6)
        
        # Restore the original row order
        calc_df = calc_df.reindex(orig_index)
        
        # Assign new columns back to the original df in-place to save memory
        df['User_Txn_Count'] = calc_df['User_Txn_Count'].astype('int32')
        df['User_Avg_Amt'] = calc_df['User_Avg_Amt'].astype('float32')
        df['Amt_to_Avg_Ratio'] = calc_df['Amt_to_Avg_Ratio'].astype('float32')
        
        print(f"Perception Agent: Added {3} new causal behavioral features in-place.")
        return df

if __name__ == "__main__":
    # Simple test with dummy data
    agent = PerceptionAgent()
    dummy_data = pd.DataFrame({
        'TransactionDT': [86400, 172800, 259200],
        'TransactionAmt': [100.0, 150.0, 50.0],
        'card1': [1, 1, 2],
        'card2': [1, 1, 2],
        'card3': [1, 1, 2],
        'card4': [1, 1, 2],
        'card5': [1, 1, 2],
        'card6': [1, 1, 2],
        'addr1': [1, 1, 2]
    })
    
    enriched = agent.enrich_features(dummy_data)
    print("Enriched Columns:", enriched.columns.tolist())
    print(enriched[['User_Txn_Count', 'User_Avg_Amt', 'Amt_to_Avg_Ratio']])
