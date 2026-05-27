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

    # TransactionDT unit: seconds since a reference date (Vesta-specific epoch)
    _SECS_PER_DAY = 86_400

    def enrich_features(self, df):
        """
        Enriches the input DataFrame with behavioural profile features.

        Produces two families of features:
        1. Cumulative (causal, leakage-free): cumulative count, running avg amount,
           ratio of current amount to running average.
        2. Sliding-window (causal): 24-hour transaction count, 7-day amount mean
           and std, 30-day transaction velocity — capturing recency effects that
           cumulative statistics miss.

        All windows are computed in forward-time order (sorted by TransactionDT)
        to guarantee zero lookahead bias, then re-indexed to the original row order.
        """
        print("Perception Agent: Enriching features with behavioural profiles "
              "(cumulative + sliding window, leakage-free)...")

        if 'TransactionDT' not in df.columns:
            print("Warning: TransactionDT not found. Cannot compute rolling windows.")
            return df

        # Work on a lightweight copy — avoids copying 400+ raw columns
        calc_df = pd.DataFrame({
            'TransactionDT':  df['TransactionDT'].values.copy(),
            'TransactionAmt': df['TransactionAmt'].values.copy(),
        })
        calc_df['UserID'] = self._create_user_proxy(df)

        orig_index = calc_df.index.copy()
        calc_df = calc_df.sort_values('TransactionDT').reset_index(drop=False)
        # 'index' column now holds the original positional index

        # ── 1. Cumulative features ────────────────────────────────────────────
        calc_df['User_Txn_Count'] = (
            calc_df.groupby('UserID').cumcount() + 1
        ).astype('int32')

        cum_amt = calc_df.groupby('UserID')['TransactionAmt'].cumsum()
        calc_df['User_Avg_Amt'] = (cum_amt / calc_df['User_Txn_Count']).astype('float32')

        calc_df['Amt_to_Avg_Ratio'] = (
            calc_df['TransactionAmt'] / (calc_df['User_Avg_Amt'] + 1e-6)
        ).astype('float32')

        # ── 2. Sliding-window features ────────────────────────────────────────
        # Convert DT to datetime for pandas rolling (uses seconds resolution)
        calc_df['dt_fake'] = pd.to_datetime(calc_df['TransactionDT'], unit='s')
        calc_df = calc_df.set_index('dt_fake')

        def _rolling_user(grp, window_str, col, agg):
            """Apply a time-based rolling aggregation per user group."""
            return (
                grp[col]
                .rolling(window=window_str, min_periods=1)
                .agg(agg)
                .astype('float32')
            )

        # 24-hour transaction count per user (velocity)
        calc_df['User_Txn_24h'] = (
            calc_df.groupby('UserID', group_keys=False)
            .apply(lambda g: _rolling_user(g, '24h', 'TransactionAmt', 'count'))
        ).astype('int32')

        # 7-day amount mean per user
        calc_df['User_Amt_Mean_7d'] = (
            calc_df.groupby('UserID', group_keys=False)
            .apply(lambda g: _rolling_user(g, '7D', 'TransactionAmt', 'mean'))
        ).astype('float32')

        # 7-day amount std per user (volatility)
        calc_df['User_Amt_Std_7d'] = (
            calc_df.groupby('UserID', group_keys=False)
            .apply(lambda g: _rolling_user(g, '7D', 'TransactionAmt', 'std'))
            .fillna(0.0)
        ).astype('float32')

        # 30-day transaction velocity per user
        calc_df['User_Txn_30d'] = (
            calc_df.groupby('UserID', group_keys=False)
            .apply(lambda g: _rolling_user(g, '30D', 'TransactionAmt', 'count'))
        ).astype('int32')

        # Restore original row order
        calc_df = calc_df.reset_index().set_index('index')
        calc_df = calc_df.reindex(orig_index)

        new_cols = [
            'User_Txn_Count', 'User_Avg_Amt', 'Amt_to_Avg_Ratio',
            'User_Txn_24h', 'User_Amt_Mean_7d', 'User_Amt_Std_7d', 'User_Txn_30d',
        ]
        for col in new_cols:
            df[col] = calc_df[col].values

        print(f"Perception Agent: Added {len(new_cols)} causal behavioural features "
              f"(3 cumulative + 4 sliding-window).")
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
