def calculate_dynamic_alpha(system_load_ratio, base_alpha=0.5, max_alpha=2.0):
    """
    Calculates a dynamic alpha penalty based on the current human review queue load.
    
    Args:
        system_load_ratio (float): Ratio of system load (0.0 to 1.0).
        base_alpha (float): Base penalty when system load is 0.
        max_alpha (float): Maximum penalty when system load is 1.0.
    """
    # Exponential scaling: penalty shoots up as load approaches 1.0
    return base_alpha + (system_load_ratio ** 2) * (max_alpha - base_alpha)

def calculate_reward(true_label, predicted_label, system_load_ratio=0.0):
    """
    Calculates the reward for the Decision Agent.
    
    Args:
        true_label (int): Ground truth label (0 or 1).
        predicted_label (int): Model prediction (0, 1, or 2 for Abstain).
        alpha (float): Penalty for abstaining. Higher alpha = less abstention.
        
    Returns:
        reward (float): The calculated reward.
    """
    # Dynamic penalty based on load
    alpha = calculate_dynamic_alpha(system_load_ratio)
    
    # Action = 2 means Abstain
    if predicted_label == 2:
        return -alpha
        
    # Correct prediction
    if true_label == predicted_label:
        return +1.0
        
    # Incorrect prediction (critical error if fraud is missed or legit blocked)
    # Different costs can be applied: False Negative (missed fraud) usually costs more
    if true_label == 1 and predicted_label == 0:
        # Missed Fraud
        return -5.0
    elif true_label == 0 and predicted_label == 1:
        # False Positive (blocked legit)
        return -2.0
        
    return -1.0 # Fallback
