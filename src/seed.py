"""
Centralized reproducibility utilities.

Sets all random seeds for PyTorch, NumPy, and Python's random module
to ensure deterministic training and evaluation.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """
    Set random seeds for full reproducibility.

    Args:
        seed: Integer seed value. Default 42.

    Sets seeds for:
        - Python's built-in `random` module
        - NumPy's random generator
        - PyTorch CPU and CUDA generators
        - cuDNN deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may reduce GPU performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For some PyTorch operations that have non-deterministic implementations
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[SEED] All random seeds set to {seed}")
