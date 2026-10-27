import os
import random
import numpy as np

def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed across Python, NumPy, and system environment
    to ensure 100% reproducibility as required by Zindi rules.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    print(f"[INFO] Global random seed set to: {seed}")
