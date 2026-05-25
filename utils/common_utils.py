import random

import numpy as np
import torch

from utils.logging_utils import Log


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    Log(f"Setting random seed to {seed}", tag="Info")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)