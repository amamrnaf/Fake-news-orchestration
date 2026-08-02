"""Deterministic seeding across python/numpy/torch (Sec. 4.4: 'reported ... random seeds')."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


SEEDS = [13, 42, 2024]  # >= 3 seeds reported for trained components (Sec 4.4 / checklist)
