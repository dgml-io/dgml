"""Deterministic seeding for all stochastic backends used in the framework."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic_cudnn: bool = False) -> None:
    """Seed Python, NumPy, PyTorch (CPU + all CUDA devices), and the hash seed.

    Args:
        seed: Integer seed shared across all backends.
        deterministic_cudnn: If True, force cuDNN into deterministic mode at
            the cost of throughput. Turn this on for reproducibility-critical
            runs; leave off for fast research iteration.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
