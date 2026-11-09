"""
Deterministic Seed Management for Federated Learning Benchmark.
Provides unified seeding for Python standard library, NumPy, and PyTorch CPU.
"""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_algorithms: bool = False) -> None:
    """
    Set seeds across all random number generators for bitwise reproducibility.

    Args:
        seed: Integer random seed (e.g. 42).
        deterministic_algorithms: Whether to enforce deterministic PyTorch operations.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some operations on certain CPU platforms may not have deterministic implementations
            pass


def get_worker_init_fn(seed: int):
    """
    Return a worker initialization function for PyTorch DataLoaders.

    Args:
        seed: Base random seed.
    """
    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _worker_init_fn
