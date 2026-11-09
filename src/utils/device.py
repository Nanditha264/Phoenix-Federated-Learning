"""
Device and CPU Thread Management for Federated Learning Benchmark.
Optimized for multi-core Intel architectures (e.g. Alder Lake i5-1240P).
"""

import os
import psutil
from typing import Dict, Any
import torch


def setup_device(num_threads: int = 4, num_interop_threads: int = 1) -> torch.device:
    """
    Configure PyTorch computation device and intra/inter-op thread counts.

    Args:
        num_threads: Number of intra-op OpenMP/MKL compute threads (default: 4).
        num_interop_threads: Number of inter-op threads (default: 1 for sequential execution).

    Returns:
        torch.device("cpu") instance.
    """
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if num_interop_threads > 0:
        try:
            torch.set_num_interop_threads(num_interop_threads)
        except RuntimeError:
            # Cannot change interop threads after parallel work has already started
            pass

    return torch.device("cpu")


def get_cpu_info() -> Dict[str, Any]:
    """
    Retrieve host CPU topology and active PyTorch thread settings.
    """
    return {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkl_available": torch.backends.mkl.is_available(),
        "openmp_available": torch.backends.openmp.is_available(),
    }
