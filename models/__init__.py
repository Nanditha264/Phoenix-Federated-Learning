"""
Model Package for Federated Learning Benchmark.
"""

from fl_benchmark.models.cnn import FedAvgCNN
from fl_benchmark.models.model_factory import create_model

__all__ = ["FedAvgCNN", "create_model"]
