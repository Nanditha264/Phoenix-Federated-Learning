"""
Model Package for Federated Learning Benchmark.
"""

from src.models.cnn import FedAvgCNN
from src.models.model_factory import create_model

__all__ = ["FedAvgCNN", "create_model"]
