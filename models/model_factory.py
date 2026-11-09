"""
Model Factory for Federated Learning Benchmark.
Provides unified model instantiation and validation.
"""

from typing import Dict, Any
import torch.nn as nn
from fl_benchmark.models.cnn import FedAvgCNN


def create_model(
    model_name: str = "FedAvgCNN",
    in_channels: int = 3,
    num_classes: int = 9,
    fc_hidden: int = 1024,
) -> nn.Module:
    """
    Factory function to instantiate models for the benchmark.
    """
    normalized_name = model_name.lower()
    if normalized_name in {"fedavgcnn", "cnn"}:
        return FedAvgCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dim=fc_hidden,
        )
    else:
        raise ValueError(f"Unsupported model architecture: '{model_name}'")
