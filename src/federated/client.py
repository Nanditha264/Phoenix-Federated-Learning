"""
Base Federated Client Architecture.
Defines common client interfaces, local data handling, and update extraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.metrics.update_geometry import compute_client_update


class FederatedClient(ABC):
    """
    Abstract base class for federated clients.
    Supports sequential execution, local training, and parameter update extraction.
    """

    def __init__(
        self,
        cid: int,
        dataset: Dataset,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        local_epochs: int = 2,
        device: Optional[torch.device] = None,
        is_noisy: bool = False,
    ):
        self.cid = cid
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.device = device if device is not None else torch.device("cpu")
        self.is_noisy = is_noisy

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )

    @abstractmethod
    def train(
        self,
        global_model: nn.Module
    ) -> Tuple[nn.Module, torch.Tensor, float]:
        """
        Execute local client training initialized from global_model.

        Args:
            global_model: The current global model from the server.

        Returns:
            Tuple of:
              - local_model: Locally updated model.
              - client_update: Flattened parameter displacement vector (Delta_k = w_k - w_global).
              - avg_loss: Average local training loss over epochs.
        """
        pass
